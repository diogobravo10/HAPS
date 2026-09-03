import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
from ambiance import Atmosphere
from datetime import datetime
import _utilities as utils


# Reference date/time and latitude for the solar model (t=0 of the phase)
start_date = datetime(2012, 6, 1, 12, 0)
lat = 37.5

# Aircraft constants (shared between the ODE and the post-run diagnostic plots)
m, g, S = 1.2, 9.81, 0.1566
K, CD0, CLmax = 0.1, 0.011, 1.5
eta_prop, solar_cell_efficiency = 0.7, 0.15

SOLUTION_DB = 'mission_profile_out/dymos_solution.db'


class SolarAircraftODE(om.ExplicitComponent):
    def initialize(self):
        # Dymos requires num_nodes to vectorize calculations across the trajectory grid
        self.options.declare('num_nodes', types=int)
        self.options.declare('start_date', types=datetime)
        self.options.declare('lat', types=(int, float))

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs: States & Controls
        self.add_input('gg', val=np.ones(nn), units='rad', desc='Angle of Climb (control)')
        self.add_input('V', val=np.ones(nn) * 15.0, units='m/s', desc='Airspeed (control)')
        self.add_input('Vdot', val=np.zeros(nn), units='m/s**2', desc='Airspeed rate (control rate)')
        # Auto-connected by Dymos: absolute phase time and current altitude (state 'h')
        self.add_input('time', val=np.zeros(nn), units='s')
        self.add_input('h', val=np.ones(nn) * 20000.0, units='m')

        # Outputs: Derivatives for Dymos
        self.add_output('dhdt', val=np.zeros(nn), units='m/s', desc='Rate of change of altitude')
        self.add_output('E_dot', val=np.zeros(nn), units='W', desc='Net power accumulation')
        self.add_output('V_margin', val=np.zeros(nn), units='m/s',
                         desc='Airspeed margin above 1.2x stall speed (must stay >= 0)')

        # Setup automatic derivatives via Complex Step or Finite Difference
        self.declare_partials('*', '*', method='fd')
        # The solar model is quantized to whole minutes internally (solarpy's
        # hour_angle only reads date.hour/date.minute), so the default FD
        # step on 'time' (~1e-6 * a few thousand seconds) is far smaller than
        # 60s and would yield a numerically zero d(E_dot)/d(time).
        self.declare_partials('E_dot', 'time', method='fd', step=60.0, step_calc='abs')


    def compute(self, inputs, outputs):

        gg = inputs['gg']
        t = inputs['time']
        h = inputs['h']
        V = inputs['V']
        Vdot = inputs['Vdot']

        rho = Atmosphere(h).density

        # Kinematic Dynamics
        dhdt = V * np.sin(gg)
        outputs['dhdt'] = dhdt

        # Solar Power Input
        irradiance = utils.instantaneous_power_density(h, self.options['lat'], self.options['start_date'], t, solar_cell_efficiency=solar_cell_efficiency)
        P_in = irradiance * S

        # Aerodynamic Drag & Required Thrust Power
        CL = m*g / (0.5 * rho * S * V**2)
        CD = CD0 + K * CL**2
        D = 0.5 * rho * S * V**2 * CD

        # Stall-speed margin: airspeed must stay >= 1.2x stall speed (Raymer 1999)
        V_stall = np.sqrt(2*m*g / (rho * S * CLmax))
        outputs['V_margin'] = V - V_stall

        # Net Power available for battery storage, incl. kinetic energy rate d(V^2/2)/dt = V*Vdot
        outputs['E_dot'] = P_in - D*V - m*g*dhdt - m*V*Vdot
        print("compute(): t = [{:.1f}, {:.1f}] s, h = [{:.1f}, {:.1f}] m, V = [{:.1f}, {:.1f}] m/s, gg = [{:.2f}, {:.2f}] deg, E_dot = [{:.1f}, {:.1f}] J".format(
            t.min(), t.max(), h.min(), h.max(), V.min(), V.max(), np.degrees(gg).min(), np.degrees(gg).max(), outputs['E_dot'].min(), outputs['E_dot'].max()))


def build_problem():
    """Construct and configure the Dymos trajectory-optimization problem (does not run it)."""
    prob = om.Problem()

    traj = dm.Trajectory()
    phase = dm.Phase(
        ode_class=SolarAircraftODE,
        transcription=dm.Radau(num_segments=10, order=3),
        ode_init_kwargs={'start_date': start_date, 'lat': lat},
    )
    traj.add_phase('phase0', phase)
    prob.model.add_subsystem('traj', traj)

    prob.driver = om.ScipyOptimizeDriver()  # Or pyOptSparseDriver(optimizer='IPOPT')
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['maxiter'] = 1500  # Default is often 100
    prob.driver.options['tol'] = 1e-5
    # SLSQP-specific optimizer options (passes directly to scipy.optimize.minimize)
    prob.driver.opt_settings['maxiter'] = 1500
    prob.driver.opt_settings['ftol'] = 1e-5  # Function tolerance for convergence

    # Configure Time Variable
    phase.set_time_options(fix_initial=True, fix_duration=True, units='s')

    # Configure States
    phase.add_state('h', rate_source='dhdt', fix_initial=True, units='m')
    phase.add_state('E', rate_source='E_dot', fix_initial=True, units='J')  # Net energy integral - auxiliary state variable

    phase.add_control('gg', lower=np.radians(-5), upper=np.radians(5), units='rad')

    phase.add_control('V', lower=5.0, upper=40.0, units='m/s', rate_targets=['Vdot'])
    phase.add_path_constraint('V_margin', lower=0.0, units='m/s')

    phase.add_boundary_constraint('h', loc='final', equals=12000.0, units='m')
    phase.add_path_constraint('h', lower=10000.0, units='m')

    # Set Objective: Maximize final accumulated net energy E_total [Eq. (16)]
    phase.add_objective('E', loc='final', scaler=-1e0)

    prob.setup()

    # Set Initial Guesses
    prob.set_val('traj.phase0.t_duration', 4*3600.0)
    prob.set_val('traj.phase0.states:h', phase.interp('h', [20000, 12000]))
    prob.set_val('traj.phase0.states:E', phase.interp('E', [0, 0]))
    prob.set_val('traj.phase0.controls:gg', np.radians(1.0))
    prob.set_val('traj.phase0.controls:V', 15.0)

    return prob


def plot_results(source):
    """Plot mission results from anything exposing .get_val() - a Problem (fresh run) or a Case (CaseReader)."""
    t = source.get_val('traj.phase0.timeseries.time').ravel()
    E = source.get_val('traj.phase0.timeseries.E').ravel()
    h = source.get_val('traj.phase0.timeseries.h').ravel()
    gg = source.get_val('traj.phase0.timeseries.gg').ravel()
    V_flown = source.get_val('traj.phase0.timeseries.V').ravel()

    fig, axes = plt.subplots(1, 3, sharex=True, figsize=(9, 7))

    axes[0].plot(t, h, 'b-', linewidth=2)
    axes[0].set_ylabel('Altitude h [m]')
    axes[0].set_title('Altitude and glide angle vs elapsed time')
    axes[0].grid(True)

    axes[1].plot(t, np.degrees(gg), 'r-', linewidth=2)
    axes[1].set_xlabel('Elapsed time [s]')
    axes[1].set_ylabel('Glide angle gg [deg]')
    axes[1].grid(True)
    axes[1].ticklabel_format(useOffset=False, style='plain')

    axes[2].plot(t, E, 'g-', linewidth=2)
    axes[2].set_xlabel('Elapsed time [s]')
    axes[2].set_ylabel('Net energy E [J]')
    axes[2].grid(True)

    plt.tight_layout()

    # Stall speed vs altitude, compared to the flown airspeed
    rho_h = Atmosphere(h).density
    V_stall_h = np.sqrt(2*m*g / (rho_h * S * CLmax))

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(h, V_stall_h, 'k--', label='V_stall')
    ax2.plot(h, 1.2 * V_stall_h, 'r--', label='1.2 x V_stall (margin threshold)')
    ax2.plot(h, V_flown, 'b-', linewidth=2, label='Flown V')
    ax2.set_xlabel('Altitude h [m]')
    ax2.set_ylabel('Speed [m/s]')
    ax2.set_title('Stall speed vs altitude')
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout()

    plt.show()


RUN_OPTIMIZATION = True  # False -> skip the solve and re-plot the last saved solution instead


if __name__ == '__main__':
    if RUN_OPTIMIZATION:
        prob = build_problem()
        dm.run_problem(prob, run_driver=True, make_plots=True)
        source = prob
    else:
        source = om.CaseReader(SOLUTION_DB).get_case('final')

    plot_results(source)
