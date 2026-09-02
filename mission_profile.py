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
        # Auto-connected by Dymos: absolute phase time and current altitude (state 'h')
        self.add_input('time', val=np.zeros(nn), units='s')
        self.add_input('h', val=np.ones(nn) * 12000.0, units='m')

        # Outputs: Derivatives for Dymos
        self.add_output('dhdt', val=np.zeros(nn), units='m/s', desc='Rate of change of altitude')
        self.add_output('E_dot', val=np.zeros(nn), units='W', desc='Net power accumulation')

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

        # Aircraft Constants - link with solar irradiance after debugging and validating
        m, g, rho, S = 1.2, 9.81, 1.29, 0.1566
        K, CD0 = 0.1, 0.011
        eta_prop, solar_cell_efficiency = 0.7, 0.15
        V = 15.0  # m/s (constant airspeed for this example)

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

        # Net Power available for battery storage
        outputs['E_dot'] = P_in - D*V - m*g*dhdt
        print("compute(): t = [{:.1f}, {:.1f}] s, h = [{:.1f}, {:.1f}] m, gg = [{:.2f}, {:.2f}] deg".format(
            t.min(), t.max(), h.min(), h.max(), np.degrees(gg).min(), np.degrees(gg).max()))



# 1. Create Problem Instance
prob = om.Problem()

# 2. Add Trajectory & Phase
traj = dm.Trajectory()
phase = dm.Phase(
    ode_class=SolarAircraftODE,
    transcription=dm.Radau(num_segments=20, order=3),
    ode_init_kwargs={'start_date': start_date, 'lat': lat},
)
traj.add_phase('phase0', phase)
prob.model.add_subsystem('traj', traj)


prob.driver = om.ScipyOptimizeDriver()  # Or pyOptSparseDriver(optimizer='IPOPT')
prob.driver.options['optimizer'] = 'SLSQP'
# 2. Increase maximum iterations for the driver
prob.driver.options['maxiter'] = 1500  # Default is often 100
prob.driver.options['tol'] = 1e-5
# 3. SLSQP-specific optimizer options (passes directly to scipy.optimize.minimize)
prob.driver.opt_settings['maxiter'] = 1500
prob.driver.opt_settings['ftol'] = 1e-5  # Function tolerance for convergence


# 3. Configure Time Variable
phase.set_time_options(fix_initial=True, duration_bounds=(50, 8*3600), units='s')


# 4. Configure States
phase.add_state('h', rate_source='dhdt', fix_initial=True, units='m')
phase.add_state('E', rate_source='E_dot', fix_initial=True, units='J')  # Net energy integral - auxilary state variable

phase.add_control('gg', lower=np.radians(0), upper=np.radians(5), units='rad')

phase.add_boundary_constraint('h', loc='final', equals=20000.0,  units='m')

# 6. Set Objective: Maximize final accumulated net energy E_total [Eq. (16)]
phase.add_objective('E', loc='final', scaler=-1e0)

# 7. Setup and Run
prob.setup()

# Set Initial Guesses
prob.set_val('traj.phase0.t_duration', 3600.0)
prob.set_val('traj.phase0.states:h', phase.interp('h', [12000, 20000]))
prob.set_val('traj.phase0.states:E', phase.interp('E', [0, 0]))
prob.set_val('traj.phase0.controls:gg', np.radians(1.0))

dm.run_problem(prob, run_driver=True, make_plots=True)

t = prob.get_val('traj.phase0.timeseries.time').ravel()
E = prob.get_val('traj.phase0.timeseries.E').ravel()
h = prob.get_val('traj.phase0.timeseries.h').ravel()
gg = prob.get_val('traj.phase0.timeseries.gg').ravel()

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
plt.show()

a=1