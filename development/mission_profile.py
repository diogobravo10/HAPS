import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
from solarpy import irradiance_on_plane, daylight_hours
from ambiance import Atmosphere
from datetime import datetime, timedelta


class SolarAircraftODE(om.ExplicitComponent):
    def initialize(self):
        # Dymos requires num_nodes to vectorize calculations across the trajectory grid
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs: States & Controls
        self.add_input('gg', val=np.ones(nn), units='rad', desc='Angle of Climb (control)')

        # Outputs: Derivatives for Dymos
        self.add_output('dhdt', val=np.zeros(nn), units='m/s', desc='Rate of change of altitude')
        self.add_output('E_dot', val=np.zeros(nn), units='W', desc='Net power accumulation')

        # Setup automatic derivatives via Complex Step or Finite Difference
        self.declare_partials('*', '*', method='fd')


    def compute(self, inputs, outputs):

        gg = inputs['gg']

        # Aircraft Constants - link with solar irradiance after debugging and validating
        m, g, rho, S = 1.2, 9.81, 1.29, 0.1566
        K, CD0 = 0.1, 0.011
        eta_sol, eta_prop, P_sd = 0.9, 0.7, 380.0
        a, e = np.radians(0), np.radians(45)  # Sun elevation and azimuth angles
        V = 15.0  # m/s (constant airspeed for this example)

        # Kinematic Dynamics
        dhdt = V * np.sin(gg)
        outputs['dhdt'] = dhdt

        # Solar Power Input
        P_in = eta_sol * P_sd * S

        # Aerodynamic Drag & Required Thrust Power
        CL = m*g / (0.5 * rho * S * V**2)
        CD = CD0 + K * CL**2
        D = 0.5 * rho * S * V**2 * CD

        # Net Power available for battery storage
        outputs['E_dot'] = P_in - D*V - m*g*dhdt
       


# 1. Create Problem Instance
prob = om.Problem()

# 2. Add Trajectory & Phase
traj = dm.Trajectory()
phase = dm.Phase(
    ode_class=SolarAircraftODE, 
    transcription=dm.Radau(num_segments=20, order=3)
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
phase.set_time_options(fix_initial=True, duration_bounds=(50, 3600), units='s')


# 4. Configure States
phase.add_state('h', rate_source='dhdt', fix_initial=True, units='m')
phase.add_state('E', rate_source='E_dot', fix_initial=True, units='J')  # Net energy integral - auxilary state variable

phase.add_control('gg', lower=np.radians(0), upper=np.radians(5), units='rad')

phase.add_boundary_constraint('h', loc='final', equals=13000.0,  units='m')
# phase.add_boundary_constraint('h', loc='initial', equals=12000.0, units='m')

# 6. Set Objective: Maximize final accumulated net energy E_total [Eq. (16)]
phase.add_objective('E', loc='final', scaler=-1e0)

# 7. Setup and Run
prob.setup()

# Set Initial Guesses
prob.set_val('traj.phase0.t_duration', 3600.0)
prob.set_val('traj.phase0.states:h', phase.interp('h', [12000, 13000]))
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