import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt



class SolarAircraftODE(om.ExplicitComponent):
    def initialize(self):
        # Dymos requires num_nodes to vectorize calculations across the trajectory grid
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs: States & Controls
        self.add_input('psi', val=np.zeros(nn), units='rad', desc='Heading angle')
        self.add_input('phi', val=np.zeros(nn), units='rad', desc='Bank angle (control)')
        self.add_input('V', val=np.ones(nn)*15.0, units='m/s', desc='Airspeed (control)')

        # Outputs: Derivatives for Dymos
        self.add_output('x_dot', val=np.zeros(nn), units='m/s')
        self.add_output('y_dot', val=np.zeros(nn), units='m/s')
        self.add_output('psi_dot', val=np.zeros(nn), units='rad/s')
        self.add_output('E_dot', val=np.zeros(nn), units='W', desc='Net power accumulation')

        # Setup automatic derivatives via Complex Step or Finite Difference
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        psi = inputs['psi']
        phi = inputs['phi']
        V = inputs['V']

        # Aircraft Constants - link with solar irradiance after debugging and validating
        m, g, rho, S = 1.2, 9.81, 1.29, 0.1566
        K, CD0 = 0.1, 0.011
        eta_sol, eta_prop, P_sd = 0.2, 0.7, 380.0
        e, a = np.radians(0), np.radians(0)  # Sun elevation and azimuth angles

        # Kinematic Dynamics (Eqs. 18-20)
        outputs['x_dot'] = V * np.cos(psi)
        outputs['y_dot'] = V * np.sin(psi)
        outputs['psi_dot'] = (g * np.tan(phi)) / V

        # Power Balance (Objective Integration)
        P_in = eta_sol * P_sd * S * (np.cos(phi)*np.sin(e) - np.cos(e)*np.sin(a - psi)*np.sin(phi))
        P_req = 0.5 * rho * S * V**3 * (CD0 + (4*K*(m*g)**2/np.cos(phi)**2)/(rho**2 * S**2 * V**4))
        P_out = P_req / eta_prop

        outputs['E_dot'] = P_in - P_out




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

# 3. SLSQP-specific optimizer options (passes directly to scipy.optimize.minimize)
prob.driver.opt_settings['maxiter'] = 1500
prob.driver.opt_settings['ftol'] = 1e-3  # Function tolerance for convergence


# 3. Configure Time Variable
phase.set_time_options(fix_initial=True, duration_bounds=(50, 400), units='s')

# 4. Configure States
phase.add_state('x', rate_source='x_dot', fix_initial=True, units='m')
phase.add_state('y', rate_source='y_dot', fix_initial=True, units='m')
phase.add_state('psi', rate_source='psi_dot', fix_initial=True, units='rad')
phase.add_state('E', rate_source='E_dot', fix_initial=True, units='J')  # Net energy integral - auxilary state variable

# Boundary constraints on terminal position
phase.add_boundary_constraint('x', loc='final', equals=700.0)
phase.add_boundary_constraint('y', loc='final', equals=1300.0)

# 5. Configure Controls (bank angle phi, speed V)
phase.add_control('phi', lower=np.radians(-45), upper=np.radians(45), units='rad')
phase.add_control('V', lower=15.0, upper=16.0, units='m/s')

# 6. Set Objective: Maximize final accumulated net energy E_total [Eq. (16)]
phase.add_objective('E', loc='final', scaler=-1.0)

# 7. Setup and Run
prob.setup()

# Set Initial Guesses
prob.set_val('traj.phase0.t_duration', 400.0)
prob.set_val('traj.phase0.states:x', phase.interp('x', [0, 700]))
prob.set_val('traj.phase0.states:y', phase.interp('y', [0, 1300]))
prob.set_val('traj.phase0.states:psi', phase.interp('psi', [np.radians(127.0), np.radians(180.0)]))
prob.set_val('traj.phase0.controls:phi', np.radians(0.0))
prob.set_val('traj.phase0.controls:V', 15.0)

dm.run_problem(prob, run_driver=True, make_plots=True)

x_values = prob.get_val('traj.phase0.states:x').ravel()
y_values = prob.get_val('traj.phase0.states:y').ravel()

plt.figure()
plt.plot(x_values, y_values)
plt.xlabel('State x [m]')
plt.ylabel('State y [m]')
plt.title('Flight trajectory')
plt.grid(True)
plt.xlim(-1000.0, 2000.0)
plt.ylim(0.0, 2500.0)

plt.show()


a=1