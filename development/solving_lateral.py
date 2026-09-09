import numpy as np
import openmdao.api as om
import dymos as dm
from dymos.examples.plotting import plot_results
import matplotlib.pyplot as plt
import lateral_kinematics as lateralkinematics


# Initialize the Problem and the optimization driver
prob = om.Problem(model=om.Group())

prob.driver = om.ScipyOptimizeDriver() # Or pyOptSparseDriver(optimizer='IPOPT')
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['maxiter'] = 1500 
prob.driver.options['tol'] = 1e-5
prob.driver.opt_settings['maxiter'] = 1500
prob.driver.opt_settings['ftol'] = 1e-5 

prob.driver.declare_coloring()

# Create a trajectory and add a phase to it
traj = prob.model.add_subsystem('traj', dm.Trajectory())
phase = traj.add_phase('phase0',
                       dm.Phase(ode_class= lateralkinematics.Kinematics,
                                transcription=dm.Radau(num_segments=10)))



# Configure Time Variable
phase.set_time_options(fix_initial=True, duration_bounds=(.5, 1100))

# Configure States
phase.add_state('x', rate_source='x_dot', fix_initial=True, fix_final=True,units='m')
phase.add_state('y', rate_source='y_dot', fix_initial=True, fix_final=True, units='m')
phase.add_state('psi', rate_source='psi_dot', fix_initial=False, fix_final=False,  lower=np.radians(-180), upper=np.radians(180), units='rad')
phase.add_state('h', rate_source='h_dot', fix_initial=True, fix_final=True, units='m')

phase.add_control('V', lower=2.5, upper=12, units='m/s')
phase.add_control('phi', lower=np.radians(-60), upper=np.radians(60), units='rad') # Singularity at phi = 90
phase.add_control('gg', lower=np.radians(-5), upper=np.radians(5), units='rad') # Glide angle control

# Minimize time at the end of the phase
phase.add_objective('time', loc='final', scaler=1)

prob.model.linear_solver = om.DirectSolver()

# Setup the Problem
prob.setup()

# Set the initial values
phase.set_time_val(initial=0.0, duration=1000.0)
phase.set_state_val('x', [0, 1000])
phase.set_state_val('y', [0, 0])
phase.set_state_val('psi', [np.radians(0), np.radians(0)])
phase.set_state_val('h', [1000, 0])

phase.set_control_val('V', [3, 10])
phase.set_control_val('phi', [np.radians(0), np.radians(0)])
phase.set_control_val('gg', [np.radians(-1), np.radians(-1)])

# Solve for the optimal trajectory
dm.run_problem(prob)
# dm.run_problem(prob, refine_iteration_limit=10, refine_method='hp')

# Check the results
print(prob.get_val('traj.phase0.timeseries.time')[-1])

# Generate the explicitly simulated trajectory
exp_out = traj.simulate()

plot_results([('traj.phase0.timeseries.x', 'traj.phase0.timeseries.y',
               'x (m)', 'y (m)'),
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.psi',
               'time (s)', 'psi (rad)')],
             title='Brachistochrone Solution\nHigh-Order Gauss-Lobatto Method',
             p_sol=prob, 
            p_sim=exp_out
             )


fig = plt.gcf()
axs = fig.axes

# x-y trajectory
axs[0].set_xlim(-1000, 1000)
axs[0].set_ylim(0, 1350)
axs[0].ticklabel_format(axis='x', style='plain', useOffset=False)

# Heading angle
axs[1].set_ylim(0, np.pi)
axs[1].set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])


# 3D trajectory (x, y, h) - plot_results only handles 2D (x, y) pairs, so this is built directly
fig3d = plt.figure(figsize=(8, 6))
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.plot(prob.get_val('traj.phase0.timeseries.x')[:, 0],
         prob.get_val('traj.phase0.timeseries.y')[:, 0],
         prob.get_val('traj.phase0.timeseries.h')[:, 0],
         marker='o', ms=4, linestyle='None', label='solution')
ax3d.plot(exp_out.get_val('traj.phase0.timeseries.x')[:, 0],
         exp_out.get_val('traj.phase0.timeseries.y')[:, 0],
         exp_out.get_val('traj.phase0.timeseries.h')[:, 0],
         linestyle='-', label='simulation')
ax3d.set_xlabel('x (m)')
ax3d.set_ylabel('y (m)')
ax3d.set_zlabel('h (m)')
ax3d.set_title('3D Glide Trajectory')
ax3d.legend()

plt.show()