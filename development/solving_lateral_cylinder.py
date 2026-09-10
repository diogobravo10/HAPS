import numpy as np
import openmdao.api as om
import dymos as dm
from dymos.examples.plotting import plot_results
import matplotlib.pyplot as plt
import lateral_kinematics_cylinder as kinematics
import time


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
                       dm.Phase(ode_class= kinematics.Kinematics,
                                transcription=dm.Radau(num_segments=30, order=3)))



# Configure Time Variable
phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 40000))

# Configure States
phase.add_state('tt', rate_source='tt_dot', fix_initial=True, fix_final=True,units='rad')

phase.add_control('V', lower=8, upper=12, units='m/s')

phase.add_parameter('R', val=1000, units='m', opt=False) 


phase.add_timeseries_output(['x', 'y'])

# Minimize time at the end of the phase
phase.add_objective('time', loc='final', scaler=1)

prob.model.linear_solver = om.DirectSolver()

# Setup the Problem
prob.setup()

# Set the initial values
phase.set_time_val(initial=0.0, duration=100.0)
phase.set_state_val('tt', [0, 10*np.pi])

phase.set_control_val('V', [3, 10])


start_time = time.perf_counter() # High-resolution start

# Solve for the optimal trajectory
dm.run_problem(prob)
# dm.run_problem(prob, refine_iteration_limit=2, refine_method='hp')

elapsed = time.perf_counter() - start_time
print(f"Elapsed time: {elapsed:.2f} seconds")


# Check the results
print(prob.get_val('traj.phase0.timeseries.time')[-1])

# Generate the explicitly simulated trajectory
exp_out = traj.simulate()

plot_results([('traj.phase0.timeseries.x', 'traj.phase0.timeseries.y',
               'x (m)', 'y (m)')],
             title='Brachistochrone Solution\nHigh-Order Gauss-Lobatto Method',
             p_sol=prob,
            p_sim=exp_out
             )


fig = plt.gcf()
axs = fig.axes

# x-y trajectory
# axs[0].set_xlim(-1000, 20000)
# axs[0].set_ylim(-1500, 1350)
axs[0].ticklabel_format(axis='x', style='plain', useOffset=False)




plt.show()