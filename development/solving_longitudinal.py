import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
import longitudinal_kinematics as kinematics
import time


cruise_altitude = 17000.0  # m
total_duration = 24 * 3600  # s - single combined duration bound for all 3 stages

# Rough estimates for the initial guess only (not bounds) - adjust as needed
climb_duration_guess = 16500.0
cruise_duration_guess = 60000.0
descent_duration_guess = total_duration - climb_duration_guess - cruise_duration_guess


# Initialize the Problem and the optimization driver
prob = om.Problem(model=om.Group())

prob.driver = om.ScipyOptimizeDriver() # Or pyOptSparseDriver(optimizer='IPOPT')
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['maxiter'] = 1500
prob.driver.options['tol'] = 1e-5
prob.driver.opt_settings['maxiter'] = 1500
prob.driver.opt_settings['ftol'] = 1e-5

prob.driver.declare_coloring()

# Create a trajectory with a climb phase followed by a cruise phase
traj = prob.model.add_subsystem('traj', dm.Trajectory())

climb = traj.add_phase('climb', dm.Phase(ode_class=kinematics.Kinematics, transcription=dm.Radau(num_segments=10, order=3)))
cruise = traj.add_phase('cruise', dm.Phase(ode_class=kinematics.Kinematics, transcription=dm.Radau(num_segments=10, order=3)))
descent = traj.add_phase('descent', dm.Phase(ode_class=kinematics.Kinematics, transcription=dm.Radau(num_segments=10, order=3)))

# Phase1 : Climb
# duration_bounds is deliberately wide/permissive here - the actual limit on the combined
# mission comes from the single boundary constraint on descent's final time, below.
climb.set_time_options(fix_initial=True, duration_bounds=(1, total_duration))
climb.add_state('h', rate_source='h_dot', fix_initial=True, fix_final=True, units='m')
climb.add_control('V', lower=8, upper=12, units='m/s')
climb.add_control('gg', lower=np.radians(-5), upper=np.radians(5), units='rad')

# Phase2 : Cruise
cruise.set_time_options(fix_initial=False, duration_bounds=(1, total_duration))
cruise.add_state('h', rate_source='h_dot', fix_initial=False, fix_final=False, units='m')
cruise.add_control('V', lower=8, upper=12, units='m/s')
cruise.add_parameter('gg', val=0.0, opt=False, units='rad')  # fixed: level flight
cruise.add_objective('time_phase', loc='final', scaler=-1)  # maximize the cruise phase's own duration

# Phase3 : Descent
descent.set_time_options(fix_initial=False, duration_bounds=(1, total_duration))
descent.add_state('h', rate_source='h_dot', fix_initial=False, fix_final=True, units='m')
descent.add_control('V', lower=8, upper=12, units='m/s')
descent.add_control('gg', lower=np.radians(-5), upper=np.radians(5), units='rad')

# Single combined duration bound for all 3 stages
descent.add_boundary_constraint('time', loc='final', equals=total_duration, units='s')

traj.link_phases(phases=['climb', 'cruise', 'descent'], vars=['h', 'time'])

prob.model.linear_solver = om.DirectSolver()

# Setup the Problem
prob.setup()

# Set the initial values (guesses only - the estimates above, not hard bounds)
climb.set_time_val(initial=0.0, duration=climb_duration_guess)
climb.set_state_val('h', [0, cruise_altitude])
climb.set_control_val('V', [10, 12])
climb.set_control_val('gg', [np.radians(5), np.radians(5)])

cruise.set_time_val(initial=climb_duration_guess, duration=cruise_duration_guess)
cruise.set_state_val('h', [cruise_altitude, cruise_altitude])
cruise.set_control_val('V', [10, 10])

descent.set_time_val(initial=climb_duration_guess + cruise_duration_guess, duration=descent_duration_guess)
descent.set_state_val('h', [cruise_altitude, 0])
descent.set_control_val('V', [10, 8])
descent.set_control_val('gg', [np.radians(-5), np.radians(-5)])

start_time = time.perf_counter() # High-resolution start

# Solve for the optimal trajectory
dm.run_problem(prob, refine_iteration_limit=2, refine_method='hp')

elapsed = time.perf_counter() - start_time
print(f"Elapsed time: {elapsed:.2f} seconds")

# Check the results
print('Climb duration (s):', prob.get_val('traj.climb.t_duration')[0])
print('Cruise duration (s):', prob.get_val('traj.cruise.t_duration')[0])
print('Descent duration (s):', prob.get_val('traj.descent.t_duration')[0])
print('Total mission time (s):', prob.get_val('traj.descent.timeseries.time')[-1, 0])

# Generate the explicitly simulated trajectory
exp_out = traj.simulate()


def stitch(varname, source):
    """Concatenate a timeseries variable across the climb, cruise and descent phases."""
    return np.concatenate([
        source.get_val(f'traj.{phase}.timeseries.{varname}')[:, 0]
        for phase in ('climb', 'cruise', 'descent')
    ])


t_sol, h_sol, V_sol = stitch('time', prob), stitch('h', prob), stitch('V', prob)
t_sim, h_sim, V_sim = stitch('time', exp_out), stitch('h', exp_out), stitch('V', exp_out)
t_split1 = prob.get_val('traj.climb.timeseries.time')[-1, 0]
t_split2 = prob.get_val('traj.cruise.timeseries.time')[-1, 0]

fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
fig.suptitle('Climb + Cruise + Descent (maximize cruise duration)')

axs[0].plot(t_sol, h_sol, 'o', ms=4, label='solution')
axs[0].plot(t_sim, h_sim, '-', label='simulation')
axs[0].axvline(t_split1, color='gray', ls='--', lw=1, label='phase boundary')
axs[0].axvline(t_split2, color='gray', ls='--', lw=1)
axs[0].set_ylabel('h (m)')
axs[0].legend()

axs[1].plot(t_sol, V_sol, 'o', ms=4, label='solution')
axs[1].plot(t_sim, V_sim, '-', label='simulation')
axs[1].axvline(t_split1, color='gray', ls='--', lw=1)
axs[1].axvline(t_split2, color='gray', ls='--', lw=1)
axs[1].set_xlabel('time (s)')
axs[1].set_ylabel('V (m/s)')

plt.show()
