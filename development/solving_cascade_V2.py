import json
import numpy as np
import openmdao.api as om
import dymos as dm
import longitudinal_kinematics
import lateral_kinematics_cylinder
import time
import plot_cascade
from velocity_profile import VelocityProfileComp

kinematics = longitudinal_kinematics


class LateralWithVProfile(om.Group):
    """Lateral kinematics driven by a precomputed V(t) profile (e.g. from a
    solved longitudinal trajectory) instead of a free V control."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('t_data', desc='1D array of time samples for the V(t) profile')
        self.options.declare('V_data', desc='1D array of V samples at t_data')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('vprofile',
                           VelocityProfileComp(num_nodes=nn,
                                                t_data=self.options['t_data'],
                                                V_data=self.options['V_data']),
                           promotes=['*'])
        self.add_subsystem('kinematics', lateral_kinematics_cylinder.Kinematics(num_nodes=nn), promotes=['*'])

cruise_altitude = 20000.0  # m
total_duration = 24 * 3600  # s - single combined duration bound for all 3 stages

# Rough estimates for the initial guess only (not bounds) - adjust as needed
climb_duration_guess = 16500.0
cruise_duration_guess = 60000.0
descent_duration_guess = total_duration - climb_duration_guess - cruise_duration_guess

paths_file = 'solving_longitudinal_paths.json'


def solve_longitudinal():
    # Initialize the Problem and the optimization driver
    prob = om.Problem(model=om.Group(), name='solving_longitudinal')

    prob.driver = om.ScipyOptimizeDriver() # Or pyOptSparseDriver(optimizer='IPOPT')
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['maxiter'] = 3500
    prob.driver.options['tol'] = 1e-5
    prob.driver.opt_settings['maxiter'] = 3500
    prob.driver.opt_settings['ftol'] = 1e-5

    prob.driver.declare_coloring()

    # Create a trajectory
    traj = prob.model.add_subsystem('traj', dm.Trajectory())

    climb = traj.add_phase('climb', dm.Phase(ode_class=kinematics.Kinematics, transcription=dm.Radau(num_segments=10, order=3)))
    cruise = traj.add_phase('cruise', dm.Phase(ode_class=kinematics.Kinematics, transcription=dm.Radau(num_segments=10, order=3)))
    descent = traj.add_phase('descent', dm.Phase(ode_class=kinematics.Kinematics, transcription=dm.Radau(num_segments=10, order=3)))

    # Phase1 : Climb
    climb.set_time_options(fix_initial=True, duration_bounds=(1, total_duration))
    climb.add_state('h', rate_source='h_dot', fix_initial=True, fix_final=True, units='m')
    climb.add_control('V', lower=8, upper=12, units='m/s')
    climb.add_control('gg', lower=np.radians(-5), upper=np.radians(5), units='rad')
    climb.add_boundary_constraint('gg', loc='final', equals=0.0, units='rad')  # level off before cruise

    # Phase2 : Cruise
    cruise.set_time_options(fix_initial=False, duration_bounds=(1, total_duration))
    cruise.add_state('h', rate_source='h_dot', fix_initial=False, fix_final=False, units='m')
    cruise.add_control('V', lower=8, upper=12, units='m/s')
    cruise.add_parameter('gg', val=0.0, opt=False, units='rad')
    cruise.add_timeseries_output('gg')
    cruise.add_objective('time_phase', loc='final', scaler=-1)  # maximize the cruise phase's own duration

    # Phase3 : Descent
    descent.set_time_options(fix_initial=False, duration_bounds=(1, total_duration))
    descent.add_state('h', rate_source='h_dot', fix_initial=False, fix_final=True, units='m')
    descent.add_control('V', lower=8, upper=12, units='m/s')
    descent.add_control('gg', lower=np.radians(-5), upper=np.radians(5), units='rad')
    descent.add_boundary_constraint('time', loc='final', equals=total_duration, units='s')

    traj.link_phases(phases=['climb', 'cruise', 'descent'], vars=['h', 'time'])

    prob.model.linear_solver = om.DirectSolver()

    # Setup the Problem
    prob.setup()

    # Set the initial values -> Provide a better initial guess
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

    start_time = time.perf_counter()

    # Solve for the optimal trajectory
    solution_record_file = 'solution.db'
    dm.run_problem(prob, solution_record_file=solution_record_file)

    elapsed = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed:.2f} seconds")

    # Generate the explicitly simulated trajectory
    simulation_record_file = 'simulation.db'
    exp_out = traj.simulate(record_file=simulation_record_file)

    # Check the results
    print('Climb duration (s):', prob.get_val('traj.climb.t_duration')[0])
    print('Cruise duration (s):', prob.get_val('traj.cruise.t_duration')[0])
    print('Descent duration (s):', prob.get_val('traj.descent.t_duration')[0])
    print('Total mission time (s):', prob.get_val('traj.descent.timeseries.time')[-1, 0])

    # Records
    solution_path = str(prob.get_outputs_dir() / solution_record_file)
    simulation_path = str(exp_out.get_outputs_dir() / simulation_record_file)
    with open(paths_file, 'w') as f:
        json.dump({'solution': solution_path, 'simulation': simulation_path}, f, indent=2)

    return prob, exp_out


lateral_paths_file = 'solving_lateral_paths.json'

lateral_R = 1000.0  # m, fixed loiter radius (parameter, not optimized) - adjust as needed
lateral_num_segments = 500  # Radau segments spanning the whole mission duration


def solve_lateral(t_data, V_data, R=lateral_R, num_segments=lateral_num_segments):
    # With lateral_kinematics_cylinder, x/y are no longer states - they're algebraic
    # outputs of the state 'tt' (angular position) at a fixed radius R. There's no
    # control left either (V is prescribed via LateralWithVProfile, R is a fixed
    # parameter), so this is a single constant-radius loiter phase spanning the whole
    # mission: tt just integrates tt_dot = V/R from t=0 to total_duration.

    # Initialize the Problem and the optimization driver
    prob = om.Problem(model=om.Group(), name='solving_lateral')

    prob.driver = om.ScipyOptimizeDriver() # Or pyOptSparseDriver(optimizer='IPOPT')
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['maxiter'] = 1500
    prob.driver.options['tol'] = 1e-5
    prob.driver.opt_settings['maxiter'] = 1500
    prob.driver.opt_settings['ftol'] = 1e-5

    prob.driver.declare_coloring()

    traj = prob.model.add_subsystem('traj', dm.Trajectory())

    phase = traj.add_phase('phase0',
                           dm.Phase(ode_class=LateralWithVProfile,
                                    ode_init_kwargs={'t_data': t_data, 'V_data': V_data},
                                    transcription=dm.Radau(num_segments=num_segments, order=3)))

    phase.set_time_options(fix_initial=True, fix_duration=True)

    phase.add_state('tt', rate_source='tt_dot', fix_initial=True, fix_final=False, units='rad')
    # R must be free (opt=True) for maximizing turns to do anything: with V(t) prescribed,
    # tt_final = (total path length) / R, so tighter turns always increase the turn count
    # and the optimizer will drive R straight to this lower bound. Set it to the tightest
    # radius your aircraft can actually sustain - this ODE has no bank-angle limit of its own.
    phase.add_parameter('R', val=R, units='m', opt=True, lower=500.0)
    phase.add_timeseries_output(['x', 'y'])

    # Maximize the number of turns around the cylinder (tt_final / 2*pi) by letting
    # the optimizer tighten R - see the R parameter below for the bound that drives this.
    phase.add_objective('tt', loc='final', scaler=-1)

    prob.model.linear_solver = om.DirectSolver()

    # Setup the Problem
    prob.setup()

    # Set the initial values
    phase.set_time_val(initial=0.0, duration=total_duration)
    phase.set_state_val('tt', [0.0, total_duration * np.mean(V_data) / R])

    start_time = time.perf_counter() # High-resolution start

    # Solve for the optimal trajectory
    solution_record_file = 'solution.db'
    dm.run_problem(prob, refine_iteration_limit=1, refine_method='hp',
                   solution_record_file=solution_record_file)

    elapsed = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed:.2f} seconds")

    # Check the results
    print('Final time:', prob.get_val('traj.phase0.timeseries.time')[-1])

    # Generate the explicitly simulated trajectory
    simulation_record_file = 'simulation.db'
    exp_out = traj.simulate(record_file=simulation_record_file)

    # Record where this run's databases actually landed so plot_cascade.py can find them.
    # exp_out.get_outputs_dir() is used (rather than assuming 'traj_simulation_0_out')
    # since grid refinement runs its own internal simulate() calls for error estimation,
    # which can consume index 0 before this explicit call gets a later index.
    # seg_names is kept as a one-element list so plot_cascade.py's phase-stitching
    # helper (built for the old multi-segment layout) still works unchanged.
    solution_path = str(prob.get_outputs_dir() / solution_record_file)
    simulation_path = str(exp_out.get_outputs_dir() / simulation_record_file)
    with open(lateral_paths_file, 'w') as f:
        json.dump({'solution': solution_path, 'simulation': simulation_path,
                   'seg_names': ['phase0']}, f, indent=2)

    return prob, exp_out


def stitch(varname, source):
    """Concatenate a timeseries variable across the climb, cruise and descent phases."""
    return np.concatenate([
        source.get_val(f'traj.{phase}.timeseries.{varname}')[:, 0]
        for phase in ('climb', 'cruise', 'descent')
    ])


if __name__ == '__main__':
    prob_long, exp_out_long = solve_longitudinal()

    # Whole-mission V(t) history (climb + cruise + descent, starting at t=0),
    # taken from the simulated (not collocated) longitudinal trajectory since it's
    # the physically continuous one - this feeds LateralWithVProfile as a spline.
    t_data = stitch('time', exp_out_long)
    V_data = stitch('V', exp_out_long)

    solve_lateral(t_data, V_data)

    plot_cascade.main()
