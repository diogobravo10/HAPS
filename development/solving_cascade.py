import json
import numpy as np
import openmdao.api as om
import dymos as dm
import longitudinal_kinematics
import lateral_kinematics
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
        self.add_subsystem('kinematics', lateral_kinematics.Kinematics(num_nodes=nn), promotes=['*'])

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


lateral_seg_duration = 3600.0  # s, exactly 1 hour per segment
lateral_dx_per_seg = 1000.0    # m, required x progress per segment
lateral_dy_per_seg = 100.0     # m, required y progress per segment
lateral_n_segments = int(total_duration / lateral_seg_duration)  # 24, tiling the whole mission
lateral_x0 = 0.0
lateral_y0 = -1000.0


def solve_lateral(t_data, V_data, n_segments=lateral_n_segments):
    # Instead of one long, weakly-constrained phase (which let the optimizer pick
    # a pathological 24h near-360deg slow loop under a flat objective), the lateral
    # maneuver is broken into n_segments 1-hour phases, linked in sequence. Each
    # phase's duration is fixed at exactly 1 hour, and its x/y endpoints are pinned
    # to an explicit arithmetic progression (+1000 m in x, +100 m in y per hour),
    # so every phase is forced to make bounded, sensible progress rather than being
    # free to wander for the whole mission.

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

    seg_names = [f'seg{i}' for i in range(n_segments)]
    phases = []
    for i, name in enumerate(seg_names):
        phase = traj.add_phase(name,
                               dm.Phase(ode_class=LateralWithVProfile,
                                        ode_init_kwargs={'t_data': t_data, 'V_data': V_data},
                                        transcription=dm.Radau(num_segments=5, order=3)))
        phases.append(phase)

        phase.set_time_options(fix_initial=(i == 0), fix_duration=True, duration_val=lateral_seg_duration)

        x_i0 = lateral_x0 + i * lateral_dx_per_seg
        x_i1 = lateral_x0 + (i + 1) * lateral_dx_per_seg
        y_i0 = lateral_y0 + i * lateral_dy_per_seg
        y_i1 = lateral_y0 + (i + 1) * lateral_dy_per_seg

        phase.add_state('x', rate_source='x_dot', fix_initial=True, fix_final=True, units='m')
        phase.add_state('y', rate_source='y_dot', fix_initial=True, fix_final=True, units='m')
        phase.add_state('psi', rate_source='psi_dot', fix_initial=(i == 0), fix_final=False,
                        lower=np.radians(-179), upper=np.radians(179), units='rad')

        # V is no longer a control - it's supplied by LateralWithVProfile as a function
        # of time, taken from the solved longitudinal trajectory's V(t) history.
        phase.add_control('phi', lower=np.radians(-60), upper=np.radians(60), units='rad') # Singularity at phi = 90

    # Continuity of heading and the timeline across consecutive 1-hour segments
    # (x and y are already pinned to matching numeric values at each boundary, so
    # they don't need linking - only psi and time are actually free at the joints)
    for i in range(n_segments - 1):
        traj.link_phases(phases=[seg_names[i], seg_names[i + 1]], vars=['psi', 'time'])

    # Feasibility only: with V(t) prescribed and each segment's duration/progress
    # already pinned explicitly, there's nothing meaningful left to optimize. This
    # is a genuine constant, never a function of any design variable, so the driver
    # purely satisfies constraints (bounds + defects + boundary conditions).
    prob.model.add_subsystem('dummy_obj', om.ExecComp('obj = 1.0'))
    prob.model.add_objective('dummy_obj.obj')

    prob.model.linear_solver = om.DirectSolver()

    # Setup the Problem
    prob.setup()

    # Set the initial values
    for i, phase in enumerate(phases):
        t_i0 = i * lateral_seg_duration
        x_i0 = lateral_x0 + i * lateral_dx_per_seg
        x_i1 = lateral_x0 + (i + 1) * lateral_dx_per_seg
        y_i0 = lateral_y0 + i * lateral_dy_per_seg
        y_i1 = lateral_y0 + (i + 1) * lateral_dy_per_seg

        phase.set_time_val(initial=t_i0, duration=lateral_seg_duration)
        phase.set_state_val('x', [x_i0, x_i1])
        phase.set_state_val('y', [y_i0, y_i1])
        phase.set_state_val('psi', [0.0, 0.0])
        phase.set_control_val('phi', [0.0, 0.0])

    start_time = time.perf_counter() # High-resolution start

    # Solve for the optimal trajectory
    solution_record_file = 'solution.db'
    dm.run_problem(prob, refine_iteration_limit=1, refine_method='hp',
                   solution_record_file=solution_record_file)

    elapsed = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed:.2f} seconds")

    # Check the results
    print('Final time:', prob.get_val(f'traj.{seg_names[-1]}.timeseries.time')[-1])

    # Generate the explicitly simulated trajectory
    simulation_record_file = 'simulation.db'
    exp_out = traj.simulate(record_file=simulation_record_file)

    # Record where this run's databases actually landed so plot_cascade.py can find them.
    # exp_out.get_outputs_dir() is used (rather than assuming 'traj_simulation_0_out')
    # since grid refinement runs its own internal simulate() calls for error estimation,
    # which can consume index 0 before this explicit call gets a later index.
    solution_path = str(prob.get_outputs_dir() / solution_record_file)
    simulation_path = str(exp_out.get_outputs_dir() / simulation_record_file)
    with open(lateral_paths_file, 'w') as f:
        json.dump({'solution': solution_path, 'simulation': simulation_path, 'seg_names': seg_names}, f, indent=2)

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
