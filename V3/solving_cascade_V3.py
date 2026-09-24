import json
import numpy as np
import openmdao.api as om
import dymos as dm
import module_lateral_kinematics_cylinder
import solving_longitudinal
import plot_cascade
import time
from velocity_profile import VelocityProfileComp, AltitudeProfileComp


class LateralWithVHProfile(om.Group):
    """Lateral kinematics driven by precomputed V(t) and h(t) profiles from a solved
    longitudinal trajectory, instead of a free V control - same principle as
    solving_cascade_V2's LateralWithVProfile, extended to also carry h(t) through so
    altitude is available directly as an output of this phase (V2 only exposed x/y/tt
    here, leaving plot_cascade.py to reconstruct h by interpolating against the
    longitudinal solution separately)."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('t_data', desc='1D array of time samples for the V(t)/h(t) profiles')
        self.options.declare('V_data', desc='1D array of V samples at t_data')
        self.options.declare('h_data', desc='1D array of h samples at t_data')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('vprofile',
                           VelocityProfileComp(num_nodes=nn,
                                                t_data=self.options['t_data'],
                                                V_data=self.options['V_data']),
                           promotes=['*'])
        self.add_subsystem('hprofile',
                           AltitudeProfileComp(num_nodes=nn,
                                                t_data=self.options['t_data'],
                                                h_data=self.options['h_data']),
                           promotes=['*'])
        self.add_subsystem('kinematics', module_lateral_kinematics_cylinder.Kinematics(num_nodes=nn), promotes=['*'])


lateral_paths_file = 'solving_lateral_paths.json'

lateral_R = 50000.0  # m, fixed loiter radius (parameter, not optimized) - adjust as needed
lateral_num_segments = 500  # Radau segments spanning the whole mission duration


def solve_lateral(t_data, V_data, h_data, R=lateral_R, num_segments=lateral_num_segments):
    # Same single constant-radius loiter phase structure as solving_cascade_V2's
    # solve_lateral - tt just integrates tt_dot = V/R from t=0 to total_duration, with
    # V(t) (and now h(t) alongside it) prescribed rather than free/derived.

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

    # Reuse solving_longitudinal's own total_duration rather than a second hardcoded
    # copy of it, so the lateral phase's duration can't silently drift out of sync with
    # the mission duration the V(t)/h(t) profiles were actually solved over.
    total_duration = solving_longitudinal.total_duration

    phase = traj.add_phase('phase0',
                           dm.Phase(ode_class=LateralWithVHProfile,
                                    ode_init_kwargs={'t_data': t_data, 'V_data': V_data, 'h_data': h_data},
                                    transcription=dm.Radau(num_segments=num_segments, order=3)))

    phase.set_time_options(fix_initial=True, fix_duration=True)

    phase.add_state('tt', rate_source='tt_dot', fix_initial=True, fix_final=False, units='rad')
    # R must be free (opt=True) for maximizing turns to do anything: with V(t) prescribed,
    # tt_final = (total path length) / R, so tighter turns always increase the turn count
    # and the optimizer will drive R straight to this lower bound. Set it to the tightest
    # radius your aircraft can actually sustain - this ODE has no bank-angle limit of its own.
    phase.add_parameter('R', val=R, units='m', opt=True, lower=20000.0)
    phase.add_timeseries_output(['x', 'y', 'h'])

    # Maximize the number of turns around the cylinder (tt_final / 2*pi) by letting
    # the optimizer tighten R - see the R parameter above for the bound that drives this.
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

    # Record where this run's database actually landed so plot_cascade.py can find it.
    # seg_names is kept as a one-element list so plot_cascade.py's phase-stitching
    # helper (built for the old multi-segment layout) still works unchanged. No
    # traj.simulate() call here - plot_cascade.py plots solution points only.
    solution_path = str(prob.get_outputs_dir() / solution_record_file)
    with open(lateral_paths_file, 'w') as f:
        json.dump({'solution': solution_path, 'seg_names': ['phase0']}, f, indent=2)

    return prob


def stitch(varname, source):
    """Concatenate a timeseries variable across the climb, cruise and descent phases."""
    return np.concatenate([
        source.get_val(f'traj.{phase}.timeseries.{varname}')[:, 0]
        for phase in ('climb', 'cruise', 'descent')
    ])


if __name__ == '__main__':
    # Solve the full longitudinal mission (aero/solar/potential/battery model) via the
    # up-to-date solving_longitudinal.py itself, rather than re-implementing a
    # simplified copy of it in this file the way solving_cascade_V2 did. This also
    # writes solving_longitudinal_paths.json, which plot_cascade.py reads directly.
    prob_long, _, _ = solving_longitudinal.main()

    # Whole-mission V(t) and h(t) history (climb + cruise + descent, starting at t=0),
    # taken from the solved (collocated) longitudinal trajectory - no traj.simulate()
    # call, so these are the solution's own node points, stored together on the same
    # time grid and fed into the lateral phase below via LateralWithVHProfile.
    t_data = stitch('time', prob_long)
    V_data = stitch('V', prob_long)
    h_data = stitch('h', prob_long)

    solve_lateral(t_data, V_data, h_data)

    plot_cascade.main()
