import json
import numpy as np
import openmdao.api as om
import dymos as dm
import longitudinal_kinematics as kinematics
import aero_module
import time
import plot_longitudinal

cruise_altitude = 17000.0  # m
maximum_altitude = 24000.0 # m
total_duration = 24 * 3600  # s - single combined duration bound for all 3 stages

# Rough estimates for the initial guess only (not bounds) - adjust as needed
climb_duration_guess = 16500.0
cruise_duration_guess = 60000.0
descent_duration_guess = total_duration - climb_duration_guess - cruise_duration_guess

# Rough estimate of DV_sw 
dv_sw_guess_rate = 120

paths_file = 'solving_longitudinal_paths.json'


class LongitudinalODE(om.Group):
    """Flight-path kinematics plus aero_module's drag-power-dissipation subsystem.

    'V' is promoted from both subsystems so they share the same airspeed control;
    'h' (state) and 'aa' (new angle-of-attack control) feed the aero subsystem only.
    Exposes 'h_dot' (for the h state) and 'DV_sw' (for the DV_sw_int integral state).
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('kinematics', kinematics.Kinematics(num_nodes=nn),
                            promotes_inputs=['V', 'gg'], promotes_outputs=['h_dot'])
        self.add_subsystem('aero', aero_module.DragPowerDissipation(num_nodes=nn),
                            promotes_inputs=['V', 'h', 'aa'], promotes_outputs=['DV_sw'])


def main():
    # Initialize the Problem and the optimization driver
    prob = om.Problem(model=om.Group(), name='solving_longitudinal')

    prob.driver = om.ScipyOptimizeDriver() # Or pyOptSparseDriver(optimizer='IPOPT')
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['maxiter'] = 500
    prob.driver.options['tol'] = 1e-5
    prob.driver.opt_settings['maxiter'] = 500
    prob.driver.opt_settings['ftol'] = 1e-5

    prob.driver.declare_coloring()

    # Create a trajectory
    traj = prob.model.add_subsystem('traj', dm.Trajectory())

    climb = traj.add_phase('climb', dm.Phase(ode_class=LongitudinalODE, transcription=dm.Radau(num_segments=10, order=3)))
    cruise = traj.add_phase('cruise', dm.Phase(ode_class=LongitudinalODE, transcription=dm.Radau(num_segments=20, order=3)))
    descent = traj.add_phase('descent', dm.Phase(ode_class=LongitudinalODE, transcription=dm.Radau(num_segments=10, order=3)))

    # Phase1 : Climb
    climb.set_time_options(fix_initial=True, duration_bounds=(3*10*90, total_duration), duration_ref=1e4)
    climb.add_state('h', rate_source='h_dot', fix_initial=True, fix_final=True, units='m',
                     lower=0.0, upper=maximum_altitude, ref=1e4, defect_ref=1e4)
    climb.add_state('DV_sw_int', rate_source='DV_sw', fix_initial=True, fix_final=False, units='J/m**2', ref=1e6, defect_ref=1e6)
    climb.add_control('V', lower=8, upper=12, units='m/s')
    climb.add_control('gg', lower=np.radians(-5), upper=np.radians(5), units='rad')
    climb.add_control('aa', lower=np.radians(-5), upper=np.radians(10), units='rad')
    climb.add_boundary_constraint('gg', loc='final', equals=0.0, units='rad')  # level off before cruise

    # Phase2 : Cruise
    cruise.set_time_options(fix_initial=False, duration_bounds=(3*10*90, total_duration), duration_ref=1e4)
    cruise.add_state('h', rate_source='h_dot', fix_initial=False, fix_final=False, units='m', ref=1e4, defect_ref=1e4)
    cruise.add_state('DV_sw_int', rate_source='DV_sw', fix_initial=False, fix_final=False, units='J/m**2', ref=1e6, defect_ref=1e6)
    cruise.add_control('V', lower=8, upper=12, units='m/s')
    cruise.add_control('aa', lower=np.radians(-5), upper=np.radians(10), units='rad')
    cruise.add_parameter('gg', val=0.0, opt=False, units='rad')
    cruise.add_timeseries_output('gg')

    # Phase3 : Descent
    descent.set_time_options(fix_initial=False, duration_bounds=(3*10*90, total_duration), duration_ref=1e4)
    descent.add_state('h', rate_source='h_dot', fix_initial=False, fix_final=True, units='m',
                       lower=0.0, upper=maximum_altitude, ref=1e4, defect_ref=1e4)
    descent.add_state('DV_sw_int', rate_source='DV_sw', fix_initial=False, fix_final=False, units='J/m**2', ref=1e6, defect_ref=1e6)
    descent.add_control('V', lower=8, upper=12, units='m/s')
    descent.add_control('gg', lower=np.radians(-5), upper=np.radians(5), units='rad')
    descent.add_control('aa', lower=np.radians(-5), upper=np.radians(10), units='rad')
    descent.add_boundary_constraint('time', loc='final', equals=total_duration, units='s', ref=1e4)
    descent.add_objective('DV_sw_int', loc='final', ref=-1e6)

    traj.link_phases(phases=['climb', 'cruise', 'descent'], vars=['h', 'time', 'DV_sw_int', 'V', 'aa'])

    prob.model.linear_solver = om.DirectSolver()

    # Setup the Problem
    prob.setup()

    # Set the initial values -> Provide a better initial guess
    climb.set_time_val(initial=0.0, duration=climb_duration_guess)
    climb.set_state_val('h', [0, cruise_altitude])
    climb.set_state_val('DV_sw_int', [0, climb_duration_guess * dv_sw_guess_rate])
    climb.set_control_val('V', [10, 12])
    climb.set_control_val('gg', [np.radians(5), np.radians(5)])
    climb.set_control_val('aa', [np.radians(2), np.radians(2)])

    dv_sw_int_climb_end = climb_duration_guess * dv_sw_guess_rate
    cruise.set_time_val(initial=climb_duration_guess, duration=cruise_duration_guess)
    cruise.set_state_val('h', [cruise_altitude, cruise_altitude])
    cruise.set_state_val('DV_sw_int', [dv_sw_int_climb_end, dv_sw_int_climb_end + cruise_duration_guess * dv_sw_guess_rate])
    cruise.set_control_val('V', [10, 10])
    cruise.set_control_val('aa', [np.radians(2), np.radians(2)])

    dv_sw_int_cruise_end = dv_sw_int_climb_end + cruise_duration_guess * dv_sw_guess_rate
    descent.set_time_val(initial=climb_duration_guess + cruise_duration_guess, duration=descent_duration_guess)
    descent.set_state_val('h', [cruise_altitude, 0])
    descent.set_state_val('DV_sw_int', [dv_sw_int_cruise_end, dv_sw_int_cruise_end + descent_duration_guess * dv_sw_guess_rate])
    descent.set_control_val('V', [10, 8])
    descent.set_control_val('gg', [np.radians(-5), np.radians(-5)])
    descent.set_control_val('aa', [np.radians(2), np.radians(2)])

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
    print('Integral of DV_sw over mission (J/m^2):', prob.get_val('traj.descent.timeseries.DV_sw_int')[-1, 0])

    # Records
    solution_path = str(prob.get_outputs_dir() / solution_record_file)
    simulation_path = str(prob.get_outputs_dir() / 'traj_simulation_0_out' / simulation_record_file)
    with open(paths_file, 'w') as f:
        json.dump({'solution': solution_path, 'simulation': simulation_path}, f, indent=2)

    return prob, exp_out


if __name__ == '__main__':
    main()
    plot_longitudinal.main()
