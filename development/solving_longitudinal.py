import json
import numpy as np
import openmdao.api as om
import dymos as dm
import longitudinal_kinematics as kinematics
import aero_module
import solar_module
import potential_module
import battery_module
import time
import plot_longitudinal

low_altitude = 10000.0  # m 
initial_altitude = 12000.0  # m 
cruise_altitude = 17000.0  # m
maximum_altitude = 24000.0 # m
total_duration = 24 * 3600  # s - single combined duration bound for all 3 stages

# Rough estimates for the initial guess only (not bounds) - adjust as needed
climb_duration_guess = 16500.0
cruise_duration_guess = 60000.0
descent_duration_guess = total_duration - climb_duration_guess - cruise_duration_guess

# Rough estimate of DV_sw (avg drag-dissipation power density) - low_altitude..cruise_altitude
# is thin air with much less drag than the old sea-level-anchored guess assumed.
dv_sw_guess_rate = 12
# Rough estimate of TV_sw (avg propulsion power density drawn by the propeller) - order of
# magnitude from Tp*Tinst_sw*V/mu_prop at representative Tp~0.3, V~20.
tv_sw_guess_rate = 80
# Rough estimate of Psol_sw (avg solar power density, day+night averaged over the mission)
psol_sw_guess_rate = 70
# Rough estimate of Net_sw = Psol_sw - DV_sw - TV_sw, used only to seed Net_sw_int's guess
net_sw_guess_rate = psol_sw_guess_rate - dv_sw_guess_rate - tv_sw_guess_rate

# Rough estimate of Epot_sw = M_sw*g*h_dot, from the average h_dot implied by the
# altitude/duration guesses above - ~0 in cruise (level flight), and opposite sign in
# climb vs. descent (unlike DV_sw/Psol_sw, which are roughly constant throughout). Climb
# only gains (cruise_altitude - low_altitude), not cruise_altitude from the ground.
climb_altitude_gain = cruise_altitude - initial_altitude
epot_sw_climb_guess_rate = potential_module.M_sw * potential_module.g * climb_altitude_gain / climb_duration_guess
epot_sw_descent_guess_rate = -potential_module.M_sw * potential_module.g * climb_altitude_gain / descent_duration_guess
# Peak magnitude of Epot_sw_int over the mission (used to scale that state) - the state
# itself dips to ~0 at climb start/descent end, but ranges over roughly this much in between.
epot_sw_int_ref = potential_module.M_sw * potential_module.g * climb_altitude_gain

# Battery energy capacity per unit wing area, same formula as battery_module.StateOfCharge
# uses internally - needed here only to seed SOC's initial guess from Net_sw_int's guess.
battery_max_energy = battery_module.mb * 3600 * battery_module.mbat_sw
soc_initial = 0.2  # start the mission fully charged

paths_file = 'solving_longitudinal_paths.json'


class LongitudinalODE(om.Group):
    """Flight-path kinematics plus aero_module's drag-power-dissipation and
    solar_module's solar-power subsystems.

    'V' (state) is promoted from kinematics/aero so they share the same airspeed; 'h'
    (state) is promoted from aero/solar so they share the same altitude; 'gg' (flight-
    path angle control) is promoted from kinematics/aero so they share it too - aero
    needs it for V_dot's gravity component, same units ('rad') on both. 'aa' (angle of
    attack control) and 'Tp' (throttle control) feed aero only. 'time' is Dymos's
    auto-supplied absolute phase time, promoted so solar can use it. Exposes 'h_dot'
    (for the h state), 'V_dot' (for the V state), 'DV_sw' (for the DV_sw_int integral
    state), 'Psol_sw' (for the Psol_sw_int integral state), 'Net_sw' (Psol_sw - DV_sw,
    for the Net_sw_int integral state - net energy collected minus dissipated),
    'Epot_sw' (for the Epot_sw_int integral state - potential energy gained/lost while
    climbing/descending, tracked separately from Net_sw for now), 'SOC_dot' (for the SOC
    state - battery state of charge, driven by the same Net_sw), and 'Vstall'/'Vmargin'
    (diagnostic outputs, not integrated into any state).
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('kinematics', kinematics.Kinematics(num_nodes=nn),
                            promotes_inputs=['V', 'gg'], promotes_outputs=['h_dot'])
        self.add_subsystem('aero', aero_module.DragPowerDissipation(num_nodes=nn),
                            promotes_inputs=['V', 'h', 'aa', 'Tp', 'gg'], promotes_outputs=['DV_sw', 'TV_sw' ,'Vmargin', 'V_dot'])
        self.add_subsystem('solar', solar_module.SolarPower(num_nodes=nn, start_date=solar_module.start_date, lat=solar_module.lat),
                            promotes_inputs=['h', 'time'], promotes_outputs=['Psol_sw'])
        self.add_subsystem('potential', potential_module.PotentialPower(num_nodes=nn),
                            promotes_inputs=['h_dot'], promotes_outputs=['Epot_sw'])
        # Dymos objectives take a single named variable, not an expression, so the
        # difference is computed here as its own ODE output and integrated as its own
        # state (Net_sw_int) below - same pattern as DV_sw_int/Psol_sw_int.
        self.add_subsystem('net', om.ExecComp('Net_sw = Psol_sw - DV_sw - TV_sw + Epot_sw',
                                               Net_sw={'units': 'W/m**2', 'shape': (nn,)},
                                               Psol_sw={'units': 'W/m**2', 'shape': (nn,)},
                                               DV_sw={'units': 'W/m**2', 'shape': (nn,)},
                                               TV_sw={'units': 'W/m**2', 'shape': (nn,)},
                                               Epot_sw={'units': 'W/m**2', 'shape': (nn,)}),
                            promotes=['Net_sw', 'Psol_sw', 'DV_sw', 'TV_sw', 'Epot_sw'])
        self.add_subsystem('battery', battery_module.StateOfCharge(num_nodes=nn),
                            promotes_inputs=[('net_sw', 'Net_sw')], promotes_outputs=['SOC_dot'])


def main():
    # Initialize the Problem and the optimization driver
    prob = om.Problem(model=om.Group(), name='solving_longitudinal')

    prob.driver = om.ScipyOptimizeDriver() # Or pyOptSparseDriver(optimizer='IPOPT')
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['maxiter'] = 5000
    prob.driver.options['tol'] = 1e-4
    prob.driver.opt_settings['maxiter'] = 5000
    prob.driver.opt_settings['ftol'] = 1e-4

    prob.driver.declare_coloring()

    # Create a trajectory
    traj = prob.model.add_subsystem('traj', dm.Trajectory())


    ##### Adaptative Radau considering duration of each stage
    climb = traj.add_phase('climb', dm.Phase(ode_class=LongitudinalODE, transcription=dm.Radau(num_segments=5, order=3)))
    cruise = traj.add_phase('cruise', dm.Phase(ode_class=LongitudinalODE, transcription=dm.Radau(num_segments=5, order=3)))
    descent = traj.add_phase('descent', dm.Phase(ode_class=LongitudinalODE, transcription=dm.Radau(num_segments=5, order=3)))

    # Phase1 : Climb
    climb.set_time_options(fix_initial=True, duration_bounds=(3*10*5*60, total_duration), duration_ref=1e4)
    climb.add_state('h', rate_source='h_dot', fix_initial=True, fix_final=True, units='m',
                     lower=low_altitude, upper=maximum_altitude, ref=1e4, defect_ref=1e4)
    climb.add_state('DV_sw_int', rate_source='DV_sw', fix_initial=True, fix_final=False, units='J/m**2', ref=1e6, defect_ref=1e6)
    climb.add_state('TV_sw_int', rate_source='TV_sw', fix_initial=True, fix_final=False, units='J/m**2', ref=1e6, defect_ref=1e6)
    climb.add_state('Psol_sw_int', rate_source='Psol_sw', fix_initial=True, fix_final=False, units='J/m**2', ref=5e6, defect_ref=5e6)
    climb.add_state('Net_sw_int', rate_source='Net_sw', fix_initial=True, fix_final=False, units='J/m**2', ref=5e6, defect_ref=5e6)
    climb.add_state('Epot_sw_int', rate_source='Epot_sw', fix_initial=True, fix_final=False, units='J/m**2', ref=epot_sw_int_ref, defect_ref=epot_sw_int_ref)
    # Battery state of charge (fraction, 0-1) - starts fully charged; bounds enforce it
    # can't over/undercharge, same pattern as h/V's physical lower/upper bounds.
    climb.add_state('SOC', rate_source='SOC_dot', fix_initial=True, fix_final=False, units=None, ref=1.0, defect_ref=1.0)
    climb.add_state('V', rate_source='V_dot', fix_initial=False, fix_final=False, units='m/s',
                     lower=10, upper=30, ref=10.0, defect_ref=10.0)
    climb.add_control('Tp', lower=0.0, upper=1.0)
    climb.add_control('gg', lower=np.radians(-5), upper=np.radians(5), units='rad')
    climb.add_control('aa', lower=np.radians(-5), upper=np.radians(10), units='rad')
    climb.add_boundary_constraint('gg', loc='final', equals=0.0, units='rad')  # level off before cruise
    climb.add_path_constraint('Vmargin', lower=0.0, units='m/s')  # stay above stall speed

    # Phase2 : Cruise
    cruise.set_time_options(fix_initial=False, duration_bounds=(3*10*5*60, total_duration), duration_ref=1e4)
    cruise.add_state('h', rate_source='h_dot', fix_initial=False, fix_final=False, units='m',  lower=0.0, upper=maximum_altitude, ref=1e4, defect_ref=1e4)
    cruise.add_state('DV_sw_int', rate_source='DV_sw', fix_initial=False, fix_final=False, units='J/m**2', ref=1e6, defect_ref=1e6)
    cruise.add_state('TV_sw_int', rate_source='TV_sw', fix_initial=False, fix_final=False, units='J/m**2', ref=1e6, defect_ref=1e6)
    cruise.add_state('Psol_sw_int', rate_source='Psol_sw', fix_initial=False, fix_final=False, units='J/m**2', ref=5e6, defect_ref=5e6)
    cruise.add_state('Net_sw_int', rate_source='Net_sw', fix_initial=False, fix_final=False, units='J/m**2', ref=5e6, defect_ref=5e6)
    cruise.add_state('Epot_sw_int', rate_source='Epot_sw', fix_initial=False, fix_final=False, units='J/m**2', ref=epot_sw_int_ref, defect_ref=epot_sw_int_ref)
    cruise.add_state('SOC', rate_source='SOC_dot', fix_initial=False, fix_final=False, units=None, ref=1.0, defect_ref=1.0)
    cruise.add_state('V', rate_source='V_dot', fix_initial=False, fix_final=False, units='m/s',
                      lower=10, upper=30, ref=10.0, defect_ref=10.0)
    cruise.add_control('Tp', lower=0.0, upper=1.0)
    cruise.add_control('aa', lower=np.radians(-5), upper=np.radians(10), units='rad')
    cruise.add_parameter('gg', val=0.0, opt=False, units='rad')
    cruise.add_path_constraint('Vmargin', lower=0.0, units='m/s')  # stay above stall speed
    cruise.add_timeseries_output('gg')

    # Phase3 : Descent
    descent.set_time_options(fix_initial=False, duration_bounds=(3*10*5*60, total_duration), duration_ref=1e4)
    descent.add_state('h', rate_source='h_dot', fix_initial=False, fix_final=True, units='m',
                       lower=low_altitude, upper=maximum_altitude, ref=1e4, defect_ref=1e4)
    descent.add_state('DV_sw_int', rate_source='DV_sw', fix_initial=False, fix_final=False, units='J/m**2', ref=1e6, defect_ref=1e6)
    descent.add_state('TV_sw_int', rate_source='TV_sw', fix_initial=False, fix_final=False, units='J/m**2', ref=1e6, defect_ref=1e6)
    descent.add_state('Psol_sw_int', rate_source='Psol_sw', fix_initial=False, fix_final=False, units='J/m**2', ref=5e6, defect_ref=5e6)
    descent.add_state('Net_sw_int', rate_source='Net_sw', fix_initial=False, fix_final=False, units='J/m**2', ref=5e6, defect_ref=5e6)
    descent.add_state('Epot_sw_int', rate_source='Epot_sw', fix_initial=False, fix_final=False, units='J/m**2', ref=epot_sw_int_ref, defect_ref=epot_sw_int_ref)
    descent.add_state('SOC', rate_source='SOC_dot', fix_initial=False, fix_final=False, units=None, ref=1.0, defect_ref=1.0)
    descent.add_state('V', rate_source='V_dot', fix_initial=False, fix_final=False, units='m/s',
                       lower=10, upper=30, ref=10.0, defect_ref=10.0)
    descent.add_control('Tp', lower=0.0, upper=1.0)
    descent.add_control('gg', lower=np.radians(-5), upper=np.radians(5), units='rad')
    descent.add_control('aa', lower=np.radians(-5), upper=np.radians(10), units='rad')
    descent.add_boundary_constraint('time', loc='final', equals=total_duration, units='s', ref=1e4)
    descent.add_path_constraint('Vmargin', lower=0.0, units='m/s')  # stay above stall speed
    # descent.add_objective('DV_sw_int', loc='final', ref=1e1)
    # descent.add_objective('Psol_sw_int', loc='final', ref=-1e5)
    descent.add_objective('Net_sw_int', loc='final', ref=-5e6)  # maximize Net_sw = Psol_sw - DV_sw + Epot_sw

    traj.link_phases(phases=['climb', 'cruise', 'descent'],
                      vars=['h', 'time', 'DV_sw_int', 'TV_sw_int', 'Psol_sw_int', 'Net_sw_int', 'Epot_sw_int', 'SOC', 'V', 'aa', 'Tp'])

    prob.model.linear_solver = om.DirectSolver()

    # Setup the Problem
    prob.setup()

    # Set the initial values -> Provide a better initial guess
    climb.set_time_val(initial=0.0, duration=climb_duration_guess)
    climb.set_state_val('h', [initial_altitude, cruise_altitude])
    climb.set_state_val('DV_sw_int', [0, climb_duration_guess * dv_sw_guess_rate])
    climb.set_state_val('TV_sw_int', [0, climb_duration_guess * tv_sw_guess_rate])
    climb.set_state_val('Psol_sw_int', [0, climb_duration_guess * psol_sw_guess_rate])
    climb.set_state_val('Net_sw_int', [0, climb_duration_guess * net_sw_guess_rate])
    climb.set_state_val('Epot_sw_int', [0, climb_duration_guess * epot_sw_climb_guess_rate])
    climb.set_state_val('SOC', [soc_initial, soc_initial + climb_duration_guess * net_sw_guess_rate / battery_max_energy])
    climb.set_state_val('V', [15, 25])
    climb.set_control_val('Tp', [0.7, 0.7])
    climb.set_control_val('gg', [np.radians(2), 0.0])
    climb.set_control_val('aa', [np.radians(2), np.radians(2)])

    dv_sw_int_climb_end = climb_duration_guess * dv_sw_guess_rate
    tv_sw_int_climb_end = climb_duration_guess * tv_sw_guess_rate
    psol_sw_int_climb_end = climb_duration_guess * psol_sw_guess_rate
    net_sw_int_climb_end = climb_duration_guess * net_sw_guess_rate
    epot_sw_int_climb_end = climb_duration_guess * epot_sw_climb_guess_rate
    soc_climb_end = soc_initial + net_sw_int_climb_end / battery_max_energy
    cruise.set_time_val(initial=climb_duration_guess, duration=cruise_duration_guess)
    cruise.set_state_val('h', [cruise_altitude, cruise_altitude])
    cruise.set_state_val('DV_sw_int', [dv_sw_int_climb_end, dv_sw_int_climb_end + cruise_duration_guess * dv_sw_guess_rate])
    cruise.set_state_val('TV_sw_int', [tv_sw_int_climb_end, tv_sw_int_climb_end + cruise_duration_guess * tv_sw_guess_rate])
    cruise.set_state_val('Psol_sw_int', [psol_sw_int_climb_end, psol_sw_int_climb_end + cruise_duration_guess * psol_sw_guess_rate])
    cruise.set_state_val('Net_sw_int', [net_sw_int_climb_end, net_sw_int_climb_end + cruise_duration_guess * net_sw_guess_rate])
    # Level cruise: h_dot ~ 0, so Epot_sw_int stays ~constant at the value climb ended with
    cruise.set_state_val('Epot_sw_int', [epot_sw_int_climb_end, epot_sw_int_climb_end])
    cruise.set_state_val('SOC', [soc_climb_end, soc_climb_end + cruise_duration_guess * net_sw_guess_rate / battery_max_energy])
    cruise.set_state_val('V', [25, 25])
    cruise.set_control_val('Tp', [0.4, 0.4])
    cruise.set_control_val('aa', [np.radians(0), np.radians(0)])

    dv_sw_int_cruise_end = dv_sw_int_climb_end + cruise_duration_guess * dv_sw_guess_rate
    tv_sw_int_cruise_end = tv_sw_int_climb_end + cruise_duration_guess * tv_sw_guess_rate
    psol_sw_int_cruise_end = psol_sw_int_climb_end + cruise_duration_guess * psol_sw_guess_rate
    net_sw_int_cruise_end = net_sw_int_climb_end + cruise_duration_guess * net_sw_guess_rate
    epot_sw_int_cruise_end = epot_sw_int_climb_end
    soc_cruise_end = soc_climb_end + cruise_duration_guess * net_sw_guess_rate / battery_max_energy
    descent.set_time_val(initial=climb_duration_guess + cruise_duration_guess, duration=descent_duration_guess)
    descent.set_state_val('h', [cruise_altitude, initial_altitude])
    descent.set_state_val('DV_sw_int', [dv_sw_int_cruise_end, dv_sw_int_cruise_end + descent_duration_guess * dv_sw_guess_rate])
    descent.set_state_val('TV_sw_int', [tv_sw_int_cruise_end, tv_sw_int_cruise_end + descent_duration_guess * tv_sw_guess_rate])
    descent.set_state_val('Psol_sw_int', [psol_sw_int_cruise_end, psol_sw_int_cruise_end + descent_duration_guess * psol_sw_guess_rate])
    descent.set_state_val('Net_sw_int', [net_sw_int_cruise_end, net_sw_int_cruise_end + descent_duration_guess * net_sw_guess_rate])
    descent.set_state_val('Epot_sw_int', [epot_sw_int_cruise_end, epot_sw_int_cruise_end + descent_duration_guess * epot_sw_descent_guess_rate])
    descent.set_state_val('SOC', [soc_cruise_end, soc_cruise_end + descent_duration_guess * net_sw_guess_rate / battery_max_energy])
    descent.set_state_val('V', [25, 15])
    descent.set_control_val('Tp', [0.2, 0.2])
    descent.set_control_val('gg', [0.0, np.radians(-2)])
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
    print('Integral of TV_sw over mission (J/m^2):', prob.get_val('traj.descent.timeseries.TV_sw_int')[-1, 0])
    print('Integral of Psol_sw over mission (J/m^2):', prob.get_val('traj.descent.timeseries.Psol_sw_int')[-1, 0])
    print('Integral of Net_sw (Psol_sw - DV_sw - TV_sw) over mission (J/m^2):', prob.get_val('traj.descent.timeseries.Net_sw_int')[-1, 0])
    # Should end up ~0 since h starts and ends at 0 - climb's gain and descent's loss should cancel
    print('Integral of Epot_sw (potential energy) over mission (J/m^2):', prob.get_val('traj.descent.timeseries.Epot_sw_int')[-1, 0])
    print('Final battery state of charge:', prob.get_val('traj.descent.timeseries.SOC')[-1, 0])

    # Records
    solution_path = str(prob.get_outputs_dir() / solution_record_file)
    simulation_path = str(prob.get_outputs_dir() / 'traj_simulation_0_out' / simulation_record_file)
    with open(paths_file, 'w') as f:
        json.dump({'solution': solution_path, 'simulation': simulation_path}, f, indent=2)

    return prob, exp_out


if __name__ == '__main__':
    main()
    plot_longitudinal.main()
