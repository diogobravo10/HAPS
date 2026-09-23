import json
import numpy as np
import openmdao.api as om
import dymos as dm
import module_longitudinal_kinematics as kinematics
import module_aero
import module_solar
import module_potential
import module_battery
import time
from datetime import datetime


# Mission parameters - kept as module-level defaults (rather than only inside main()'s
# signature) since solving_cascade_V3.py reads solving_longitudinal.total_duration
# directly, to size the lateral phase's duration without duplicating the constant.
low_altitude = 10000.0  # m -> Lower altitudes -> Lower stall speed -> Decrease in Speed -> Decrease in energy consumption (V***3)
initial_altitude = 12000.0  # m 
cruise_altitude = 17000.0  # m
maximum_altitude = 24000.0 # m
total_duration = 24 * 3600  # s - single combined duration bound for all 3 stages

paths_file = 'solving_longitudinal_paths.json'


class LongitudinalODE(om.Group):
    """Flight-path kinematics plus aero_module's drag-power-dissipation and
    solar_module's solar-power subsystems.

    'V' (state) is promoted from kinematics/aero so they share the same airspeed; 'h'
    (state) is promoted from aero/solar so they share the same altitude; 'gg' (flight-
    path angle, state in climb/descent) is promoted from kinematics/aero so they share
    it too - aero needs it for V_dot's gravity component, same units ('rad') on both.
    'aa' (angle of attack control) and 'Tp' (throttle control) feed aero only. 'time' is
    Dymos's auto-supplied absolute phase time, promoted so solar can use it. Exposes 'h_dot'
    (for the h state), 'V_dot' (for the V state), 'gg_dot' (for the gg state in
    climb/descent - cruise keeps 'gg' as a fixed parameter instead), 'DV_sw' (for the
    DV_sw_int integral state), 'Psol_sw' (for the Psol_sw_int integral state), 'Net_sw'
    (Psol_sw - DV_sw, for the Net_sw_int integral state - net energy collected minus
    dissipated), 'Epot_sw' (for the Epot_sw_int integral state - potential energy
    gained/lost while climbing/descending, tracked separately from Net_sw for now),
    'SOC_dot' (for the SOC state - battery state of charge, driven by the same Net_sw),
    and 'Vstall'/'Vmargin' (diagnostic outputs, not integrated into any state).

    All physical parameters used by the aero/potential/battery/solar subsystems are
    declared as options here and forwarded to each subsystem's own constructor, rather
    than each module hardcoding its own copy - main() passes them through Dymos's
    ode_init_kwargs, so a single main() call fully determines the aircraft/mission this
    ODE represents (e.g. for an outer optimization sweeping M_sw/mbat_sw, or cycling
    start_date/lat).
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        # Aero + potential (M_sw and g are shared between the two subsystems)
        self.options.declare('g', default=9.81, types=(int, float), desc='Gravitational acceleration [m/s^2]')
        self.options.declare('chord', default=1.0, types=(int, float), desc='Reference chord [m]')
        self.options.declare('M_sw', default=3.0, types=(int, float), desc='Wing loading (mass per unit wing area) [kg/m^2]')
        self.options.declare('CLmax', default=1.2, types=(int, float), desc='Max lift coefficient, for stall speed')
        self.options.declare('Tinst_sw', default=10.0, types=(int, float), desc='Installed thrust per unit wing area [N/m^2]')
        self.options.declare('mu_prop', default=0.7, types=(int, float), desc='Propeller efficiency')
        # Battery
        self.options.declare('mbat_sw', default=2.0, types=(int, float), desc='Battery mass per unit wing area [kg/m^2]')
        self.options.declare('mb', default=450.0, types=(int, float), desc='Battery energy density [Wh/kg]')
        self.options.declare('mu_e', default=0.9, types=(int, float), desc='Energy management system efficiency')
        self.options.declare('mu_LS', default=0.9, types=(int, float), desc='LS-battery efficiency')
        # Solar
        self.options.declare('start_date', default=datetime(2012, 6, 1, 6, 0), types=datetime)
        self.options.declare('lat', default=37.5, types=(int, float))
        self.options.declare('solar_cell_efficiency', default=0.15, types=(int, float))
        self.options.declare('vnorm', default=np.array([0, 0, -1]))

    def setup(self):
        nn = self.options['num_nodes']
        opts = self.options

        self.add_subsystem('kinematics', kinematics.Kinematics(num_nodes=nn),
                            promotes_inputs=['V', 'gg'], promotes_outputs=['h_dot'])
        self.add_subsystem('aero', module_aero.DragPowerDissipation(
                                num_nodes=nn, g=opts['g'], chord=opts['chord'], M_sw=opts['M_sw'],
                                CLmax=opts['CLmax'], Tinst_sw=opts['Tinst_sw'], mu_prop=opts['mu_prop']),
                            promotes_inputs=['V', 'h', 'aa', 'Tp', 'gg'], promotes_outputs=['DV_sw', 'TV_sw' ,'Vmargin', 'V_dot', 'gg_dot'])
        self.add_subsystem('solar', module_solar.SolarPower(
                                num_nodes=nn, start_date=opts['start_date'], lat=opts['lat'],
                                solar_cell_efficiency=opts['solar_cell_efficiency'], vnorm=opts['vnorm']),
                            promotes_inputs=['h', 'time'], promotes_outputs=['Psol_sw'])
        self.add_subsystem('potential', module_potential.PotentialPower(num_nodes=nn, g=opts['g'], M_sw=opts['M_sw']),
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
        self.add_subsystem('battery', module_battery.StateOfCharge(
                                num_nodes=nn, mbat_sw=opts['mbat_sw'], mb=opts['mb'], mu_e=opts['mu_e'], mu_LS=opts['mu_LS']),
                            promotes_inputs=['SOC', 'Net_sw'], promotes_outputs=['SOC_dot'])


def main(*, M_sw=3.7, mbat_sw=2.0, start_date=datetime(2012, 6, 1, 6, 0), lat=37.5,
         g=9.81, chord=1.0, CLmax=1.2, Tinst_sw=10.0, mu_prop=0.7,
         mb=450.0, mu_e=0.9, mu_LS=0.9, soc_initial=0.2,
         solar_cell_efficiency=0.15, vnorm=None,
         low_altitude=low_altitude, initial_altitude=initial_altitude,
         cruise_altitude=cruise_altitude, maximum_altitude=maximum_altitude,
         total_duration=total_duration):
    """Build and solve the climb/cruise/descent longitudinal trajectory.

    M_sw and mbat_sw are meant as the outer optimization's design variables;
    start_date and lat are meant to be cycled (e.g. over a season/latitude sweep);
    everything else is a fixed, user-supplied parameter. All of them are plain
    keyword arguments (module-level constants only supply their defaults) so an
    outer loop can call main(M_sw=..., mbat_sw=..., start_date=..., lat=...)
    repeatedly without touching global state.
    """
    if vnorm is None:
        vnorm = np.array([0, 0, -1])

    # Bundles every physical parameter the ODE's subsystems need, forwarded to each
    # of the three phases below via ode_init_kwargs - the single place that determines
    # which aircraft/mission/date this particular solve represents.
    ode_kwargs = dict(g=g, chord=chord, M_sw=M_sw, CLmax=CLmax, Tinst_sw=Tinst_sw, mu_prop=mu_prop,
                       mbat_sw=mbat_sw, mb=mb, mu_e=mu_e, mu_LS=mu_LS,
                       start_date=start_date, lat=lat,
                       solar_cell_efficiency=solar_cell_efficiency, vnorm=vnorm)

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
    epot_sw_climb_guess_rate = M_sw * g * climb_altitude_gain / climb_duration_guess
    epot_sw_descent_guess_rate = -M_sw * g * climb_altitude_gain / descent_duration_guess
    # Peak magnitude of Epot_sw_int over the mission (used to scale that state) - the state
    # itself dips to ~0 at climb start/descent end, but ranges over roughly this much in between.
    epot_sw_int_ref = M_sw * g * climb_altitude_gain

    # Battery energy capacity per unit wing area, same formula as module_battery.StateOfCharge
    # uses internally - needed here only to seed SOC's initial guess from Net_sw_int's guess.
    battery_max_energy = mb * 3600 * mbat_sw

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
    climb = traj.add_phase('climb', dm.Phase(ode_class=LongitudinalODE, ode_init_kwargs=ode_kwargs,
                                              transcription=dm.Radau(num_segments=5, order=3)))
    cruise = traj.add_phase('cruise', dm.Phase(ode_class=LongitudinalODE, ode_init_kwargs=ode_kwargs,
                                                transcription=dm.Radau(num_segments=5, order=3)))
    descent = traj.add_phase('descent', dm.Phase(ode_class=LongitudinalODE, ode_init_kwargs=ode_kwargs,
                                                  transcription=dm.Radau(num_segments=5, order=3)))

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
    climb.add_state('gg', rate_source='gg_dot', fix_initial=False, fix_final=False, units='rad',
                     lower=np.radians(-5), upper=np.radians(5), ref=0.1, defect_ref=0.1)
    climb.add_control('aa', lower=np.radians(-5), upper=np.radians(10), units='rad')
    climb.add_boundary_constraint('gg', loc='final', equals=0.0, units='rad')  # level off before cruise
    climb.add_path_constraint('Vmargin', lower=0.0, units='m/s')  # stay above stall speed
    climb.add_timeseries_output('V_dot')  # commented out for debugging
    climb.add_timeseries_output('gg_dot')

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
    # cruise.add_path_constraint('gg_dot', lower=0.005, upper=0.005,  units='rad/s')  # stay above stall speed
    cruise.add_timeseries_output('gg')
    cruise.add_timeseries_output('V_dot')  # commented out for debugging
    cruise.add_timeseries_output('gg_dot')

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
    descent.add_state('gg', rate_source='gg_dot', fix_initial=False, fix_final=False, units='rad',
                       lower=np.radians(-5), upper=np.radians(5), ref=0.1, defect_ref=0.1)
    descent.add_control('aa', lower=np.radians(-5), upper=np.radians(10), units='rad')
    descent.add_boundary_constraint('time', loc='final', equals=total_duration, units='s', ref=1e4)
    descent.add_path_constraint('Vmargin', lower=0.0, units='m/s')  # stay above stall speed
    descent.add_timeseries_output('V_dot')  # commented out for debugging
    descent.add_timeseries_output('gg_dot')
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
    climb.set_state_val('gg', [np.radians(2), 0.0])
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
    descent.set_state_val('gg', [0.0, np.radians(-2)])
    descent.set_control_val('aa', [np.radians(2), np.radians(2)])

    start_time = time.perf_counter()

    # Solve for the optimal trajectory
    solution_record_file = 'solution.db'
    dm.run_problem(prob, solution_record_file=solution_record_file)
    # dm.run_problem(prob, refine_method='hp', refine_iteration_limit=1, solution_record_file=solution_record_file)

    elapsed = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed:.2f} seconds")

    # Generate the explicitly simulated trajectory - comment out the traj.simulate() call
    # (e.g. while iterating on the solve itself) to skip it; plot_longitudinal then just
    # plots the solution, with no simulation curves and no warnings.
    exp_out = None
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

    # Records - 'simulation' is only included if traj.simulate() actually ran above;
    # plot_longitudinal.py treats its absence as "solution only, no simulation curves".
    solution_path = str(prob.get_outputs_dir() / solution_record_file)
    paths = {'solution': solution_path}
    if exp_out is not None:
        paths['simulation'] = str(prob.get_outputs_dir() / 'traj_simulation_0_out' / simulation_record_file)
    with open(paths_file, 'w') as f:
        json.dump(paths, f, indent=2)

    return prob, exp_out


if __name__ == '__main__':
    # Imported here rather than at module level: plot_longitudinal imports paths_file
    # back from this module, and importing it up top (before paths_file is defined
    # below) breaks any plain `import solving_longitudinal` from another script (e.g.
    # solving_cascade_V3.py) with a circular-import ImportError.
    import plot_longitudinal
    main()
    plot_longitudinal.main()
