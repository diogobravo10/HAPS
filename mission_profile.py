import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
from ambiance import Atmosphere
from datetime import datetime
import _utilities as utils
import time


# Reference date/time and latitude for the solar model (t=0 of the phase)
start_date = datetime(2012, 6, 1, 6, 0)
lat = 37.5

# Aircraft constants (shared between the ODE and the post-run diagnostic plots).
# Physics is normalized per unit wing area via the wing loading M_Sw [kg/m^2], so the
# whole energy balance - and the optimization objective E - is expressed per m^2 of
# wing (W/m^2, J/m^2) instead of depending on an absolute mass and wing area.
M_Sw = 3.0  # [kg/m^2] wing loading
g = 9.81
CD, CLmax = 0.0708, 1.5
eta_prop, solar_cell_efficiency = 0.7, 0.15
mb = 450     # [Wh/kg] energy density of the LS-battery
mu_LS = 0.9  # [-] round-trip efficiency of the LS-battery

SOLUTION_DB = 'mission_profile_out/dymos_solution.db'


class SolarAircraftODE(om.ExplicitComponent):
    def initialize(self):
        # Dymos requires num_nodes to vectorize calculations across the trajectory grid
        self.options.declare('num_nodes', types=int)
        self.options.declare('start_date', types=datetime)
        self.options.declare('lat', types=(int, float))
        # Fixed climb/sink rate for this phase: positive for climb, negative for descent,
        # 0.0 for a level cruise. Leave as None to make dhdt itself a free control instead
        # (the design variable) - gg (flight-path angle) is then always just a derived
        # diagnostic of whatever dhdt and V happen to be, in either mode.
        self.options.declare('dhdt_fixed', types=float, allow_none=True, default=None)

    def setup(self):
        nn = self.options['num_nodes']
        dhdt_is_control = self.options['dhdt_fixed'] is None

        # Inputs: Controls
        self.add_input('V', val=np.ones(nn) * 15.0, units='m/s', desc='Airspeed (control)')
        self.add_input('Vdot', val=np.zeros(nn), units='m/s**2', desc='Airspeed rate (control rate)')
        # Auto-connected by Dymos: absolute phase time and current altitude/energy states
        self.add_input('time', val=np.zeros(nn), units='s')
        self.add_input('h', val=np.ones(nn) * 12000.0, units='m')
        self.add_input('E', val=np.zeros(nn), units='J/m**2', desc='Accumulated net energy per unit wing area (state)')
        if dhdt_is_control:
            self.add_input('dhdt', val=np.zeros(nn), units='m/s', desc='Rate of climb/descent (control)')

        # Outputs: Derivatives for Dymos. The state 'h' rates off 'hdot' rather than 'dhdt'
        # directly so that 'dhdt' can be an *input* (control) above without name colliding
        # with an output of the same name.
        self.add_output('hdot', val=np.zeros(nn), units='m/s', desc='Rate of change of altitude (state rate source)')
        self.add_output('E_dot', val=np.zeros(nn), units='W/m**2', desc='Net power accumulation per unit wing area')
        self.add_output('V_margin', val=np.zeros(nn), units='m/s',
                         desc='Airspeed margin above stall speed (must stay >= 0)')
        self.add_output('gg', val=np.zeros(nn), units='rad',
                         desc='Flight-path angle implied by dhdt and V (diagnostic only, not a control)')
        self.add_output('Mbat_Sw', val=np.zeros(nn), units='kg/m**2',
                         desc='Battery mass per unit wing area needed to store the accumulated energy E')
        self.add_output('P_in', val=np.zeros(nn), units='W/m**2',
                         desc='Incident solar power density (diagnostic only)')

        # Every output here depends only on its own node's inputs - there's no cross-node
        # coupling anywhere in this ODE - so the true Jacobian is diagonal, not dense.
        # declare_coloring() below discovers that sparsity itself (via a few dense probe
        # evaluations at setup time) and collapses FD from one perturbation per node to a
        # small constant number of "colors" - each covering all nn nodes of an input at
        # once, since none of them overlap. NOTE: don't also pass rows=/cols= here - on
        # this OpenMDAO version (3.45.0) that combination corrupts the subjac's cached
        # value once coloring is active (`ValueError: object too deep for desired array`
        # in dictionary_jacobian's apply_fwd/apply_rev), reproduced even in a two-line
        # component with no connection to this ODE's actual physics. Let declare_coloring
        # find the sparsity on its own instead.
        self.declare_partials('*', '*', method='fd')
        # The solar model is quantized to whole minutes internally (solarpy's
        # hour_angle only reads date.hour/date.minute), so the default FD
        # step on 'time' (~1e-6 * a few thousand seconds) is far smaller than
        # 60s and would yield a numerically zero d(E_dot)/d(time).
        self.declare_partials('E_dot', 'time', method='fd', step=60.0, step_calc='abs')
        self.declare_coloring(method='fd', show_summary=True)


    def compute(self, inputs, outputs):

        t = inputs['time']
        h = inputs['h']
        E = inputs['E']
        V = inputs['V']
        Vdot = inputs['Vdot']

        rho = Atmosphere(h).density

        # Battery mass per unit wing area needed to store the accumulated energy E: convert
        # J/m^2 -> Wh/m^2, then divide by the battery's round-trip efficiency and energy
        # density (same mass = energy/(mu_LS*mb) relation used for battery sizing elsewhere
        # in this project, e.g. _utilities.py's battery_mass calculations).
        outputs['Mbat_Sw'] = (E / 3600.0) / (mu_LS * mb)

        # Kinematic Dynamics: dhdt is either a free control (this phase's design variable)
        # or a fixed constant for this phase; either way, gg is just the flight-path angle
        # that dhdt and V imply, not something chosen directly.
        dhdt_fixed = self.options['dhdt_fixed']
        dhdt = inputs['dhdt'] if dhdt_fixed is None else np.full_like(V, dhdt_fixed)
        outputs['hdot'] = dhdt
        outputs['gg'] = np.arcsin(np.clip(dhdt / V, -1.0, 1.0))

        # Solar Power Input - already a power density (W/m^2), so no wing area to multiply by
        P_in = utils.instantaneous_power_density(h, self.options['lat'], self.options['start_date'], t, solar_cell_efficiency=solar_cell_efficiency)
        outputs['P_in'] = P_in

        # Aerodynamic Drag & Required Thrust Power, per unit wing area.
        # Lift = weight => CL = (m/S)*g / (0.5*rho*V^2) = M_Sw*g / (0.5*rho*V^2): CL depends
        # on wing loading, it is NOT simply "drop S like the drag term below".
        CL = M_Sw*g / (0.5 * rho * V**2)
        # Drag per unit area D/S = 0.5*rho*V^2*CD - here S genuinely cancels out (drag
        # scales with S, and we're dividing by that same S), unlike CL/CD and the PE/KE
        # terms below, which depend on wing loading (M_Sw) rather than on S directly.
        D_per_area = 0.5 * rho * V**2 * CD

        # Stall-speed margin: airspeed must stay >= stall speed, from CL=CLmax in the CL
        # relation above (Raymer 1999 suggests a 1.2x safety factor on top of this; adjust
        # the threshold here if you want that margin back)
        V_stall = np.sqrt(2*M_Sw*g / (rho * CLmax))
        outputs['V_margin'] = V - V_stall

        # Net power per unit wing area available for battery storage, incl. kinetic
        # energy rate d(V^2/2)/dt = V*Vdot
        outputs['E_dot'] = P_in - D_per_area*V - M_Sw*g*dhdt - M_Sw*V*Vdot
        # print("compute(): t = [{:.1f}, {:.1f}] s, h = [{:.1f}, {:.1f}] m, V = [{:.1f}, {:.1f}] m/s, E_dot = [{:.1f}, {:.1f}] W/m^2".format(
            # t.min(), t.max(), h.min(), h.max(), V.min(), V.max(), outputs['E_dot'].min(), outputs['E_dot'].max()))



#################################################################################################
#
#
#
#
#       INCREASING ALTITUDES HELP WITH BATTERY PERFORMANCE -> PARAMETRIC STUDY
#
#
#
#
#################################################################################################

MISSION_DURATION = 24*3600.0  # total climb+cruise+descent mission length, fixed [s]
START_ALTITUDE = 15000.0      # [m]
CHECKPOINT_ALTITUDE = 24000.0  # [m] must be reached at some (free) time during the mission
FINAL_ALTITUDE = 15000.0      # [m] same as start altitude


MBAT_SW_INITIAL = 2.0  # [kg/m^2] battery mass already charged at the start of the mission
E_INITIAL = MBAT_SW_INITIAL * 3600.0 * mu_LS * mb  # [J/m^2] equivalent starting value of E

# Neither climb nor descent has a prescribed rate anymore - dhdt is a free control in both
# (bounded via gg's +-5deg path constraint above), so their durations are genuinely unknown
# ahead of time. Everything below is just an initial guess for the solver, not exact values.
CLIMB_DURATION_GUESS = 10*3600.0
DESCENT_RATE_GUESS = -1.0  # [m/s] seeds descent's dhdt control and duration guess
DESCENT_DURATION_GUESS = (CHECKPOINT_ALTITUDE - FINAL_ALTITUDE) / -DESCENT_RATE_GUESS
CRUISE_DURATION_GUESS = MISSION_DURATION - CLIMB_DURATION_GUESS - DESCENT_DURATION_GUESS


def _make_phase(dhdt_fixed=None, fix_initial=False, num_segments=6):
    """Build a Phase using the shared ODE, airspeed control, and stall-margin constraint.

    `dhdt_fixed`: pass a float (climb/sink rate in m/s, 0.0 for level) to keep dhdt fixed
    in this phase; pass None to make dhdt itself a free control instead - gg (flight-path
    angle) is always just a derived diagnostic of dhdt and V, never a control directly.

    `fix_initial` should be True only for the first phase of the trajectory, whose starting
    h/E are genuinely known in advance; a downstream phase gets its initial h/E from
    `traj.link_phases(...)` continuity instead, so they stay free design variables.
    """
    phase = dm.Phase(
        ode_class=SolarAircraftODE,
        transcription=dm.Radau(num_segments=num_segments, order=3),
        ode_init_kwargs={'start_date': start_date, 'lat': lat, 'dhdt_fixed': dhdt_fixed},
    )

    # lower/upper here are hard box bounds on the state itself, enforced by the optimizer at
    # every iterate (unlike the path constraint below, which is only satisfied at convergence)
    # - this is what stops SLSQP from wandering into physically impossible altitudes (e.g.
    # negative h, or h beyond solarpy/ambiance's valid range) while it searches for a step.
    # rate_source is 'hdot', not 'dhdt' - see SolarAircraftODE.setup() for why.
    phase.add_state('h', rate_source='hdot', fix_initial=fix_initial, units='m',
                     lower=0.0, upper=30000.0)
    phase.add_state('E', rate_source='E_dot', fix_initial=fix_initial, units='J/m**2')  # Net energy-per-area integral - auxiliary state variable

    # V is always a free control, subject to the stall-margin constraint below. When
    # dhdt_fixed is None, dhdt (climb/sink rate) becomes a second free control too.
    phase.add_control('V', lower=5.0, upper=30.0, units='m/s', rate_targets=['Vdot'])
    if dhdt_fixed is None:
        phase.add_control('dhdt', lower=-3.0, upper=3.0, units='m/s')

    phase.add_path_constraint('V_margin', lower=0.0, units='m/s')
    phase.add_path_constraint('h', lower=10000.0, units='m')
    # gg is derived (dhdt/V), not a control, so bounding it is a path constraint rather
    # than a control lower=/upper= box bound.
    phase.add_path_constraint('gg', lower=np.radians(-5), upper=np.radians(5), units='rad')
    # Physical battery capacity limit: the aircraft can't store more energy than a 2 kg/m^2
    # battery can hold, at any point in the mission, not just at the end.
    phase.add_path_constraint('Mbat_Sw', upper=15.0, units='kg/m**2')
    phase.add_timeseries_output('gg')  # always a derived diagnostic now, never a control
    phase.add_timeseries_output('Mbat_Sw')
    phase.add_timeseries_output('P_in')
    phase.add_timeseries_output('E_dot')

    return phase


def build_problem():
    """Construct and configure the 3-phase (climb, cruise, descent) Dymos trajectory problem
    (does not run it).

    Phases 'climb' and 'descent' both have dhdt as a free control (bounded indirectly via
    gg's +-5deg path constraint) - their climb/sink rate, and so their duration, is
    optimized rather than prescribed (gg is derived from dhdt and V, not chosen directly).
    Phase 'cruise' holds CHECKPOINT_ALTITUDE (dhdt=0) for however long is left in the day -
    it's the one phase whose duration is "slack" by construction, absorbing whatever time
    climb and descent don't use, and the mission is pinned to exactly MISSION_DURATION via
    a boundary constraint on descent's final absolute time. All three phases are linked for
    continuity of time/h/E, and V is a free control (subject to the V_margin >= 0 stall
    constraint) in every phase.
    """
    prob = om.Problem()

    traj = dm.Trajectory()
    climb = _make_phase(dhdt_fixed=None, fix_initial=True)
    cruise = _make_phase(0.0, fix_initial=False)
    descent = _make_phase(dhdt_fixed=None, fix_initial=False)
    traj.add_phase('climb', climb)
    traj.add_phase('cruise', cruise)
    traj.add_phase('descent', descent)
    traj.link_phases(['climb', 'cruise', 'descent'], vars=['time', 'h', 'E'])
    prob.model.add_subsystem('traj', traj)

    prob.driver = om.ScipyOptimizeDriver()  # Or pyOptSparseDriver(optimizer='IPOPT')
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['maxiter'] = 2500  # Default is often 100
    prob.driver.options['tol'] = 1e-3
    # SLSQP-specific optimizer options (passes directly to scipy.optimize.minimize)
    prob.driver.opt_settings['maxiter'] = 2500
    prob.driver.opt_settings['ftol'] = 1e-3  # Function tolerance for convergence

    # Climb: starts the mission at t=0. Duration is genuinely free now that dhdt is a control -
    # the optimizer picks both the climb-rate profile and how long the climb takes.
    climb.set_time_options(fix_initial=True, duration_bounds=(60, MISSION_DURATION), units='s')
    climb.add_boundary_constraint('h', loc='final', equals=CHECKPOINT_ALTITUDE, units='m')

    # Cruise: picks up where the climb ends (continuity via link_phases). This is the one
    # phase whose duration is genuinely free - it absorbs whatever time is left in the day.
    cruise.set_time_options(fix_initial=False, initial_bounds=(0, MISSION_DURATION),
                             duration_bounds=(0, MISSION_DURATION), units='s')

    # Descent: picks up where the cruise ends, duration free just like climb - the
    # *mission's* total length is pinned by constraining descent's final absolute time to
    # MISSION_DURATION instead, which is what actually determines how long is left for it.
    descent.set_time_options(fix_initial=False, initial_bounds=(0, MISSION_DURATION),
                              duration_bounds=(60, MISSION_DURATION), units='s')
    descent.add_boundary_constraint('time', loc='final', equals=MISSION_DURATION, units='s')
    descent.add_boundary_constraint('h', loc='final', equals=FINAL_ALTITUDE, units='m')

    # Set Objective: Maximize the battery mass charged by the end of the flight.
    # (E, and so Mbat_Sw, is continuous across all three phases, so descent's final value
    # is the whole-mission result.) Capped at 2 kg/m^2 by the path constraint above.
    descent.add_objective('Mbat_Sw', loc='final', scaler=-1e0)

    prob.setup()

    # Set Initial Guesses. climb/descent durations are just starting guesses now that dhdt
    # makes both of them free unknowns, and cruise's guess is whatever's left over assuming
    # those climb/descent guesses hold.
    prob.set_val('traj.climb.t_duration', CLIMB_DURATION_GUESS)
    prob.set_val('traj.climb.states:h', climb.interp('h', [START_ALTITUDE, CHECKPOINT_ALTITUDE]))
    prob.set_val('traj.climb.states:E', climb.interp('E', [E_INITIAL, E_INITIAL]))
    prob.set_val('traj.climb.controls:V', 15.0)
    # Matches the average rate implied by CLIMB_DURATION_GUESS, for a consistent starting point
    prob.set_val('traj.climb.controls:dhdt', (CHECKPOINT_ALTITUDE - START_ALTITUDE) / CLIMB_DURATION_GUESS)

    prob.set_val('traj.cruise.t_initial', CLIMB_DURATION_GUESS)
    prob.set_val('traj.cruise.t_duration', CRUISE_DURATION_GUESS)
    prob.set_val('traj.cruise.states:h', cruise.interp('h', [CHECKPOINT_ALTITUDE, CHECKPOINT_ALTITUDE]))
    prob.set_val('traj.cruise.states:E', cruise.interp('E', [0, 0]))
    prob.set_val('traj.cruise.controls:V', 15.0)

    prob.set_val('traj.descent.t_initial', CLIMB_DURATION_GUESS + CRUISE_DURATION_GUESS)
    prob.set_val('traj.descent.t_duration', DESCENT_DURATION_GUESS)
    prob.set_val('traj.descent.states:h', descent.interp('h', [CHECKPOINT_ALTITUDE, FINAL_ALTITUDE]))
    prob.set_val('traj.descent.states:E', descent.interp('E', [0, 0]))
    prob.set_val('traj.descent.controls:V', 15.0)
    prob.set_val('traj.descent.controls:dhdt', DESCENT_RATE_GUESS)

    return prob


def _stacked_timeseries(source, var):
    """Concatenate a timeseries variable across the climb, cruise and descent phases, in mission order."""
    return np.concatenate([
        source.get_val(f'traj.{phase_name}.timeseries.{var}').ravel()
        for phase_name in ('climb', 'cruise', 'descent')
    ])


def plot_results(source):
    """Plot mission results from anything exposing .get_val() - a Problem (fresh run) or a Case (CaseReader)."""
    t = _stacked_timeseries(source, 'time')
    Mbat_Sw = _stacked_timeseries(source, 'Mbat_Sw')
    h = _stacked_timeseries(source, 'h')
    gg = _stacked_timeseries(source, 'gg')
    V_flown = _stacked_timeseries(source, 'V')
    P_in = _stacked_timeseries(source, 'P_in')
    E_dot = _stacked_timeseries(source, 'E_dot')

    # x-ticks at 6h, 12h, 18h, 24h - set on the bottom subplot of each figure and
    # shared to the rest via sharex=True
    hour_marks_s = [6*3600, 12*3600, 18*3600, 24*3600]
    hour_labels = ['6h', '12h', '18h', '24h']

    # Figure 1: Trajectory - altitude, glide angle and airspeed vs elapsed time (left
    # column, sharing the time x-axis), stall speed vs altitude alongside (right column,
    # spanning all 3 rows - it has its own x-axis, altitude rather than elapsed time)
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle('Trajectory')
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1])

    ax_alt = fig.add_subplot(gs[0, 0])
    ax_gg = fig.add_subplot(gs[1, 0], sharex=ax_alt)
    ax_speed = fig.add_subplot(gs[2, 0], sharex=ax_alt)
    ax_stall = fig.add_subplot(gs[:, 1])

    ax_alt.plot(t, h, 'b-', linewidth=2)
    ax_alt.set_ylabel('Altitude h [m]')
    ax_alt.grid(True)
    ax_alt.set_ylim([10000, 24000])
    ax_alt.set_xlim([0, 24*3600])

    ax_gg.plot(t, np.degrees(gg), 'r-', linewidth=2)
    ax_gg.set_ylabel('Glide angle gg [deg]')
    ax_gg.grid(True)
    ax_gg.ticklabel_format(useOffset=False, style='plain')
    ax_gg.set_ylim([-5, 5])

    ax_speed.plot(t, V_flown, 'g-', linewidth=2)
    ax_speed.set_xlabel('Elapsed time [s]')
    ax_speed.set_ylabel('Speed V [m/s]')
    ax_speed.grid(True)
    ax_speed.set_xticks(hour_marks_s, hour_labels)

    # Stall speed vs altitude, compared to the flown airspeed
    rho_h = Atmosphere(h).density
    V_stall_h = np.sqrt(2*M_Sw*g / (rho_h * CLmax))

    ax_stall.plot(h, V_stall_h, 'k--', label='V_stall')
    ax_stall.plot(h, 1.2 * V_stall_h, 'r--', label='1.2 x V_stall (margin threshold)')
    ax_stall.plot(h, V_flown, 'b-', linewidth=2, label='Flown V')
    ax_stall.set_xlabel('Altitude h [m]')
    ax_stall.set_ylabel('Speed [m/s]')
    ax_stall.set_title('Stall speed vs altitude')
    ax_stall.grid(True)
    ax_stall.legend()

    plt.tight_layout()

    # Figure 2: Energy concepts - irradiance/net power and battery mass vs elapsed time
    fig2, axes2 = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    fig2.suptitle('Energy concepts')

    axes2[0].plot(t, P_in, 'orange', linewidth=2, label='Irradiance P_in')
    axes2[0].plot(t, E_dot, 'm-', linewidth=2, label='Net power E_dot')
    axes2[0].set_ylabel('Power per wing area [W/m^2]')
    axes2[0].grid(True)
    axes2[0].legend()

    axes2[1].plot(t, Mbat_Sw, 'g-', linewidth=2)
    axes2[1].set_xlabel('Elapsed time [s]')
    axes2[1].set_ylabel('Battery mass per wing area [kg/m^2]')
    axes2[1].grid(True)
    axes2[1].set_xticks(hour_marks_s, hour_labels)

    plt.tight_layout()

    plt.show()


RUN_OPTIMIZATION = True  # False -> skip the solve and re-plot the last saved solution instead


if __name__ == '__main__':
    start_time = time.perf_counter() # High-resolution start
    if RUN_OPTIMIZATION:
        prob = build_problem()
        dm.run_problem(prob, run_driver=True, make_plots=False)
        source = prob
    else:
        source = om.CaseReader(SOLUTION_DB).get_case('final')
    elapsed = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed:.2f} seconds")
    plot_results(source)
