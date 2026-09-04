import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
from ambiance import Atmosphere
from datetime import datetime
import _utilities as utils


# Reference date/time and latitude for the solar model (t=0 of the phase)
start_date = datetime(2012, 6, 1, 6, 0)
lat = 37.5

# Aircraft constants (shared between the ODE and the post-run diagnostic plots).
# Physics is normalized per unit wing area via the wing loading M_Sw [kg/m^2], so the
# whole energy balance - and the optimization objective E - is expressed per m^2 of
# wing (W/m^2, J/m^2) instead of depending on an absolute mass and wing area.
M_Sw = 3.0  # [kg/m^2] wing loading
g = 9.81
K, CD0, CLmax = 0.1, 0.011, 1.5
eta_prop, solar_cell_efficiency = 0.7, 0.15

SOLUTION_DB = 'mission_profile_out/dymos_solution.db'


class SolarAircraftODE(om.ExplicitComponent):
    def initialize(self):
        # Dymos requires num_nodes to vectorize calculations across the trajectory grid
        self.options.declare('num_nodes', types=int)
        self.options.declare('start_date', types=datetime)
        self.options.declare('lat', types=(int, float))
        # Fixed climb/sink rate for this phase: positive for climb, negative for descent,
        # 0.0 for a level cruise. dhdt no longer comes from V*sin(gg) - gg is now just a
        # derived (non-control) diagnostic of what flight-path angle that rate implies at
        # the current (optimized) V.
        self.options.declare('dhdt_fixed', types=float)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs: Controls
        self.add_input('V', val=np.ones(nn) * 15.0, units='m/s', desc='Airspeed (control)')
        self.add_input('Vdot', val=np.zeros(nn), units='m/s**2', desc='Airspeed rate (control rate)')
        # Auto-connected by Dymos: absolute phase time and current altitude (state 'h')
        self.add_input('time', val=np.zeros(nn), units='s')
        self.add_input('h', val=np.ones(nn) * 12000.0, units='m')

        # Outputs: Derivatives for Dymos
        self.add_output('dhdt', val=np.zeros(nn), units='m/s', desc='Rate of change of altitude (fixed)')
        self.add_output('E_dot', val=np.zeros(nn), units='W/m**2', desc='Net power accumulation per unit wing area')
        self.add_output('V_margin', val=np.zeros(nn), units='m/s',
                         desc='Airspeed margin above stall speed (must stay >= 0)')
        self.add_output('gg', val=np.zeros(nn), units='rad',
                         desc='Flight-path angle implied by the fixed dhdt and the current V (diagnostic only)')

        # Setup automatic derivatives via Complex Step or Finite Difference
        self.declare_partials('*', '*', method='fd')
        # The solar model is quantized to whole minutes internally (solarpy's
        # hour_angle only reads date.hour/date.minute), so the default FD
        # step on 'time' (~1e-6 * a few thousand seconds) is far smaller than
        # 60s and would yield a numerically zero d(E_dot)/d(time).
        self.declare_partials('E_dot', 'time', method='fd', step=60.0, step_calc='abs')


    def compute(self, inputs, outputs):

        t = inputs['time']
        h = inputs['h']
        V = inputs['V']
        Vdot = inputs['Vdot']

        rho = Atmosphere(h).density

        # Kinematic Dynamics: dhdt is fixed for this phase, V is free
        dhdt = np.full_like(V, self.options['dhdt_fixed'])
        outputs['dhdt'] = dhdt
        outputs['gg'] = np.arcsin(np.clip(dhdt / V, -1.0, 1.0))

        # Solar Power Input - already a power density (W/m^2), so no wing area to multiply by
        P_in = utils.instantaneous_power_density(h, self.options['lat'], self.options['start_date'], t, solar_cell_efficiency=0.75)

        # Aerodynamic Drag & Required Thrust Power, per unit wing area.
        # Lift = weight => CL = (m/S)*g / (0.5*rho*V^2) = M_Sw*g / (0.5*rho*V^2): CL depends
        # on wing loading, it is NOT simply "drop S like the drag term below".
        CL = M_Sw*g / (0.5 * rho * V**2)
        CD = CD0 + K * CL**2
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
        print("compute(): t = [{:.1f}, {:.1f}] s, h = [{:.1f}, {:.1f}] m, V = [{:.1f}, {:.1f}] m/s, E_dot = [{:.1f}, {:.1f}] W/m^2".format(
            t.min(), t.max(), h.min(), h.max(), V.min(), V.max(), outputs['E_dot'].min(), outputs['E_dot'].max()))


MISSION_DURATION = 24*3600.0  # total climb+cruise+descent mission length, fixed [s]
START_ALTITUDE = 12000.0      # [m]
CHECKPOINT_ALTITUDE = 20000.0  # [m] must be reached at some (free) time during the mission
FINAL_ALTITUDE = 12000.0      # [m] same as start altitude

CLIMB_RATE = 1.0    # [m/s] fixed rate of climb
DESCENT_RATE = -1.0  # [m/s] fixed rate of descent
CLIMB_DURATION = (CHECKPOINT_ALTITUDE - START_ALTITUDE) / CLIMB_RATE       # deterministic given CLIMB_RATE
DESCENT_DURATION = (CHECKPOINT_ALTITUDE - FINAL_ALTITUDE) / -DESCENT_RATE  # deterministic given DESCENT_RATE
CRUISE_DURATION = MISSION_DURATION - CLIMB_DURATION - DESCENT_DURATION    # absorbs whatever time is left


def _make_phase(dhdt_fixed, fix_initial=False, num_segments=10):
    """Build a Phase using the shared ODE, airspeed control, and stall-margin constraint.

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
    phase.add_state('h', rate_source='dhdt', fix_initial=fix_initial, units='m',
                     lower=0.0, upper=24000.0)
    phase.add_state('E', rate_source='E_dot', fix_initial=fix_initial, units='J/m**2')  # Net energy-per-area integral - auxiliary state variable

    # V is the only control now - dhdt (and so the climb/descent angle) is fixed per phase,
    # V is what the optimizer is free to choose, subject to the stall-margin constraint below.
    phase.add_control('V', lower=5.0, upper=40.0, units='m/s', rate_targets=['Vdot'])

    phase.add_path_constraint('V_margin', lower=0.0, units='m/s')
    phase.add_path_constraint('h', lower=10000.0, units='m')
    phase.add_timeseries_output('gg')  # diagnostic only, not a control anymore

    return phase


def build_problem():
    """Construct and configure the 3-phase (climb, cruise, descent) Dymos trajectory problem
    (does not run it).

    Phase 'climb' goes from START_ALTITUDE up to CHECKPOINT_ALTITUDE at a fixed CLIMB_RATE.
    Phase 'cruise' holds CHECKPOINT_ALTITUDE (dhdt=0) for however long is left in the day.
    Phase 'descent' goes from CHECKPOINT_ALTITUDE down to FINAL_ALTITUDE at a fixed DESCENT_RATE.
    Since dhdt is fixed in climb/descent, their durations are fully determined by the required
    altitude change - only the cruise phase's duration is actually free, and it's what absorbs
    the mission into exactly MISSION_DURATION (enforced via a boundary constraint on descent's
    final absolute time). All three phases are linked for continuity of time/h/E, and V is a
    free control (subject to the V_margin >= 0 stall constraint) in every phase.
    """
    prob = om.Problem()

    traj = dm.Trajectory()
    climb = _make_phase(CLIMB_RATE, fix_initial=True)
    cruise = _make_phase(0.0, fix_initial=False)
    descent = _make_phase(DESCENT_RATE, fix_initial=False)
    traj.add_phase('climb', climb)
    traj.add_phase('cruise', cruise)
    traj.add_phase('descent', descent)
    traj.link_phases(['climb', 'cruise', 'descent'], vars=['time', 'h', 'E'])
    prob.model.add_subsystem('traj', traj)

    prob.driver = om.ScipyOptimizeDriver()  # Or pyOptSparseDriver(optimizer='IPOPT')
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['maxiter'] = 1500  # Default is often 100
    prob.driver.options['tol'] = 1e-5
    # SLSQP-specific optimizer options (passes directly to scipy.optimize.minimize)
    prob.driver.opt_settings['maxiter'] = 1500
    prob.driver.opt_settings['ftol'] = 1e-5  # Function tolerance for convergence

    # Climb: starts the mission at t=0. Duration is nominally free but is effectively pinned
    # by CLIMB_RATE and the h boundary constraint below (h(t) = START_ALTITUDE + CLIMB_RATE*t).
    climb.set_time_options(fix_initial=True, duration_bounds=(60, MISSION_DURATION), units='s')
    climb.add_boundary_constraint('h', loc='final', equals=CHECKPOINT_ALTITUDE, units='m')

    # Cruise: picks up where the climb ends (continuity via link_phases). This is the one
    # phase whose duration is genuinely free - it absorbs whatever time is left in the day.
    cruise.set_time_options(fix_initial=False, initial_bounds=(0, MISSION_DURATION),
                             duration_bounds=(0, MISSION_DURATION), units='s')

    # Descent: picks up where the cruise ends, duration effectively pinned by DESCENT_RATE
    # and the h boundary constraint below, same as climb. The *mission's* total length is
    # pinned by constraining descent's final absolute time to MISSION_DURATION.
    descent.set_time_options(fix_initial=False, initial_bounds=(0, MISSION_DURATION),
                              duration_bounds=(60, MISSION_DURATION), units='s')
    descent.add_boundary_constraint('time', loc='final', equals=MISSION_DURATION, units='s')
    descent.add_boundary_constraint('h', loc='final', equals=FINAL_ALTITUDE, units='m')

    # Set Objective: Maximize the mission's total accumulated net energy [Eq. (16)]
    # (E is continuous across all three phases, so descent's final E is the whole-mission total)
    descent.add_objective('E', loc='final', scaler=-1e1)

    prob.setup()

    # Set Initial Guesses. climb/descent durations are set to their exact deterministic
    # values (given the fixed rates), since there's no ambiguity there for the solver.
    prob.set_val('traj.climb.t_duration', CLIMB_DURATION)
    prob.set_val('traj.climb.states:h', climb.interp('h', [START_ALTITUDE, CHECKPOINT_ALTITUDE]))
    prob.set_val('traj.climb.states:E', climb.interp('E', [0, 0]))
    prob.set_val('traj.climb.controls:V', 15.0)

    prob.set_val('traj.cruise.t_initial', CLIMB_DURATION)
    prob.set_val('traj.cruise.t_duration', CRUISE_DURATION)
    prob.set_val('traj.cruise.states:h', cruise.interp('h', [CHECKPOINT_ALTITUDE, CHECKPOINT_ALTITUDE]))
    prob.set_val('traj.cruise.states:E', cruise.interp('E', [0, 0]))
    prob.set_val('traj.cruise.controls:V', 15.0)

    prob.set_val('traj.descent.t_initial', CLIMB_DURATION + CRUISE_DURATION)
    prob.set_val('traj.descent.t_duration', DESCENT_DURATION)
    prob.set_val('traj.descent.states:h', descent.interp('h', [CHECKPOINT_ALTITUDE, FINAL_ALTITUDE]))
    prob.set_val('traj.descent.states:E', descent.interp('E', [0, 0]))
    prob.set_val('traj.descent.controls:V', 15.0)

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
    E = _stacked_timeseries(source, 'E')
    h = _stacked_timeseries(source, 'h')
    gg = _stacked_timeseries(source, 'gg')
    V_flown = _stacked_timeseries(source, 'V')

    fig, axes = plt.subplots(1, 3, sharex=True, figsize=(9, 7))

    axes[0].plot(t, h, 'b-', linewidth=2)
    axes[0].set_ylabel('Altitude h [m]')
    axes[0].set_title('Altitude and glide angle vs elapsed time')
    axes[0].grid(True)

    axes[1].plot(t, np.degrees(gg), 'r-', linewidth=2)
    axes[1].set_xlabel('Elapsed time [s]')
    axes[1].set_ylabel('Glide angle gg [deg]')
    axes[1].grid(True)
    axes[1].ticklabel_format(useOffset=False, style='plain')

    axes[2].plot(t, E, 'g-', linewidth=2)
    axes[2].set_xlabel('Elapsed time [s]')
    axes[2].set_ylabel('Net energy E [J/m^2]')
    axes[2].grid(True)

    plt.tight_layout()

    # Stall speed vs altitude, compared to the flown airspeed
    rho_h = Atmosphere(h).density
    V_stall_h = np.sqrt(2*M_Sw*g / (rho_h * CLmax))

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(h, V_stall_h, 'k--', label='V_stall')
    ax2.plot(h, 1.2 * V_stall_h, 'r--', label='1.2 x V_stall (margin threshold)')
    ax2.plot(h, V_flown, 'b-', linewidth=2, label='Flown V')
    ax2.set_xlabel('Altitude h [m]')
    ax2.set_ylabel('Speed [m/s]')
    ax2.set_title('Stall speed vs altitude')
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout()

    plt.show()


RUN_OPTIMIZATION = True  # False -> skip the solve and re-plot the last saved solution instead


if __name__ == '__main__':
    if RUN_OPTIMIZATION:
        prob = build_problem()
        dm.run_problem(prob, run_driver=True, make_plots=True)
        source = prob
    else:
        source = om.CaseReader(SOLUTION_DB).get_case('final')

    plot_results(source)
