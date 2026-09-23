import json
import os

import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt
from ambiance import Atmosphere

from solving_longitudinal import paths_file

# Diagnostic-only constants for the stall-speed-vs-altitude overlay below -
# development's aero_module.py has no lift/weight balance (DV_sw is a drag-power
# term only, with no CL/wing-loading requirement), so these mirror the wing
# loading/CLmax convention mission_profile.py uses for the same comparison,
# purely to draw this reference curve; they don't feed back into the model.
M_Sw = 3.0  # [kg/m^2] wing loading
g = 9.81
CLmax = 1.2

# Anchored to this file's own location (not the current working directory), same
# convention as plot_cascade.py's plots_out_dir.
_this_dir = os.path.dirname(os.path.abspath(__file__))
plots_out_dir = os.path.join(_this_dir, '..', 'nice_plots_3d_trajectory',
                              'Trajectory Optimization', 'Cylinder', 'UP2DATE')


def stitch(varname, case):
    """Concatenate a timeseries variable across the climb, cruise and descent phases.
    case=None (no simulation was run) passes through as None."""
    if case is None:
        return None
    return np.concatenate([
        case.get_val(f'traj.{phase}.timeseries.{varname}')[:, 0]
        for phase in ('climb', 'cruise', 'descent')
    ])


def plot_solsim(ax, t_sol, y_sol, t_sim, y_sim, color=None, label=None):
    """Plot the solution as points and, only if a simulation was run (t_sim/y_sim not
    None), overlay it as a line - the common sol/sim pair used throughout these plots.
    label, if given, prefixes the legend entries ('label (solution)'/'label (simulation)');
    omit it for a plain 'solution'/'simulation' pair, or pass False to set no label at all."""
    kwargs = {'color': color} if color is not None else {}
    if label is False:
        sol_label = sim_label = None
    elif label is None:
        sol_label, sim_label = 'solution', 'simulation'
    else:
        sol_label, sim_label = f'{label} (solution)', f'{label} (simulation)'
    ax.plot(t_sol, y_sol, 'o', ms=4, label=sol_label, **kwargs)
    if t_sim is not None:
        ax.plot(t_sim, y_sim, '-', label=sim_label, **kwargs)


def main():
    with open(paths_file) as f:
        paths = json.load(f)

    sol_case = om.CaseReader(paths['solution']).get_case('final')
    # Optional: traj.simulate() may have been skipped when the solution was generated
    # (paths.json then has no 'simulation' entry) - plot the solution only in that case.
    have_sim = 'simulation' in paths and os.path.exists(paths['simulation'])
    sim_case = om.CaseReader(paths['simulation']).get_case('final') if have_sim else None

    t_sol, h_sol, V_sol, gg_sol = stitch('time', sol_case), stitch('h', sol_case), stitch('V', sol_case), stitch('gg', sol_case)
    t_sim, h_sim, V_sim, gg_sim = stitch('time', sim_case), stitch('h', sim_case), stitch('V', sim_case), stitch('gg', sim_case)
    Tp_sol, Tp_sim = stitch('Tp', sol_case), stitch('Tp', sim_case)
    aa_sol, aa_sim = stitch('aa', sol_case), stitch('aa', sim_case)
    Vdot_sol, Vdot_sim = stitch('V_dot', sol_case), stitch('V_dot', sim_case)
    ggdot_sol, ggdot_sim = stitch('gg_dot', sol_case), stitch('gg_dot', sim_case)
    gg_sol = np.degrees(gg_sol)
    gg_sim = np.degrees(gg_sim) if gg_sim is not None else None
    aa_sol = np.degrees(aa_sol)
    aa_sim = np.degrees(aa_sim) if aa_sim is not None else None
    ggdot_sol = np.degrees(ggdot_sol)
    ggdot_sim = np.degrees(ggdot_sim) if ggdot_sim is not None else None
    J_PER_KWH = 3.6e6
    DV_sw_int_sol = stitch('DV_sw_int', sol_case) / J_PER_KWH
    TV_sw_int_sol = stitch('TV_sw_int', sol_case) / J_PER_KWH
    Psol_sw_int_sol = stitch('Psol_sw_int', sol_case) / J_PER_KWH
    Epot_sw_int_sol = stitch('Epot_sw_int', sol_case) / J_PER_KWH
    Net_sw_int_sol = stitch('Net_sw_int', sol_case) / J_PER_KWH
    SOC_sol = 100 * stitch('SOC', sol_case)
    if have_sim:
        DV_sw_int_sim = stitch('DV_sw_int', sim_case) / J_PER_KWH
        TV_sw_int_sim = stitch('TV_sw_int', sim_case) / J_PER_KWH
        Psol_sw_int_sim = stitch('Psol_sw_int', sim_case) / J_PER_KWH
        Epot_sw_int_sim = stitch('Epot_sw_int', sim_case) / J_PER_KWH
        Net_sw_int_sim = stitch('Net_sw_int', sim_case) / J_PER_KWH
        SOC_sim = 100 * stitch('SOC', sim_case)
    else:
        DV_sw_int_sim = TV_sw_int_sim = Psol_sw_int_sim = Epot_sw_int_sim = Net_sw_int_sim = SOC_sim = None
    t_split1 = sol_case.get_val('traj.climb.timeseries.time')[-1, 0]
    t_split2 = sol_case.get_val('traj.cruise.timeseries.time')[-1, 0]

    # Figure 1: Trajectory - altitude and speed vs elapsed time (left column,
    # sharing the time x-axis), stall speed vs altitude alongside (right
    # column, spanning both rows - it has its own x-axis, altitude rather than time)
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle('Climb + Cruise + Descent (maximize cruise duration)')
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1])

    ax_alt = fig.add_subplot(gs[0, 0])
    ax_speed = fig.add_subplot(gs[1, 0], sharex=ax_alt)
    ax_stall = fig.add_subplot(gs[:, 1])

    plot_solsim(ax_alt, t_sol, h_sol, t_sim, h_sim)
    ax_alt.axvline(t_split1, color='gray', ls='--', lw=1, label='phase boundary')
    ax_alt.axvline(t_split2, color='gray', ls='--', lw=1)
    ax_alt.set_ylabel('h (m)')
    ax_alt.legend()

    plot_solsim(ax_speed, t_sol, V_sol, t_sim, V_sim, label=False)
    ax_speed.axvline(t_split1, color='gray', ls='--', lw=1)
    ax_speed.axvline(t_split2, color='gray', ls='--', lw=1)
    ax_speed.set_xlabel('time (s)')
    ax_speed.set_ylabel('V (m/s)')

    # Stall speed vs altitude, compared to the flown airspeed - uses the simulation's
    # (continuous) altitude/speed trace when available, else falls back to the solution's
    # (sparser, node-only) trace.
    h_ref = h_sim if have_sim else h_sol
    rho_ref = Atmosphere(h_ref).density
    V_stall_ref = np.sqrt(2 * M_Sw * g / (rho_ref * CLmax))

    ax_stall.plot(h_ref, V_stall_ref, 'k--', label='V_stall')
    ax_stall.plot(h_ref, 1.2 * V_stall_ref, 'r--', label='1.2 x V_stall (margin threshold)')
    plot_solsim(ax_stall, h_sol, V_sol, h_sim, V_sim, color='tab:blue', label='Flown V')
    ax_stall.set_xlabel('Altitude h (m)')
    ax_stall.set_ylabel('Speed (m/s)')
    ax_stall.set_title('Stall speed vs altitude')
    ax_stall.grid(True)
    ax_stall.legend()

    plt.tight_layout()

    fig2, (ax2, ax_soc) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                        gridspec_kw={'height_ratios': [2, 1]})
    fig2.suptitle('Drag dissipation vs. propulsion use vs. solar energy stored vs. potential energy vs. net')
    plot_solsim(ax2, t_sol, DV_sw_int_sol, t_sim, DV_sw_int_sim, color='tab:red', label='Drag dissipated')
    plot_solsim(ax2, t_sol, TV_sw_int_sol, t_sim, TV_sw_int_sim, color='tab:purple', label='Propulsion energy used')
    plot_solsim(ax2, t_sol, Psol_sw_int_sol, t_sim, Psol_sw_int_sim, color='tab:orange', label='Solar energy stored')
    plot_solsim(ax2, t_sol, Epot_sw_int_sol, t_sim, Epot_sw_int_sim, color='tab:green', label='Potential energy')
    plot_solsim(ax2, t_sol, Net_sw_int_sol, t_sim, Net_sw_int_sim, color='tab:blue', label='Net energy')
    ax2.axvline(t_split1, color='gray', ls='--', lw=1, label='phase boundary')
    ax2.axvline(t_split2, color='gray', ls='--', lw=1)
    ax2.set_ylabel('Energy (kWh/m^2)')
    ax2.legend()

    plot_solsim(ax_soc, t_sol, SOC_sol, t_sim, SOC_sim, color='tab:blue')
    ax_soc.axvline(t_split1, color='gray', ls='--', lw=1)
    ax_soc.axvline(t_split2, color='gray', ls='--', lw=1)
    ax_soc.set_xlabel('time (s)')
    ax_soc.set_ylabel('State of charge (%)')
    ax_soc.legend()

    plt.tight_layout()

    # Figure 3: Glide angle, angle of attack and Tp vs elapsed time
    fig3, (ax_gg, ax_aa, ax_Tp) = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    fig3.suptitle('Glide angle, angle of attack and Tp vs. time')

    plot_solsim(ax_gg, t_sol, gg_sol, t_sim, gg_sim)
    ax_gg.axvline(t_split1, color='gray', ls='--', lw=1, label='phase boundary')
    ax_gg.axvline(t_split2, color='gray', ls='--', lw=1)
    ax_gg.set_ylabel('gg (deg)')
    ax_gg.legend()

    plot_solsim(ax_aa, t_sol, aa_sol, t_sim, aa_sim, label=False)
    ax_aa.axvline(t_split1, color='gray', ls='--', lw=1)
    ax_aa.axvline(t_split2, color='gray', ls='--', lw=1)
    ax_aa.set_ylabel('aa (deg)')

    plot_solsim(ax_Tp, t_sol, Tp_sol, t_sim, Tp_sim, label=False)
    ax_Tp.axvline(t_split1, color='gray', ls='--', lw=1)
    ax_Tp.axvline(t_split2, color='gray', ls='--', lw=1)
    ax_Tp.set_xlabel('time (s)')
    ax_Tp.set_ylabel('Tp')

    plt.tight_layout()

    # Figure 4: State rates (V_dot, gg_dot) vs elapsed time
    fig4, (ax_Vdot, ax_ggdot) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    fig4.suptitle('V_dot and gg_dot vs. time')

    plot_solsim(ax_Vdot, t_sol, Vdot_sol, t_sim, Vdot_sim)
    ax_Vdot.axvline(t_split1, color='gray', ls='--', lw=1, label='phase boundary')
    ax_Vdot.axvline(t_split2, color='gray', ls='--', lw=1)
    ax_Vdot.set_ylim(-0.002, 0.002)
    ax_Vdot.set_ylabel('V_dot (m/s^2)')
    ax_Vdot.legend()

    plot_solsim(ax_ggdot, t_sol, ggdot_sol, t_sim, ggdot_sim, label=False)
    ax_ggdot.axvline(t_split1, color='gray', ls='--', lw=1)
    ax_ggdot.axvline(t_split2, color='gray', ls='--', lw=1)
    ax_ggdot.set_xlabel('time (s)')
    ax_ggdot.set_ylabel('gg_dot (deg/s)')

    plt.tight_layout()

    os.makedirs(plots_out_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_out_dir, 'longitudinal_trajectory.png'), dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(plots_out_dir, 'energy_balance.png'), dpi=150, bbox_inches='tight')
    fig3.savefig(os.path.join(plots_out_dir, 'gg_aa_Tp.png'), dpi=150, bbox_inches='tight')
    fig4.savefig(os.path.join(plots_out_dir, 'Vdot_ggdot.png'), dpi=150, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    main()
