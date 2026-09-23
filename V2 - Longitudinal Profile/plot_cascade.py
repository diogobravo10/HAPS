import json
import os

import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt


paths_file = 'solving_longitudinal_paths.json'
lateral_paths_file = 'solving_lateral_paths.json'

# Anchored to this file's own location (not the current working directory),
# since the working directory a script runs with depends on how it's launched
# (e.g. many IDE "Run" buttons use the workspace root, not the file's folder)
# and a plain relative path can silently resolve outside the repo.
_this_dir = os.path.dirname(os.path.abspath(__file__))
plots_out_dir = os.path.join(_this_dir, '..', 'nice_plots_3d_trajectory', 'Trajectory Optimization', 'Cylinder')

# Shared phase color code - same mapping used across all figures
colors = {'climb': 'tab:blue', 'cruise': 'tab:green', 'descent': 'tab:red'}

J_PER_KWH = 3.6e6


def stitch(varname, case, phases=('climb', 'cruise', 'descent')):
    """Concatenate a timeseries variable across the given phases, in order."""
    return np.concatenate([
        case.get_val(f'traj.{phase}.timeseries.{varname}')[:, 0]
        for phase in phases
    ])


def load_case(paths, key):
    return om.CaseReader(paths[key]).get_case('final')


def plot_by_phase(ax, t, y, phase_of_t, legend=False):
    """Plot y(t) as bare points (no connecting line - collocation nodes aren't a
    physically continuous trace, unlike a simulated trajectory), colored by which
    longitudinal phase each point falls into."""
    for name, color in colors.items():
        mask = phase_of_t == name
        if np.any(mask):
            ax.plot(t[mask], y[mask], 'o', ms=4, color=color, label=name if legend else None)


def main():
    with open(paths_file) as f:
        long_paths = json.load(f)
    with open(lateral_paths_file) as f:
        lat_paths = json.load(f)

    long_sol = load_case(long_paths, 'solution')
    lat_sol = load_case(lat_paths, 'solution')

    # --- Longitudinal solution data ---
    # Solution points only - neither solving_longitudinal.py nor solving_cascade_V3.py's
    # solve_lateral() runs traj.simulate() anymore, so there's no continuous curve to
    # overlay, and the points themselves are plotted unconnected below: adjacent
    # collocation nodes aren't a physically continuous trace the way a simulated
    # trajectory would be.
    t_sol, h_sol, V_sol = (stitch(v, long_sol) for v in ('time', 'h', 'V'))
    gg_sol, aa_sol, Tp_sol = (stitch(v, long_sol) for v in ('gg', 'aa', 'Tp'))
    Vdot_sol, ggdot_sol = (stitch(v, long_sol) for v in ('V_dot', 'gg_dot'))
    gg_sol = np.rad2deg(gg_sol)
    aa_sol = np.rad2deg(aa_sol)
    ggdot_sol = np.rad2deg(ggdot_sol)

    DV_sw_int_sol = stitch('DV_sw_int', long_sol) / J_PER_KWH
    TV_sw_int_sol = stitch('TV_sw_int', long_sol) / J_PER_KWH
    Psol_sw_int_sol = stitch('Psol_sw_int', long_sol) / J_PER_KWH
    Epot_sw_int_sol = stitch('Epot_sw_int', long_sol) / J_PER_KWH
    Net_sw_int_sol = stitch('Net_sw_int', long_sol) / J_PER_KWH
    SOC_sol = 100 * stitch('SOC', long_sol)

    t_split1 = long_sol.get_val('traj.climb.timeseries.time')[-1, 0]
    t_split2 = long_sol.get_val('traj.cruise.timeseries.time')[-1, 0]

    # Which longitudinal phase each longitudinal solution point belongs to
    phase_of_long_t = np.where(t_sol < t_split1, 'climb',
                       np.where(t_sol < t_split2, 'cruise', 'descent'))

    # --- Figure 1: altitude and airspeed vs time ---
    fig1, (ax_h, ax_V) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    fig1.suptitle('Altitude and airspeed')

    plot_by_phase(ax_h, t_sol, h_sol, phase_of_long_t, legend=True)
    ax_h.axvline(t_split1, color='gray', ls='--', lw=1)
    ax_h.axvline(t_split2, color='gray', ls='--', lw=1)
    ax_h.set_ylabel('h (m)')
    ax_h.legend()

    plot_by_phase(ax_V, t_sol, V_sol, phase_of_long_t)
    ax_V.axvline(t_split1, color='gray', ls='--', lw=1)
    ax_V.axvline(t_split2, color='gray', ls='--', lw=1)
    ax_V.set_xlabel('time (s)')
    ax_V.set_ylabel('V (m/s)')

    plt.tight_layout()

    # --- Figure 2: glide angle, angle of attack and throttle vs time ---
    fig2, (ax_gg, ax_aa, ax_Tp) = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
    fig2.suptitle('Glide angle, angle of attack and throttle')

    plot_by_phase(ax_gg, t_sol, gg_sol, phase_of_long_t, legend=True)
    ax_gg.axvline(t_split1, color='gray', ls='--', lw=1)
    ax_gg.axvline(t_split2, color='gray', ls='--', lw=1)
    ax_gg.set_ylabel('gg (deg)')
    ax_gg.legend()

    plot_by_phase(ax_aa, t_sol, aa_sol, phase_of_long_t)
    ax_aa.axvline(t_split1, color='gray', ls='--', lw=1)
    ax_aa.axvline(t_split2, color='gray', ls='--', lw=1)
    ax_aa.set_ylabel('aa (deg)')

    plot_by_phase(ax_Tp, t_sol, Tp_sol, phase_of_long_t)
    ax_Tp.axvline(t_split1, color='gray', ls='--', lw=1)
    ax_Tp.axvline(t_split2, color='gray', ls='--', lw=1)
    ax_Tp.set_xlabel('time (s)')
    ax_Tp.set_ylabel('Tp')

    plt.tight_layout()

    # --- Figure 3: drag dissipation vs. propulsion use vs. solar energy stored vs.
    # potential energy vs. net, plus battery state of charge. Colored by quantity
    # here (not by phase) since several quantities share the same axes. ---
    fig3, (ax_e, ax_soc) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                         gridspec_kw={'height_ratios': [2, 1]})
    fig3.suptitle('Drag dissipation vs. propulsion use vs. solar energy stored vs. potential energy vs. net')

    ax_e.plot(t_sol, DV_sw_int_sol, 'o', ms=4, color='tab:red', label='Drag dissipated')
    ax_e.plot(t_sol, TV_sw_int_sol, 'o', ms=4, color='tab:purple', label='Propulsion energy used')
    ax_e.plot(t_sol, Psol_sw_int_sol, 'o', ms=4, color='tab:orange', label='Solar energy stored')
    ax_e.plot(t_sol, Epot_sw_int_sol, 'o', ms=4, color='tab:green', label='Potential energy')
    ax_e.plot(t_sol, Net_sw_int_sol, 'o', ms=4, color='tab:blue', label='Net energy')
    ax_e.axvline(t_split1, color='gray', ls='--', lw=1, label='phase boundary')
    ax_e.axvline(t_split2, color='gray', ls='--', lw=1)
    ax_e.set_ylabel('Energy (kWh/m^2)')
    ax_e.legend()

    ax_soc.plot(t_sol, SOC_sol, 'o', ms=4, color='tab:blue')
    ax_soc.axvline(t_split1, color='gray', ls='--', lw=1)
    ax_soc.axvline(t_split2, color='gray', ls='--', lw=1)
    ax_soc.set_xlabel('time (s)')
    ax_soc.set_ylabel('State of charge (%)')

    plt.tight_layout()

    # --- Figure 4: state rates (V_dot, gg_dot) vs time ---
    fig4, (ax_Vdot, ax_ggdot) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    fig4.suptitle('V_dot and gg_dot vs. time')

    plot_by_phase(ax_Vdot, t_sol, Vdot_sol, phase_of_long_t, legend=True)
    ax_Vdot.axvline(t_split1, color='gray', ls='--', lw=1)
    ax_Vdot.axvline(t_split2, color='gray', ls='--', lw=1)
    ax_Vdot.set_ylabel('V_dot (m/s^2)')
    ax_Vdot.legend()

    plot_by_phase(ax_ggdot, t_sol, ggdot_sol, phase_of_long_t)
    ax_ggdot.axvline(t_split1, color='gray', ls='--', lw=1)
    ax_ggdot.axvline(t_split2, color='gray', ls='--', lw=1)
    ax_ggdot.set_xlabel('time (s)')
    ax_ggdot.set_ylabel('gg_dot (deg/s)')

    plt.tight_layout()

    # --- Lateral: y(x), tt(t) ---
    # The lateral maneuver is a single phase ('phase0' for the cylinder loiter model,
    # or seg0, seg1, ... for the older multi-segment layout), so seg_names still
    # drives the stitching either way.
    seg_names = lat_paths['seg_names']
    x_sol, y_sol, t_lat_sol, tt_sol = (
        stitch(v, lat_sol, seg_names) for v in ('x', 'y', 'time', 'tt'))

    tt_sol = np.rad2deg(tt_sol)

    # Which longitudinal phase each lateral solution point falls into, based on
    # absolute time - same color code as the longitudinal plots and the 3D plot.
    phase_of_lat_t = np.where(t_lat_sol < t_split1, 'climb',
                     np.where(t_lat_sol < t_split2, 'cruise', 'descent'))

    fig5, axs5 = plt.subplots(2, 1, figsize=(6, 6))
    fig5.suptitle('Lateral trajectory')

    plot_by_phase(axs5[0], x_sol, y_sol, phase_of_lat_t, legend=True)
    plot_by_phase(axs5[1], t_lat_sol, tt_sol, phase_of_lat_t)

    axs5[0].set_xlabel('x (m)')
    axs5[0].set_ylabel('y (m)')
    axs5[0].set_aspect('equal')
    axs5[0].legend()

    axs5[1].set_xlabel('time (s)')
    axs5[1].set_ylabel('tt (deg)')

    # --- 3D (x, y, h): lateral ground track combined with the longitudinal altitude
    # profile at those same times, colored by which longitudinal phase each point
    # falls into. Solution points only, unconnected. ---
    h_at_lat_sol_t = np.interp(t_lat_sol, t_sol, h_sol)

    fig6 = plt.figure(figsize=(9, 7))
    ax6 = fig6.add_subplot(111, projection='3d')

    for name, color in colors.items():
        mask = phase_of_lat_t == name
        if np.any(mask):
            ax6.plot(x_sol[mask], y_sol[mask], h_at_lat_sol_t[mask], 'o', ms=4, color=color, label=name)

    ax6.set_xlabel('x (m)')
    ax6.set_ylabel('y (m)')
    ax6.set_zlabel('h (m)')
    ax6.set_title('3D trajectory')
    ax6.legend()

    os.makedirs(plots_out_dir, exist_ok=True)
    fig1.savefig(os.path.join(plots_out_dir, 'longitudinal_altitude_speed.png'), dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(plots_out_dir, 'longitudinal_gg_aa_Tp.png'), dpi=150, bbox_inches='tight')
    fig3.savefig(os.path.join(plots_out_dir, 'longitudinal_energy_balance.png'), dpi=150, bbox_inches='tight')
    fig4.savefig(os.path.join(plots_out_dir, 'longitudinal_Vdot_ggdot.png'), dpi=150, bbox_inches='tight')
    fig5.savefig(os.path.join(plots_out_dir, 'lateral_trajectory.png'), dpi=150, bbox_inches='tight')
    fig6.savefig(os.path.join(plots_out_dir, 'trajectory_3d.png'), dpi=150, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    main()
