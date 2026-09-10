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
plots_out_dir = os.path.join(_this_dir, '..', 'nice_plots_3d_trajectory', 'Trajectory Optimization')

# Shared phase color code - same mapping used across all figures
colors = {'climb': 'tab:blue', 'cruise': 'tab:green', 'descent': 'tab:red'}


def stitch(varname, case, phases=('climb', 'cruise', 'descent')):
    """Concatenate a timeseries variable across the given phases, in order."""
    return np.concatenate([
        case.get_val(f'traj.{phase}.timeseries.{varname}')[:, 0]
        for phase in phases
    ])


def load_case(paths, key):
    return om.CaseReader(paths[key]).get_case('final')


def main():
    with open(paths_file) as f:
        long_paths = json.load(f)
    with open(lateral_paths_file) as f:
        lat_paths = json.load(f)

    long_sol = load_case(long_paths, 'solution')
    long_sim = load_case(long_paths, 'simulation')
    lat_sol = load_case(lat_paths, 'solution')
    lat_sim = load_case(lat_paths, 'simulation')

    # --- Longitudinal: h(t), V(t), gamma(t) ---
    t_sol, h_sol, V_sol, gg_sol = (stitch(v, long_sol) for v in ('time', 'h', 'V', 'gg'))
    t_sim, h_sim, V_sim, gg_sim = (stitch(v, long_sim) for v in ('time', 'h', 'V', 'gg'))
    t_split1 = long_sol.get_val('traj.climb.timeseries.time')[-1, 0]
    t_split2 = long_sol.get_val('traj.cruise.timeseries.time')[-1, 0]

    gg_sol = np.rad2deg(gg_sol)
    gg_sim = np.rad2deg(gg_sim)

    # Which longitudinal phase each longitudinal solution point belongs to
    phase_of_long_t = np.where(t_sol < t_split1, 'climb',
                       np.where(t_sol < t_split2, 'cruise', 'descent'))

    fig1, axs1 = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    fig1.suptitle('Longitudinal profile')

    for name, color in colors.items():
        mask = phase_of_long_t == name
        if np.any(mask):
            axs1[0].plot(t_sol[mask], h_sol[mask], '-o', ms=4, color=color, label=name)
            axs1[1].plot(t_sol[mask], V_sol[mask], '-o', ms=4, color=color)
            axs1[2].plot(t_sol[mask], gg_sol[mask], '-o', ms=4, color=color)

    axs1[0].plot(t_sim, h_sim, '-', color='gray', lw=2, alpha=0.6, label='simulation')
    axs1[1].plot(t_sim, V_sim, '-', color='gray', lw=2, alpha=0.6)
    axs1[2].plot(t_sim, gg_sim, '-', color='gray', lw=2, alpha=0.6)

    axs1[0].axvline(t_split1, color='gray', ls='--', lw=2)
    axs1[0].axvline(t_split2, color='gray', ls='--', lw=2)
    axs1[0].set_ylabel('h (m)')
    axs1[0].legend()

    axs1[1].axvline(t_split1, color='gray', ls='--', lw=2)
    axs1[1].axvline(t_split2, color='gray', ls='--', lw=2)
    axs1[1].set_ylabel('V (m/s)')

    axs1[2].axvline(t_split1, color='gray', ls='--', lw=2)
    axs1[2].axvline(t_split2, color='gray', ls='--', lw=2)
    axs1[2].set_xlabel('time (s)')
    axs1[2].set_ylabel('gamma (deg)')

    # --- Lateral: y(x), psi(t), phi(t) ---
    # The lateral maneuver is now a sequence of linked 1-hour phases (seg0, seg1, ...)
    # rather than a single 'phase0', so its variables need stitching too.
    seg_names = lat_paths['seg_names']
    x_sol, y_sol, t_lat_sol, psi_sol, phi_sol = (
        stitch(v, lat_sol, seg_names) for v in ('x', 'y', 'time', 'psi', 'phi'))
    x_sim, y_sim, t_lat_sim, psi_sim, phi_sim = (
        stitch(v, lat_sim, seg_names) for v in ('x', 'y', 'time', 'psi', 'phi'))

    psi_sol = np.rad2deg(psi_sol)
    psi_sim = np.rad2deg(psi_sim)
    phi_sol = np.rad2deg(phi_sol)
    phi_sim = np.rad2deg(phi_sim)

    # Which longitudinal phase each lateral solution point falls into, based on
    # absolute time - same color code as the longitudinal plots and the 3D plot.
    phase_of_lat_t = np.where(t_lat_sol < t_split1, 'climb',
                     np.where(t_lat_sol < t_split2, 'cruise', 'descent'))

    fig2, axs2 = plt.subplots(3, 1, figsize=(6, 8))
    fig2.suptitle('Lateral trajectory')

    for name, color in colors.items():
        mask = phase_of_lat_t == name
        if np.any(mask):
            axs2[0].plot(x_sol[mask], y_sol[mask], '-o', ms=4, color=color, label=name)
            axs2[1].plot(t_lat_sol[mask], psi_sol[mask], '-o', ms=4, color=color)
            axs2[2].plot(t_lat_sol[mask], phi_sol[mask], '-o', ms=4, color=color)

    axs2[0].plot(x_sim, y_sim, '-', color='gray', lw=2, alpha=0.6, label='simulation')
    axs2[1].plot(t_lat_sim, psi_sim, '-', color='gray', lw=2, alpha=0.6)
    axs2[2].plot(t_lat_sim, phi_sim, '-', color='gray', lw=2, alpha=0.6)

    axs2[0].set_xlabel('x (m)')
    axs2[0].set_ylabel('y (m)')
    axs2[0].set_aspect('equal')
    axs2[0].legend()

    axs2[1].set_ylabel('psi (deg)')

    axs2[2].set_xlabel('time (s)')
    axs2[2].set_ylabel('phi (deg)')

    # --- 3D (x, y, h): lateral ground track combined with the longitudinal altitude
    # profile at those same times, colored by which longitudinal phase each point
    # falls into. Solution points are connected by a line (in time order); the
    # simulation is overlaid as a thin continuous line for comparison. ---
    h_at_lat_sol_t = np.interp(t_lat_sol, t_sol, h_sol)
    h_at_lat_sim_t = np.interp(t_lat_sim, t_sim, h_sim)

    fig3 = plt.figure(figsize=(9, 7))
    ax3 = fig3.add_subplot(111, projection='3d')

    for name, color in colors.items():
        mask = phase_of_lat_t == name
        if np.any(mask):
            ax3.plot(x_sol[mask], y_sol[mask], h_at_lat_sol_t[mask], '-o', ms=4, color=color, label=name)

    ax3.plot(x_sim, y_sim, h_at_lat_sim_t, '-', color='gray', lw=2, alpha=0.6, label='simulation')

    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    ax3.set_zlabel('h (m)')
    ax3.set_title('3D trajectory')
    ax3.legend()

    os.makedirs(plots_out_dir, exist_ok=True)
    fig1.savefig(os.path.join(plots_out_dir, 'longitudinal_profile.png'), dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(plots_out_dir, 'lateral_trajectory.png'), dpi=150, bbox_inches='tight')
    fig3.savefig(os.path.join(plots_out_dir, 'trajectory_3d.png'), dpi=150, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    main()
