import json

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


def stitch(varname, case):
    """Concatenate a timeseries variable across the climb, cruise and descent phases."""
    return np.concatenate([
        case.get_val(f'traj.{phase}.timeseries.{varname}')[:, 0]
        for phase in ('climb', 'cruise', 'descent')
    ])


def main():
    with open(paths_file) as f:
        paths = json.load(f)

    sol_case = om.CaseReader(paths['solution']).get_case('final')
    sim_case = om.CaseReader(paths['simulation']).get_case('final')

    t_sol, h_sol, V_sol, gg_sol = stitch('time', sol_case), stitch('h', sol_case), stitch('V', sol_case), stitch('gg', sol_case)
    t_sim, h_sim, V_sim, gg_sim = stitch('time', sim_case), stitch('h', sim_case), stitch('V', sim_case), stitch('gg', sim_case)
    J_PER_KWH = 3.6e6
    DV_sw_int_sol = stitch('DV_sw_int', sol_case) / J_PER_KWH
    DV_sw_int_sim = stitch('DV_sw_int', sim_case) / J_PER_KWH
    Psol_sw_int_sol = stitch('Psol_sw_int', sol_case) / J_PER_KWH
    Psol_sw_int_sim = stitch('Psol_sw_int', sim_case) / J_PER_KWH
    Epot_sw_int_sol = stitch('Epot_sw_int', sol_case) / J_PER_KWH
    Epot_sw_int_sim = stitch('Epot_sw_int', sim_case) / J_PER_KWH
    Net_sw_int_sol = stitch('Net_sw_int', sol_case) / J_PER_KWH
    Net_sw_int_sim = stitch('Net_sw_int', sim_case) / J_PER_KWH
    t_split1 = sol_case.get_val('traj.climb.timeseries.time')[-1, 0]
    t_split2 = sol_case.get_val('traj.cruise.timeseries.time')[-1, 0]

    # Figure 1: Trajectory - altitude, speed and glide angle vs elapsed time (left
    # column, sharing the time x-axis), stall speed vs altitude alongside (right
    # column, spanning all 3 rows - it has its own x-axis, altitude rather than time)
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle('Climb + Cruise + Descent (maximize cruise duration)')
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1])

    ax_alt = fig.add_subplot(gs[0, 0])
    ax_speed = fig.add_subplot(gs[1, 0], sharex=ax_alt)
    ax_gg = fig.add_subplot(gs[2, 0], sharex=ax_alt)
    ax_stall = fig.add_subplot(gs[:, 1])

    ax_alt.plot(t_sol, h_sol, 'o', ms=4, label='solution')
    ax_alt.plot(t_sim, h_sim, '-', label='simulation')
    ax_alt.axvline(t_split1, color='gray', ls='--', lw=1, label='phase boundary')
    ax_alt.axvline(t_split2, color='gray', ls='--', lw=1)
    ax_alt.set_ylabel('h (m)')
    ax_alt.legend()

    ax_speed.plot(t_sol, V_sol, 'o', ms=4, label='solution')
    ax_speed.plot(t_sim, V_sim, '-', label='simulation')
    ax_speed.axvline(t_split1, color='gray', ls='--', lw=1)
    ax_speed.axvline(t_split2, color='gray', ls='--', lw=1)
    ax_speed.set_ylabel('V (m/s)')

    ax_gg.plot(t_sol, gg_sol, 'o', ms=4, label='solution')
    ax_gg.plot(t_sim, gg_sim, '-', label='simulation')
    ax_gg.axvline(t_split1, color='gray', ls='--', lw=1)
    ax_gg.axvline(t_split2, color='gray', ls='--', lw=1)
    ax_gg.set_xlabel('time (s)')
    ax_gg.set_ylabel('gg (rad)')

    # Stall speed vs altitude, compared to the flown airspeed
    rho_sim = Atmosphere(h_sim).density
    V_stall_sim = np.sqrt(2 * M_Sw * g / (rho_sim * CLmax))

    ax_stall.plot(h_sim, V_stall_sim, 'k--', label='V_stall')
    ax_stall.plot(h_sim, 1.2 * V_stall_sim, 'r--', label='1.2 x V_stall (margin threshold)')
    ax_stall.plot(h_sol, V_sol, 'o', ms=4, color='tab:blue', label='Flown V (solution)')
    ax_stall.plot(h_sim, V_sim, '-', color='tab:blue', label='Flown V (simulation)')
    ax_stall.set_xlabel('Altitude h (m)')
    ax_stall.set_ylabel('Speed (m/s)')
    ax_stall.set_title('Stall speed vs altitude')
    ax_stall.grid(True)
    ax_stall.legend()

    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    fig2.suptitle('Drag dissipation vs. solar energy stored vs. potential energy vs. net')
    ax2.plot(t_sol, DV_sw_int_sol, 'o', ms=4, color='tab:red', label='Drag dissipated (solution)')
    ax2.plot(t_sim, DV_sw_int_sim, '-', color='tab:red', label='Drag dissipated (simulation)')
    ax2.plot(t_sol, Psol_sw_int_sol, 'o', ms=4, color='tab:orange', label='Solar energy stored (solution)')
    ax2.plot(t_sim, Psol_sw_int_sim, '-', color='tab:orange', label='Solar energy stored (simulation)')
    ax2.plot(t_sol, Epot_sw_int_sol, 'o', ms=4, color='tab:green', label='Potential energy (solution)')
    ax2.plot(t_sim, Epot_sw_int_sim, '-', color='tab:green', label='Potential energy (simulation)')
    ax2.plot(t_sol, Net_sw_int_sol, 'o', ms=4, color='tab:blue', label='Net energy (solution)')
    ax2.plot(t_sim, Net_sw_int_sim, '-', color='tab:blue', label='Net energy (simulation)')
    ax2.axvline(t_split1, color='gray', ls='--', lw=1, label='phase boundary')
    ax2.axvline(t_split2, color='gray', ls='--', lw=1)
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('Energy (kWh/m^2)')
    ax2.legend()

    plt.show()


if __name__ == '__main__':
    main()
