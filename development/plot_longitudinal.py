import json

import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt

from solving_longitudinal import paths_file


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
    t_split1 = sol_case.get_val('traj.climb.timeseries.time')[-1, 0]
    t_split2 = sol_case.get_val('traj.cruise.timeseries.time')[-1, 0]

    fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    fig.suptitle('Climb + Cruise + Descent (maximize cruise duration)')

    axs[0].plot(t_sol, h_sol, 'o', ms=4, label='solution')
    axs[0].plot(t_sim, h_sim, '-', label='simulation')
    axs[0].axvline(t_split1, color='gray', ls='--', lw=1, label='phase boundary')
    axs[0].axvline(t_split2, color='gray', ls='--', lw=1)
    axs[0].set_ylabel('h (m)')
    axs[0].legend()

    axs[1].plot(t_sol, V_sol, 'o', ms=4, label='solution')
    axs[1].plot(t_sim, V_sim, '-', label='simulation')
    axs[1].axvline(t_split1, color='gray', ls='--', lw=1)
    axs[1].axvline(t_split2, color='gray', ls='--', lw=1)
    axs[1].set_xlabel('time (s)')
    axs[1].set_ylabel('V (m/s)')

    axs[2].plot(t_sol, gg_sol, 'o', ms=4, label='solution')
    axs[2].plot(t_sim, gg_sim, '-', label='simulation')
    axs[2].axvline(t_split1, color='gray', ls='--', lw=1)
    axs[2].axvline(t_split2, color='gray', ls='--', lw=1)
    axs[2].set_xlabel('time (s)')
    axs[2].set_ylabel('gg (rad)')

    plt.show()


if __name__ == '__main__':
    main()
