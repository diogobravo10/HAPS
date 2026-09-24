import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime, timedelta
import solving_longitudinal


def stitch(varname, prob):
    """Concatenate a timeseries variable across the climb, cruise and descent phases."""
    return np.concatenate([
        prob.get_val(f'traj.{phase}.timeseries.{varname}')[:, 0]
        for phase in ('climb', 'cruise', 'descent')
    ])


start_date = datetime(2027, 1, 1, 6, 0)
end_date = datetime(2028, 1, 1, 6, 0)
dday = 20
N_lat = 60
S_lat = -60
dlat = 15

def filtering_yearly_mean_power_contour():
    """Like `yearly_mean_power_contour` but filter cells where available
    power (from batteries or propulsion limit) is insufficient.

    Parameters are pulled from `optimum_mass_properties`,
    `global_time_and_location` and `user_params`. Produces and shows
    a contour plot (PNG file) of feasible mean power values.
    """
  
    
    
    days_array = np.array([
        start_date + timedelta(days=i)
        for i in range(0, (end_date - start_date).days + dday, dday)
    ], dtype=object)

    start_day = days_array[0]
    total_days = (days_array[-1] - start_day).days + 1
    day_numbers = np.array([
        (current_day - start_day).days + 1
        for current_day in days_array
    ], dtype=int)

    latitudes_array = np.arange(S_lat, N_lat + dlat, dlat, dtype=float)


    SOC_distribution = np.zeros((len(latitudes_array), len(days_array)))


    for lat_idx, lat in enumerate(latitudes_array):
        for day_idx, current_day in enumerate(days_array):
            current_lat = lat
            day_number = day_numbers[day_idx]

            prob, exp_out, success = solving_longitudinal.main(M_sw=3.7, mbat_sw=2.0, start_date=current_day, lat=current_lat, soc_initial=0.2)

            if not success:
                SOC_distribution[lat_idx, day_idx] = -1
                print(f'Optimization failed at latitude {current_lat:.2f} deg, day {day_number}/{total_days}: SOC = -1')
                continue

            SOC_array = stitch('SOC', prob)
            SOC = 1 - (max(SOC_array) - SOC_array[-1]) if max(SOC_array) > 1 else SOC_array[-1]
            SOC_distribution[lat_idx, day_idx] = SOC if SOC > 0.2 else 0

            print(f'Processed latitude {current_lat:.2f} deg, day {day_number}/{total_days}: mean power = {SOC_distribution[lat_idx, day_idx]:.2f} W/m^2')


    fig, ax = plt.subplots(figsize=(14, 7))
    max_SOC = float(np.max(SOC_distribution))
    levels = np.arange(0, max_SOC + 0.01, 0.1)
    if levels.size < 2:
        levels = np.array([0.0, 1.0])
    contour = ax.contourf(day_numbers, latitudes_array, SOC_distribution, levels=levels, cmap='cividis')

    ax.set_xlabel('Day of year')
    ax.set_ylabel('Latitude (deg)')
    ax.set_title(f'SOC distribution')
    ax.set_xlim(1, total_days)

    square_x = [1, 1, 365, 365, 1]
    square_y = [33, 43, 43, 33, 33]
    line_azores, = ax.plot(square_x, square_y, color='red', linewidth=2, label='Azores EEZ')    
    
    patch_azores = Patch(
        facecolor='none',
        edgecolor=line_azores.get_color(),
        linewidth=line_azores.get_linewidth(),
        label=line_azores.get_label()
    )

    square_x = [1, 1, 365, 365, 1]
    square_y = [-28, -10, -10, -28, -28]
    line_moz, = ax.plot(square_x, square_y, color='orange', linewidth=2, label='Mozambique EEZ')
    
    patch_moz = Patch(
        facecolor='none',
        edgecolor=line_moz.get_color(),
        linewidth=line_moz.get_linewidth(),
        label=line_moz.get_label()
    )

    ax.legend(handles=[patch_azores, patch_moz], loc='upper right')

    # ax.scatter(355, 43, marker='x', color='red', s=150, linewidths=2)

    fig.colorbar(contour, ax=ax, ticks=levels, label=r'SOC')
    plt.tight_layout()
    filename = f'fig_mean_power_distribution.png'
    plt.savefig(filename, dpi=200)
    plt.pause(0.1)
    plt.show(block=False)


    print(f'Contour image saved to {filename}')

filtering_yearly_mean_power_contour()