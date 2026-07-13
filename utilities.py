import numpy as np
import matplotlib.pyplot as plt
from solarpy import irradiance_on_plane
from datetime import datetime, timedelta



def daily_irradiance(h, lat, start_date, vnorm = np.array([0, 0, -1]), step=timedelta(minutes=15)):
    dates = []
    current = start_date
 
    end_date = start_date + timedelta(days=1)

    while current <= end_date:
        dates.append(current)
        current += step

    t = np.array([(d - start_date).total_seconds() for d in dates], dtype=float)
    G = np.array([irradiance_on_plane(vnorm, h, d, lat) for d in dates], dtype=float)

    # J/m²
    energy_j_per_m2 = np.trapezoid(G, t)

    # Wh/m²
    energy_wh_per_m2 = energy_j_per_m2 / 3600.0
    return energy_wh_per_m2





def yearly_mean_power_contour(h, latitudes, days, solar_cell_efficiency = 0.15, vnorm = np.array([0, 0, -1]), step=timedelta(minutes=15)):

    mean_power_distribution = np.zeros((len(latitudes), len(days)))
    start_day = days[0]
    total_days = (days[-1] - start_day).days + 1
    day_numbers = np.array([
        (current_day - start_day).days + 1
        for current_day in days
    ], dtype=int)

    for lat_idx, lat in enumerate(latitudes):
        for day_idx, current_day in enumerate(days):
            current_lat = lat
            day_number = day_numbers[day_idx]

            daily_energy = daily_irradiance(
                h,
                current_lat,
                current_day,
                step=step,
            )

            mean_power_distribution[lat_idx, day_idx] = daily_energy / 24.0 * solar_cell_efficiency

            print(f'Processed latitude {current_lat:.2f} deg, day {day_number}/{total_days}: mean power = {mean_power_distribution[lat_idx, day_idx]:.2f} W/m²')

    fig, ax = plt.subplots(figsize=(14, 7))
    max_mean_power = float(np.max(mean_power_distribution))
    levels = np.arange(0, max_mean_power + 5, 5)
    contour = ax.contourf(day_numbers, latitudes, mean_power_distribution, levels=levels, cmap='cividis')

    ax.set_xlabel('Day of year')
    ax.set_ylabel('Latitude (deg)')
    ax.set_title('Mean power distribution')
    ax.set_xlim(1, total_days)

    fig.colorbar(contour, ax=ax, ticks=levels, label='Mean power (W/m²)')
    plt.tight_layout()
    plt.savefig('mean_power_distribution.png', dpi=200)
    plt.close(fig)

    print('Contour image saved to mean_power_distribution.png')
    print('Mean power distribution shape:', mean_power_distribution.shape)
