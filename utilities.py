import numpy as np
import matplotlib.pyplot as plt
from solarpy import irradiance_on_plane
from datetime import datetime, timedelta

vnorm = np.array([0, 0, -1])  # plane pointing zenith


def daily_irradiance(h, lat, start_date, solar_cell_efficiency=0.15, vnorm = np.array([0, 0, -1]), step=timedelta(minutes=15)):
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
    
    power_density_available = energy_wh_per_m2 / 24.0 * solar_cell_efficiency
    
    return power_density_available





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
                solar_cell_efficiency=solar_cell_efficiency,
                step=step,
            )

            mean_power_distribution[lat_idx, day_idx] = daily_energy

            print(f'Processed latitude {current_lat:.2f} deg, day {day_number}/{total_days}: mean power = {mean_power_distribution[lat_idx, day_idx]:.2f} W/m²')

    fig, ax = plt.subplots(figsize=(14, 7))
    max_mean_power = float(np.max(mean_power_distribution))
    levels = np.arange(0, max_mean_power + 5, 5)
    contour = ax.contourf(day_numbers, latitudes, mean_power_distribution, levels=levels, cmap='cividis')

    ax.set_xlabel('Day of year')
    ax.set_ylabel('Latitude (deg)')
    ax.set_title('Mean power distribution')
    ax.set_xlim(1, total_days)

    square_x = [1, 1, 365, 365, 1]
    square_y = [33, 43, 43, 33, 33]
    ax.plot(square_x, square_y, color='red', linewidth=2, label='Azores EEZ')
    ax.scatter(355, 43, marker='x', color='red', s=150, linewidths=2)
    ax.legend(loc='upper right')

    fig.colorbar(contour, ax=ax, ticks=levels, label='Mean power (W/m²)')
    plt.tight_layout()
    plt.savefig('mean_power_distribution.png', dpi=200)
    plt.close(fig)

    print('Contour image saved to mean_power_distribution.png')
    print('Mean power distribution shape:', mean_power_distribution.shape)


