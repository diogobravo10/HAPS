import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from solarpy import irradiance_on_plane
from datetime import datetime, timedelta






def daily_irradiance(vnorm, h, lat, start_date, step=timedelta(minutes=15)):
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


def yearly_mean_power_contour(vnorm, h, latitudes, start_date, end_date, solar_cell_efficiency = 0.15, step=timedelta(minutes=15)):

    days = np.arange(1, (end_date - start_date).days + 1)
    mean_power_distribution = np.zeros((len(latitudes), len(days)))

    for lat_idx, lat in enumerate(latitudes):
        for day_idx in range(len(days)):
            current_day = start_date + timedelta(days=day_idx)

            daily_energy = daily_irradiance(
                vnorm,
                h,
                lat,
                current_day,
                step=timedelta(minutes=15),
            )

            mean_power_distribution[lat_idx, day_idx] = daily_energy / 24.0 * solar_cell_efficiency

            print(f'Processed latitude {lat:.2f} deg, day {day_idx + 1}/{len(days)}: mean power = {mean_power_distribution[lat_idx, day_idx]:.2f} W/m²')

    fig, ax = plt.subplots(figsize=(14, 7))
    max_mean_power = float(np.max(mean_power_distribution))
    levels = np.arange(0, max_mean_power + 5, 5)
    contour = ax.contourf(days, latitudes, mean_power_distribution, levels=levels, cmap='cividis')

    ax.set_xlabel('Day of year')
    ax.set_ylabel('Latitude (deg)')
    ax.set_title('Mean power distribution')
    ax.set_xlim(1, 365)

    fig.colorbar(contour, ax=ax, ticks=levels, label='Mean power (W/m²)')
    plt.tight_layout()
    plt.savefig('mean_power_distribution.png', dpi=200)
    plt.close(fig)

    print('Contour image saved to mean_power_distribution.png')
    print('Mean power distribution shape:', mean_power_distribution.shape)


# Build a latitude/day contour of the mean power distribution
vnorm = np.array([0, 0, -1])  # plane pointing zenith
h = 20000  # sea-level
lat = 0

energy = daily_irradiance(vnorm, h, lat, datetime(2027, 1, 1, 0, 0))


# Build a latitude/day contour of the mean power distribution
vnorm = np.array([0, 0, -1])  # plane pointing zenith
h = 20000  # sea-level
start_date = datetime(2027, 1, 1, 0, 0)
end_date = datetime(2028, 1, 1, 0, 0)
latitudes = np.arange(-80, 80 + 5, 5)

solar_cell_efficiency = 0.15


yearly_mean_power_contour(vnorm, h, latitudes, start_date, end_date, solar_cell_efficiency=solar_cell_efficiency)

