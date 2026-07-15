import numpy as np
import matplotlib.pyplot as plt
from solarpy import daylight_hours, irradiance_on_plane
from datetime import datetime, timedelta
from ambiance import Atmosphere
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


def obj_fun(x):

    score = 0
    M_Sw, Mbat_Sw = x[0], x[1]

    h_start = 2000
    h_end = 2400
    dh = 5
    h_array = np.arange(h_start, h_end + dh, h_end, dtype=float)


    date_start = datetime(2027, 1, 1, 0, 0)
    date_end = datetime(2028, 1, 1, 0, 0)
    dday = 5
    day_array = np.array([
        date_start + timedelta(days=i)
        for i in range(0, (date_end - date_start).days + dday, dday)
    ], dtype=object)

    N_lat = 80
    S_lat = -80
    dlat = 5
    lat_array = np.arange(S_lat, N_lat + dlat, dlat, dtype=float)

    carrying_ability = 0.2


    # Propolsion/Energy System
    mb = 450 # [Wh/Kg] -> energy density LS-battery
    mu_m = 0.6 # effficiency propulsion system 
    mu_e = 0.9 # efficiency energy management system
    mu_LS = 0.9 # efficiency LS-battery
    k_prop = 0.0045 # [kg/W] -> propeller mass2power ratio (empyrical relation)

    # Aerodynamic Parametrs
    CL = 1.5
    CD = 0.0708
    g = 9.81

    for h in h_array:

        for lat in lat_array:

            for day in day_array:


                P_mean = daily_irradiance(h, lat, day)
                T_night = (24 - daylight_hours(day, lat))
                atm = Atmosphere(h)
                rho = atm.density[0] # [kg/m3] -> air density at altitude h


                if Mbat_Sw >= P_mean * T_night / mu_LS / mb: # There is enough Mass of Batteries to Power the flight through the night


                    if P_mean * mu_m * mu_e >= CD/CL**(3/2) * (M_Sw)**(3/2) * np.sqrt(2*g**3 / rho): # There is enough Irradiance to Power the Flight


                        P_available = min(Mbat_Sw / T_night * mu_LS * mb, CD/CL**(3/2) * (M_Sw)**(3/2) * np.sqrt(2*g**3 / rho) / mu_m / mu_e)

                        if k_prop * P_available < carrying_ability * M_Sw:


                            if 0.8 * M_Sw - Mbat_Sw > 0.5:

                                score = score + 1

                                print(score)



    return -score
