from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
            current_h = h[0]

            daily_energy = daily_irradiance(
                current_h,
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
    if levels.size < 2:
        levels = np.array([0.0, 1.0])
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

def filtering_yearly_mean_power_contour(optimum_mass_properties, global_time_and_location, user_params, solar_cell_efficiency = 0.15, vnorm = np.array([0, 0, -1]), step=timedelta(minutes=15)):

    M_Sw_opt = optimum_mass_properties.M_Sw
    Mbat_Sw_opt = optimum_mass_properties.Mbat_Sw

    start_h = global_time_and_location.start_h
    end_h = global_time_and_location.end_h
    dh = global_time_and_location.dh
    start_date = global_time_and_location.start_date
    end_date = global_time_and_location.end_date
    dday = global_time_and_location.dday
    N_lat = global_time_and_location.N_lat
    S_lat = global_time_and_location.S_lat
    dlat = global_time_and_location.dlat

    carrying_ability = user_params.carrying_ability
    mb = user_params.mb
    k_prop = user_params.k_prop
    mu_m = user_params.mu_m
    mu_e = user_params.mu_e
    mu_LS = user_params.mu_LS
    CD = user_params.CD
    CL = user_params.CL
    g = user_params.g
    
    
    
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
    h_array = np.arange(start_h, end_h + dh, dh, dtype=float)

    atm = Atmosphere(h_array[0])
    rho = atm.density[0] # [kg/m3] -> air density at altitude h

    mean_power_distribution = np.zeros((len(latitudes_array), len(days_array)))


    for lat_idx, lat in enumerate(latitudes_array):
        for day_idx, current_day in enumerate(days_array):
            current_lat = lat
            day_number = day_numbers[day_idx]
            current_h = h_array[0]
            daily_energy = daily_irradiance(
                current_h,
                current_lat,
                current_day,
                solar_cell_efficiency=solar_cell_efficiency,
                step=step,
            )

            T_night = (24 - daylight_hours(current_day, current_lat))

            P_available = min(Mbat_Sw_opt / T_night * mu_LS * mb, CD/CL**(3/2) * (M_Sw_opt)**(3/2) * np.sqrt(2*g**3 / rho) / mu_m / mu_e)

            mean_power_distribution[lat_idx, day_idx] = daily_energy if daily_energy > P_available else 0

            print(f'Processed latitude {current_lat:.2f} deg, day {day_number}/{total_days}: mean power = {mean_power_distribution[lat_idx, day_idx]:.2f} W/m²')


    fig, ax = plt.subplots(figsize=(14, 7))
    max_mean_power = float(np.max(mean_power_distribution))
    levels = np.arange(0, max_mean_power + 5, 5)
    if levels.size < 2:
        levels = np.array([0.0, 1.0])
    contour = ax.contourf(day_numbers, latitudes_array, mean_power_distribution, levels=levels, cmap='cividis')

    ax.set_xlabel('Day of year')
    ax.set_ylabel('Latitude (deg)')
    ax.set_title(f'Mean power distribution at h = {h_array[0]}')
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

    fig.colorbar(contour, ax=ax, ticks=levels, label='Mean power (W/m²)')
    plt.tight_layout()
    filename = f'mean_power_distribution_{h_array[0]}.png'
    plt.savefig(filename, dpi=200)
    plt.close(fig)

    print(f'Contour image saved to {filename}')
    print('Mean power distribution shape:', mean_power_distribution.shape)

def feasibility_study(x_values, optimum_mass_properties, time_and_location, user_params, solar_cell_efficiency=0.15):


    M_Sw_opt = optimum_mass_properties.M_Sw
    Mbat_Sw_opt = optimum_mass_properties.Mbat_Sw

    carrying_ability = user_params.carrying_ability
    mb = user_params.mb
    k_prop = user_params.k_prop
    mu_m = user_params.mu_m
    mu_e = user_params.mu_e
    mu_LS = user_params.mu_LS
    CD = user_params.CD
    CL = user_params.CL
    g = user_params.g

    day = time_and_location.day
    h = time_and_location.h
    lat = time_and_location.lat

    atm = Atmosphere(h)
    rho = atm.density[0] # [kg/m3] -> air density at altitude h

    T_night = (24 - daylight_hours(day, lat))

    fig, ax = plt.subplots()

    #############################################################################
    #                                                                           #
    #                                                                           #
    #   Constraint Curves for Wing Loading, Battery Mass vs. Solar Irradiance   #
    #                                                                           #
    #                                                                           #
    #############################################################################

    y_values_wing_loading = CD/CL**(3/2) * x_values**(3/2) * np.sqrt(2*g**3/rho) / mu_m /mu_e
    line_wing_loading, = ax.plot(x_values, y_values_wing_loading, label="Max. Wing Mass", color='orange')

    y_values_batteries = x_values * (mu_LS * mb/ T_night)
    line_battery_mass, = ax.plot(x_values, y_values_batteries, label="Min. Battery Mass", color='magenta')

    P_mean = daily_irradiance(h, lat, day, solar_cell_efficiency)
    line_max_power = ax.axhline(P_mean, color='red', label='Max. Power')
    unfeasible_region = ax.axhspan(P_mean, ax.get_ylim()[1], color='grey', alpha=0.3, label='Unfeasible region')

    #############################################################################
    #                                                                           #
    #                                                                           #
    #   Mass Propulsion System and Payload Mass vs. Solar Irradiance            #
    #                                                                           #
    #                                                                           #
    #############################################################################

    y_values_propolsion = x_values / k_prop
    x_values_payload = carrying_ability * x_values - k_prop * y_values_wing_loading
    line_propulsion_mass, = ax.plot(x_values, y_values_propolsion, label="Propulsion Mass", color='green')
    line_payload_mass, = ax.plot(x_values_payload, y_values_wing_loading, label=f"Payload Mass ({carrying_ability:.2f})", color="brown")

    #############################################################################
    #                                                                           #
    #                                                                           #
    #   Non Structural Mass vs. Solar Irradiance                                #
    #                                                                           #
    #                                                                           #
    #############################################################################

    # x_non_structural_mass = carrying_ability * x_values + y_values_wing_loading * T_night /mu_LS / mb
    # line_non_structural_mass, = ax.plot(x_non_structural_mass, y_values_wing_loading, label="Non Structural Mass", color='purple')

    #############################################################################
    #                                                                           #
    #                                                                           #
    #   Structural Mass vs. Solar Irradiance                                    #
    #                                                                           #
    #                                                                           #
    #############################################################################

    # structural_mass_region =ax.fill_betweenx(
    #     y_values_wing_loading,
    #     x_values,
    #     x_non_structural_mass,
    #     where=(y_values_wing_loading <= P_mean),  # Restrict filling to below P_mean
    #     color='c',
    #     alpha=0.3,
    #     label='Structural Mass'
    # )


    upper_boundary = np.minimum(y_values_batteries, P_mean)

    tradeoff_mass_region = ax.fill_between(
        x_values,
        y1=upper_boundary,             # Capped upper limit (below battery AND below P_mean)
        y2=y_values_wing_loading,      # Lower limit (above wing loading)
        where=(y_values_wing_loading < P_mean),  # Only fill where the lower boundary is valid
        color='c',
        alpha=0.3,
        label='Trade-Off Mass'
    )

    #############################################################################
    #                                                                           #
    #                                                                           #
    #   Optimum Design Point                                                    #
    #                                                                           #
    #                                                                           #
    #############################################################################

    P_available = min(Mbat_Sw_opt / T_night * mu_LS * mb, CD/CL**(3/2) * (M_Sw_opt)**(3/2) * np.sqrt(2*g**3 / rho) / mu_m / mu_e)
    non_structural_mass = carrying_ability * M_Sw_opt + Mbat_Sw_opt

    line_available_power = ax.axhline(P_available, color='blue', linestyle='--', label='Available Power')
    dp_battery_mass, = ax.plot(Mbat_Sw_opt, P_available, marker='.', markerfacecolor='magenta', linestyle='None', markersize=10, markeredgecolor='black', markeredgewidth=1, label="Battery Mass")
    dp_total_mass, = ax.plot(M_Sw_opt, P_available, marker='.', markerfacecolor='orange', linestyle='None', markersize=10, markeredgecolor='black', markeredgewidth=1, label="Total Mass")
    dp_non_structural_mass, = ax.plot(non_structural_mass, P_available, marker='.', markerfacecolor='purple', linestyle='None', markersize=10, markeredgecolor='black', markeredgewidth=1, label="Non Structural Mass")


    #############################################################################
    #                                                                           #
    #                                                                           #
    #   Plots                                                                   #
    #                                                                           #
    #                                                                           #
    #############################################################################

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 100)

    ax.set_xlabel("Wing Loading (kg/m^2)")
    ax.set_title("Trade-Off Study: Mass vs Solar Irradiance")

    ax.set_ylabel("Solar Irradiance (W/m²)", color='red')
    ax.tick_params(axis='y', colors='red')
    ax.spines['left'].set_color('red')

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim()[0] * mu_m * mu_e, ax.get_ylim()[1] * mu_m * mu_e)
    ax2.set_ylabel("Available Power (W/m^2)", color='blue')
    ax2.tick_params(axis='y', colors='blue')
    ax2.spines['right'].set_color('blue')

    ax.grid(True)
    ax.legend(handles=[
        line_max_power,
        line_available_power,
        line_wing_loading,
        line_battery_mass,
        line_propulsion_mass,
        line_payload_mass,
        # line_non_structural_mass,
        unfeasible_region,
        tradeoff_mass_region,
        dp_total_mass,
        dp_battery_mass,
        dp_non_structural_mass
    ])
    plt.show()

    return

def mission_planning(payload_mass, optimum_mass_properties, time_and_location, user_params):

    M_Sw_opt = optimum_mass_properties.M_Sw
    Mbat_Sw_opt = optimum_mass_properties.Mbat_Sw

    carrying_ability = user_params.carrying_ability
    mb = user_params.mb
    k_prop = user_params.k_prop
    mu_m = user_params.mu_m
    mu_e = user_params.mu_e
    mu_LS = user_params.mu_LS
    CD = user_params.CD
    CL = user_params.CL
    g = user_params.g

    day = time_and_location.day
    h = time_and_location.h[0]
    lat = time_and_location.lat

    atm = Atmosphere(h)
    rho = atm.density[0] # [kg/m3] -> air density at altitude h

    T_night = (24 - daylight_hours(day, lat))

    # P_prop = -np.inf
    # for day in days_array:
    #     for lat in latitudes_array:
    #         for h in h_array:

    #             T_night = (24 - daylight_hours(day, lat))
    #             atm = Atmosphere(h)
    #             rho = atm.density[0] # [kg/m3] -> air density at altitude h

    #             P_available = min(Mbat_Sw / T_night * mu_LS * mb, CD/CL**(3/2) * (M_Sw)**(3/2) * np.sqrt(2*g**3 / rho) / mu_m / mu_e)
                
    #             if P_available > P_prop:
    #                 P_prop = P_available
    # mpayload_SW = carrying_ability * M_Sw_opt -  k_prop * P_available


    return


def obj_fun(design_vars, global_time_and_location, user_params, solar_cell_efficiency=0.15):

    score = 0
    M_Sw, Mbat_Sw = design_vars[0], design_vars[1]

    start_h = global_time_and_location.start_h
    end_h = global_time_and_location.end_h
    dh = global_time_and_location.dh
    start_date = global_time_and_location.start_date
    end_date = global_time_and_location.end_date
    dday = global_time_and_location.dday
    N_lat = global_time_and_location.N_lat
    S_lat = global_time_and_location.S_lat
    dlat = global_time_and_location.dlat

    carrying_ability = user_params.carrying_ability
    mb = user_params.mb
    k_prop = user_params.k_prop
    mu_m = user_params.mu_m
    mu_e = user_params.mu_e
    mu_LS = user_params.mu_LS
    CD = user_params.CD
    CL = user_params.CL
    g = user_params.g
       
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
    h_array = np.arange(start_h, end_h + dh, dh, dtype=float)

    P_prop = -np.inf
    for day in days_array:
        for lat in latitudes_array:
            for h in h_array:

                T_night = (24 - daylight_hours(day, lat))
                atm = Atmosphere(h)
                rho = atm.density[0] # [kg/m3] -> air density at altitude h

                P_available = min(Mbat_Sw / T_night * mu_LS * mb, CD/CL**(3/2) * (M_Sw)**(3/2) * np.sqrt(2*g**3 / rho) / mu_m / mu_e)
                
                if P_available > P_prop:
                    P_prop = P_available

    for day in days_array:
        day_verified = False

        for lat in latitudes_array:
            if day_verified:
                break

            for h in h_array:

                P_mean = daily_irradiance(h, lat, day, solar_cell_efficiency)
                T_night = (24 - daylight_hours(day, lat))
                atm = Atmosphere(h)
                rho = atm.density[0] # [kg/m3] -> air density at altitude h


                if Mbat_Sw >= P_mean * T_night / mu_LS / mb: # There is enough Mass of Batteries to Power the flight through the night

                    if P_mean * mu_m * mu_e >= CD/CL**(3/2) * (M_Sw)**(3/2) * np.sqrt(2*g**3 / rho): # There is enough Irradiance to Power the Flight

                        if k_prop * P_prop < carrying_ability * M_Sw: # There is enough mass for payload and the payload is heavier than the propolsion system

                            if 0.8 * M_Sw - Mbat_Sw > 0.2: # There is enough mass for the structure

                                score = score + 1
                                print(score)
                                day_verified = True
                                break


    return -score
