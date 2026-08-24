import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from solarpy import daylight_hours, irradiance_on_plane
from datetime import datetime, timedelta
from ambiance import Atmosphere

# import scienceplots

# plt.style.use("science")

vnorm = np.array([0, 0, -1])  # plane pointing zenith


def daily_irradiance(h, lat, start_date, solar_cell_efficiency=0.15, vnorm = np.array([0, 0, -1]), step=timedelta(minutes=15)):
    """Estimate mean available power density from solar irradiance over one day.

    Parameters
    - h: altitude in meters
    - lat: latitude in degrees
    - start_date: day under analysis
    - solar_cell_efficiency: fraction of incident irradiance converted to electrical power
    - vnorm: surface normal vector for the solar panel
    - step: time resolution for numerical integration

    Returns
    - power_density_available: mean available power density (W/m^2) averaged over 24h
    """
    dates = []
    current = start_date
 
    end_date = start_date + timedelta(days=1)

    while current <= end_date:
        dates.append(current)
        current += step

    t = np.array([(d - start_date).total_seconds() for d in dates], dtype=float)
    G = np.array([irradiance_on_plane(vnorm, h, d, lat) for d in dates], dtype=float)

    # J/m^2
    energy_j_per_m2 = np.trapezoid(G, t)

    # Wh/m^2
    energy_wh_per_m2 = energy_j_per_m2 / 3600.0  
    
    power_density_available = energy_wh_per_m2 / 24.0 * solar_cell_efficiency
    
    return power_density_available


def yearly_mean_power_contour(h, latitudes, days, solar_cell_efficiency = 0.15, vnorm = np.array([0, 0, -1]), step=timedelta(minutes=15)):
    """Compute and plot a latitude vs day contour of mean daily power.

    Iterates over provided `latitudes` and `days`, computes the daily
    irradiance using `daily_irradiance`, and writes a contour PNG
    `mean_power_distribution.png` showing mean power (W/m^2).
    """

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

            print(f'Processed latitude {current_lat:.2f} deg, day {day_number}/{total_days}: mean power = {mean_power_distribution[lat_idx, day_idx]:.2f} W/m^2')

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

    fig.colorbar(contour, ax=ax, ticks=levels, label=r'Mean power ($W/m^2$)')
    plt.tight_layout()
    plt.savefig('mean_power_distribution.png', dpi=200)
    plt.close(fig)

    print('Contour image saved to mean_power_distribution.png')

def filtering_yearly_mean_power_contour(optimum_mass_properties, global_time_and_location, user_params, solar_cell_efficiency = 0.15, vnorm = np.array([0, 0, -1]), step=timedelta(minutes=15)):
    """Like `yearly_mean_power_contour` but filter cells where available
    power (from batteries or propulsion limit) is insufficient.

    Parameters are pulled from `optimum_mass_properties`,
    `global_time_and_location` and `user_params`. Produces and shows
    a contour plot (PNG file) of feasible mean power values.
    """

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

            print(f'Processed latitude {current_lat:.2f} deg, day {day_number}/{total_days}: mean power = {mean_power_distribution[lat_idx, day_idx]:.2f} W/m^2')


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

    fig.colorbar(contour, ax=ax, ticks=levels, label=r'Mean power ($W/m^2$)')
    plt.tight_layout()
    filename = f'fig_mean_power_distribution_{h_array[0]}.png'
    plt.savefig(filename, dpi=200)
    plt.pause(0.1)
    plt.show(block=False)


    print(f'Contour image saved to {filename}')


def feasibility_study(x_values, optimum_mass_properties, time_and_location, user_params, solar_cell_efficiency=0.15):
    """Create a feasibility trade-off plot for wing loading vs irradiance.

    Plots constraints (wing loading, battery mass, propulsion mass,
    payload) and highlights feasible/unfeasible regions given the
    `optimum_mass_properties` and `user_params` at the provided
    `time_and_location`.
    """

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

    P_available = min(Mbat_Sw_opt / T_night * mu_LS * mb, CD/CL**(3/2) * (M_Sw_opt)**(3/2) * np.sqrt(2*g**3 / rho) / mu_m / mu_e) # P_prop_installed > P_bat or P_used
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

    ax.set_xlabel(r"Wing Loading ($kg/m^2$)")
    ax.set_title("Trade-Off Study: Mass vs Solar Irradiance")

    ax.set_ylabel(r"Solar Irradiance ($W/m^2$)", color='red')
    ax.tick_params(axis='y', colors='red')
    ax.spines['left'].set_color('red')

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim()[0] * mu_m * mu_e, ax.get_ylim()[1] * mu_m * mu_e)
    ax2.set_ylabel(r"Available Power ($W/m^2$)", color='blue')
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

    # legend = ax.legend(frameon=True)
    # legend.get_frame().set_edgecolor('black')


    plt.pause(0.1)
    plt.show(block=False)

    filename = f'fig_feasibility_plot.png'
    plt.savefig(filename, dpi=200)

    print(f'Contour image saved to {filename}')

    return

def mission_planning(optimum_mass_properties, time_and_location, flight_envelope_time_and_location, user_params, payload_mass = None, Sw = None, solar_cell_efficiency=0.15):

    #############################################################################
    #                                                                           #
    #                                                                           #
    #   Initialization                                                          #
    #                                                                           #
    #                                                                           #
    #############################################################################

    """Produce mass breakdown and basic mission metrics for a design.

    Generates a pie chart of structure, battery, payload and propulsion
    mass using the provided `optimum_mass_properties` and flight
    envelope. Prints several mission summary metrics (power, speed,
    altitude, surface area estimates).
    """

    M_Sw_opt = optimum_mass_properties.M_Sw
    Mbat_Sw_opt = optimum_mass_properties.Mbat_Sw

    current_day = time_and_location.day
    current_h = time_and_location.h
    current_lat = time_and_location.lat

    start_h = flight_envelope_time_and_location.start_h
    end_h = flight_envelope_time_and_location.end_h
    dh = flight_envelope_time_and_location.dh
    start_date = flight_envelope_time_and_location.start_date
    end_date = flight_envelope_time_and_location.end_date
    dday = flight_envelope_time_and_location.dday
    N_lat = flight_envelope_time_and_location.N_lat
    S_lat = flight_envelope_time_and_location.S_lat
    dlat = flight_envelope_time_and_location.dlat

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
    for d in days_array:
        for l in latitudes_array:
            for hh in h_array:

                T_night = (24 - daylight_hours(d, l))
                atm = Atmosphere(hh)
                rho = atm.density[0] # [kg/m3] -> air density at altitude h

                P_available = min(Mbat_Sw_opt / T_night * mu_LS * mb, CD/CL**(3/2) * (M_Sw_opt)**(3/2) * np.sqrt(2*g**3 / rho) / mu_m / mu_e)
                
                if P_available > P_prop:
                    P_prop = P_available
    
    mprop_Sw = k_prop * P_prop
    mpayload_Sw = (carrying_ability * M_Sw_opt -  k_prop * P_available)
    
    if payload_mass:
        Sw = payload_mass / mpayload_Sw

    if Sw:
        payload_mass = Sw * mpayload_Sw


    mstruct_Sw = M_Sw_opt - Mbat_Sw_opt - mpayload_Sw
    labels = ['Structure', 'Battery', 'Payload', 'Propolsion']
    sizes = [mstruct_Sw, Mbat_Sw_opt, mpayload_Sw, mprop_Sw]
    colors = ['blue', 'magenta', 'brown', 'green']
    # colors = ["#1F3A73", '#ff7f0e', '#2ca02c', '#d62728']
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False,
        wedgeprops={'edgecolor': 'black'}
    )

    legend_handles = [
        Patch(facecolor=colors[i], edgecolor='black', label=labels[i])
        for i in range(len(labels))
    ]
    ax.legend(
        handles=legend_handles,
        loc='lower center',
        frameon=True,
        bbox_to_anchor=(0.5, -0.1),
        ncol=4,
        borderaxespad=0.0
    )

    ax.set_title(f'Mass breakdown\nTotal Mass = {M_Sw_opt*Sw:.2f} kg')
    ax.axis('equal')  # keeps the pie circular
    fig.subplots_adjust(right=0.75)

    plt.pause(0.1)
    plt.show(block=False)

    filename = 'fig_mass_pie_chart.png'
    plt.savefig(filename, dpi=200)

    print(f'Contour image saved to {filename}')
  
    T_night = (24 - daylight_hours(current_day, current_lat))
    atm = Atmosphere(current_h)
    rho = atm.density[0] 

    P_mean = daily_irradiance(current_h, current_lat, current_day, solar_cell_efficiency)
    P_available = min(Mbat_Sw_opt / T_night * mu_LS * mb, CD/CL**(3/2) * (M_Sw_opt)**(3/2) * np.sqrt(2*g**3 / rho) / mu_m / mu_e)
    
    V = np.sqrt(2*M_Sw_opt/CL/rho)
    
    print(f"Payload Mass: {payload_mass} [kg]")
    print(f"Surface Area: {Sw} [m^2]")
    print(f"P_solar: {P_mean * Sw} [W]")
    print(f"P_prop_installed: {P_prop * Sw} [W]")
    print(f"P_available: {P_available * Sw} [W]")
    print(f"Flight Speed: {V} [m/s]")
    print(f"Altitude: {current_h} [m]")




    return


def obj_fun(design_vars, global_time_and_location, user_params, solar_cell_efficiency=0.15):
    """Objective function used by optimizers to find feasible designs.

    Evaluates a candidate design (`design_vars` = [M_Sw, Mbat_Sw]) over a
    grid of days, latitudes and altitudes from `global_time_and_location`.
    Returns a negative count of days that meet all feasibility checks
    (higher is better for the optimizer). Returns a large penalty if
    the design is invalid (e.g., total mass less than battery mass).
    """

    score = 0
    M_Sw, Mbat_Sw = design_vars[0], design_vars[1]

    if M_Sw < Mbat_Sw:
        
        return 1e9

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
                                print(f'score = {score}', end=', ', flush=True)
                                day_verified = True
                                break


    return -score
