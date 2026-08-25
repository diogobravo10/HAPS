import _utilities as utils
import numpy as np
from solarpy import irradiance_on_plane, daylight_hours
from ambiance import Atmosphere
from datetime import datetime, timedelta
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt



day = datetime(2027, 1, 21, 0, 0)
h = 20000  # altitude in meters
lat = 33 # latitude in degrees

carrying_ability = 0.2 # -> historical guideline
mb = 450 # [Wh/Kg] -> energy density LS-battery
k_prop = 0.0045 # [kg/W] -> propeller mass2power ratio (empyrical relation)

mu_m = 0.6 # effficiency propulsion system 
mu_e = 0.9 # efficiency energy management system
mu_LS = 0.9 # efficiency LS-battery

# Aerodynamic Parametrs
CL = 1.5
CD = 0.0708
g = 9.81
atm = Atmosphere(h)
rho = atm.density[0] # [kg/m3] -> air density at altitude h

x_values = np.linspace(0, 5, 1000) # -> loading (kg/m^2)


# Tracing Curves for Wing Loading, Battery Mass, Propulsion Mass, Payload Mass, and Non-Structural Mass
y_values_wing_loading = CD/CL**(3/2) * x_values**(3/2) * np.sqrt(2*g**3/rho) / mu_m /mu_e
y_values_batteries = x_values * (mu_LS * mb/ (24 - daylight_hours(day, lat)))
y_values_propolsion = x_values / k_prop
# y_values_non_structural_mass =  x_values/ (k_prop + (24 - daylight_hours(day, lat))/(mu_LS * mb))

x_values_payload = carrying_ability * x_values - k_prop * y_values_wing_loading
x_non_structural_mass = carrying_ability * x_values + y_values_wing_loading * (24 - daylight_hours(day, lat)) /mu_LS / mb

P_mean = utils.daily_irradiance(h, lat, day)


plt.plot(x_values, y_values_wing_loading, label="Wing Loading", color='orange')
plt.plot(x_values, y_values_batteries, label="Battery Mass", color='blue')
plt.plot(x_values, y_values_propolsion, label="Propulsion Mass", color='green')
plt.plot(x_values_payload, y_values_wing_loading, label="Payload Mass", color='brown')
plt.plot(x_non_structural_mass, y_values_wing_loading, label="Non Structural Mass", color='purple')
plt.axhline(P_mean, color='red', label='P_mean')

plt.fill_betweenx(
    y_values_wing_loading,
    x_values,
    x_non_structural_mass,
    where=(y_values_wing_loading <= P_mean),  # Restrict filling to below P_mean
    color='c',
    alpha=0.3,
    label='Structural Mass'
)



plt.axhspan(P_mean, plt.ylim()[1], color='grey', alpha=0.3, label='Unfeasible region')



M_Sw = 3.1
Mbat_Sw = 2.1
T_night = (24 - daylight_hours(day, lat))

P_available = min(Mbat_Sw / T_night * mu_LS * mb, CD/CL**(3/2) * (M_Sw)**(3/2) * np.sqrt(2*g**3 / rho) / mu_m / mu_e)



plt.plot(Mbat_Sw, P_available, marker='*', color='b', linestyle='None', label="Battery Mass")
plt.plot(M_Sw, P_available, marker='*', color='orange', linestyle='None', label="Total Mass")




plt.xlim(0, 5)
plt.ylim(0, 100)

plt.xlabel("Wing Loading (kg/m^2)")
plt.ylabel("Available Power (W/m^2)")
plt.title("Battery Mass vs. Wing Loading")
plt.grid(True)
plt.legend()
plt.show(block=False)



h = 20000


start_date = datetime(2027, 1, 1, 0, 0)
end_date = datetime(2028, 1, 1, 0, 0)
dday = 5
days = np.array([
    start_date + timedelta(days=i)
    for i in range(0, (end_date - start_date).days + dday, dday)
], dtype=object)

N_lat = 60
S_lat = -60
dlat = 5
latitudes = np.arange(S_lat, N_lat + dlat, dlat, dtype=float)



solar_cell_efficiency = 0.15



utils.filtering_yearly_mean_power_contour(h, latitudes, days, solar_cell_efficiency=solar_cell_efficiency)








def battery_and_propolsion_mass(x):

    h, lat, day = x[0], x[1], int(round(x[2]))

    if isinstance(day, (int, np.integer)):
        day = datetime(2027, 1, 1, 0, 0) + timedelta(days=int(day))

    rho_solar_cells = 1.0 # [kg/m2] -> mass density solar cells
    k_prop = 0.0045 # [kg/W] -> propeller mass2power ratio (empyrical relation)
    mb = 300 # [Wh/Kg] -> energy density LS-battery

    mu_m = 0.6 # effficiency propulsion system 
    mu_e = 0.9 # efficiency energy management system
    mu_LS = 0.9 # efficiency LS-battery

    P_mean = utils.daily_irradiance(h, lat, day) # [W/m2] -> power per unit area
    T_night = 24 - daylight_hours(day, lat)

    battery_mass = P_mean * T_night /mu_LS/mb
    propolsion_mass = k_prop * P_mean 

    print(f'Battery mass: {battery_mass:.2f} kg/m^2')
    print(f'Propulsion mass: {propolsion_mass:.2f} kg/m^2')
    print(f'Night hours: {T_night:.2f} h')
    print(f'Power density: {P_mean:.2f} W/m^2')


    return battery_mass + propolsion_mass


def max_wing_loading(x):

    h, lat, day = x[0], x[1], int(round(x[2]))

    if isinstance(day, (int, np.integer)):
        day = datetime(2027, 1, 1, 0, 0) + timedelta(days=int(day))

    mb = 300 # [Wh/Kg] -> energy density LS-battery

    mu_m = 0.6 # effficiency propulsion system 
    mu_e = 0.9 # efficiency energy management system
    mu_LS = 0.9 # efficiency LS-battery

    CL = 1.8
    CD = 0.0866
    g = 9.81
    atm = Atmosphere(h)
    rho = atm.density[0] # [kg/m3] -> air density at altitude h


    P_mean = utils.daily_irradiance(h, lat, day) # [W/m2] -> power per unit area
    T_night = 24 - daylight_hours(day, lat)

    battery_mass = P_mean * T_night /mu_LS/mb

    msw = (CL**(3/2) / CD * P_mean * mu_e * mu_m * np.sqrt(rho/2))**(2/3) * g


    print(f'Power density: {P_mean:.2f} W/m^2')
    print(f'Night hours: {T_night:.2f} h')
    print(f'Battery mass: {battery_mass:.2f} kg/m^2')
    print(f'Maximum wing loading: {msw:.2f} kg/m^2')



    return msw - battery_mass




bounds = [(10000., 24000.), # cruise altitudes
          (33.,43.),        # Azores Exclusive Economic Zone
          (0, 365)]         # Year round

integrality = [0, 0, 1]

result = differential_evolution(
    func = max_wing_loading, 
    bounds = bounds, 
    integrality = integrality,
    seed = 25  # For reproducibility
)

print(f'Optimal altitude: {result.x[0]:.2f} m')
print(f'Optimal latitude: {result.x[1]:.2f} deg')
print(f'Optimal day: {datetime(2027, 1, 1, 0, 0) + timedelta(days=int(result.x[2]))} (day of the year)')
print(f'Mass Difference: {max_wing_loading(result.x):.2f} kg/m^2')



# bounds = [(10000., 24000.), # cruise altitudes
#           (33.,43.),        # Azores Exclusive Economic Zone
#           (0, 365)]         # Year round

# integrality = [0, 0, 1]

# result = differential_evolution(
#     func = battery_and_propolsion_mass, 
#     bounds = bounds, 
#     integrality = integrality,
#     seed = 42  # For reproducibility
# )

# print(f'Optimal altitude: {result.x[0]:.2f} m')
# print(f'Optimal latitude: {result.x[1]:.2f} deg')
# print(f'Optimal day: {datetime(2027, 1, 1, 0, 0) + timedelta(days=int(result.x[2]))} (day of the year)')
# print(f'Total mass: {battery_and_propolsion_mass(result.x):.2f} kg/m^2')











# Build a latitude/day contour of the mean power distribution
# h = 20000  # altitude in meters


start_date = datetime(2027, 1, 1, 0, 0)
end_date = datetime(2028, 1, 1, 0, 0)
dday = 5
days = np.array([
    start_date + timedelta(days=i)
    for i in range(0, (end_date - start_date).days + dday, dday)
], dtype=object)

N_lat = 80
S_lat = -80
dlat = 5
latitudes = np.arange(S_lat, N_lat + dlat, dlat, dtype=float)



solar_cell_efficiency = 0.15


utils.yearly_mean_power_contour(h, latitudes, days, solar_cell_efficiency=solar_cell_efficiency)

