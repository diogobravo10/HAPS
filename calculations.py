import numpy as np
from solarpy import irradiance_on_plane, daylight_hours
from datetime import datetime, timedelta
from scipy.optimize import differential_evolution
import utilities as utils


solar_cell_efficiency = 0.15

# Irradiance on a given day
h = 20000  # altitude in meters
lat = 43

P_level = utils.daily_irradiance(h, lat, datetime(2027, 12, 21, 0, 0))
print(P_level)




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

    P_level = utils.daily_irradiance(h, lat, day) # [W/m2] -> power per unit area
    T_night = 24 - daylight_hours(day, lat)

    battery_mass = P_level * T_night /mu_LS/mb
    propolsion_mass = k_prop * P_level 

    print(f'Battery mass: {battery_mass:.2f} kg/m^2')
    print(f'Propulsion mass: {propolsion_mass:.2f} kg/m^2')
    print(f'Night hours: {T_night:.2f} h')
    print(f'Power density: {P_level:.2f} W/m^2')


    return battery_mass + propolsion_mass



bounds = [(10000., 24000.), # cruise altitudes
          (33.,43.),        # Azores Exclusive Economic Zone
          (0, 365)]         # Year round

integrality = [0, 0, 1]

result = differential_evolution(
    func = battery_and_propolsion_mass, 
    bounds = bounds, 
    integrality = integrality,
    seed = 42  # For reproducibility
)

print(f'Optimal altitude: {result.x[0]:.2f} m')
print(f'Optimal latitude: {result.x[1]:.2f} deg')
print(f'Optimal day: {datetime(2027, 1, 1, 0, 0) + timedelta(days=int(result.x[2]))} (day of the year)')
print(f'Total mass: {battery_and_propolsion_mass(result.x):.2f} kg/m^2')











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





utils.yearly_mean_power_contour(h, latitudes, days, solar_cell_efficiency=solar_cell_efficiency)

