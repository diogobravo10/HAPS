import numpy as np
from datetime import datetime, timedelta
import utilities as utils


vnorm = np.array([0, 0, -1])  # plane pointing zenith


# Irradiance on a given day
h = 20000  # altitude in meters
lat = 0

energy = utils.daily_irradiance(h, lat, datetime(2027, 1, 1, 0, 0))
print(energy)

# Build a latitude/day contour of the mean power distribution
h = 20000  # altitude in meters


start_date = datetime(2027, 1, 1, 0, 0)
end_date = datetime(2028, 1, 1, 0, 0)
dday = 5
days = np.array([
    start_date + timedelta(days=i)
    for i in range(0, (end_date - start_date).days, dday)
], dtype=object)

N_lat = 80
S_lat = -80
dlat = 5
latitudes = np.arange(S_lat, N_lat + dlat, dlat, dtype=float)


solar_cell_efficiency = 0.15


utils.yearly_mean_power_contour(h, latitudes, days, solar_cell_efficiency=solar_cell_efficiency)

