import utilities as utils
import numpy as np
from solarpy import irradiance_on_plane, daylight_hours
from ambiance import Atmosphere
from datetime import datetime, timedelta
from scipy.optimize import brute


ranges = (slice(0.1, 5, 0.1), slice(0.1, 5, 0.1))  # Define the ranges for M_Sw and Mbat_Sw


result = brute(utils.obj_fun, ranges, finish=None) # I need more constraints

print(f"Optimal M_Sw: {result[0]:.2f} kg, Optimal Mbat_Sw: {result[1]:.2f} kg: Points Validated: {-utils.obj_fun(result)}")



# h_start = 2000
# h_end = 2400
# dh = 5
# h_array = np.arange(h_start, h_end + dh, h_end, dtype=float)


# date_start = datetime(2027, 1, 1, 0, 0)
# date_end = datetime(2028, 1, 1, 0, 0)
# dday = 5
# day_array = np.array([
#     date_start + timedelta(days=i)
#     for i in range(0, (date_end - date_start).days + dday, dday)
# ], dtype=object)

# N_lat = 80
# S_lat = -80
# dlat = 5
# lat_array = np.arange(S_lat, N_lat + dlat, dlat, dtype=float)

# print("Total Points:", len(h_array) * len(day_array) * len(lat_array))
