import utilities as utils
import numpy as np
from solarpy import irradiance_on_plane, daylight_hours
from ambiance import Atmosphere
from datetime import datetime, timedelta
from scipy.optimize import brute


ranges = (slice(0.1, 5.1, 0.2), slice(0.1, 5.1, 0.2))  # Define the ranges for M_Sw and Mbat_Sw


result = brute(utils.obj_fun, ranges, finish=None) # I need more constraints

print(f"Optimal M_Sw: {result[0]:.2f} kg, Optimal Mbat_Sw: {result[1]:.2f} kg: Points Validated: {-utils.obj_fun(result)}")



