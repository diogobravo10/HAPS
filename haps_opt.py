import user_defined_parameters as user
import utilities as utils
import numpy as np
from solarpy import irradiance_on_plane, daylight_hours
from ambiance import Atmosphere
from datetime import datetime, timedelta
from scipy.optimize import brute
from dataclasses import dataclass, field

    


ranges = (slice(1.0, 4.0, 0.5), slice(1.0, 4.0, 0.5))  # Define the ranges for M_Sw and Mbat_Sw

user_defined_parameters = user.get_user_defined_parameters()
azores_time_and_location_parameters = user.get_azores_time_location()
solar_cell_efficiency = 0.15

result = brute(utils.obj_fun, ranges, args= (azores_time_and_location_parameters, user_defined_parameters, solar_cell_efficiency), finish=None) # I need more constraints

print(f"Optimal M_Sw: {result[0]:.2f} kg, Optimal Mbat_Sw: {result[1]:.2f} kg: Points Validated: {-utils.obj_fun(result, azores_time_and_location_parameters, user_defined_parameters, solar_cell_efficiency=0.15)}")



