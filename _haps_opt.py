import _user_defined_parameters as user
import _utilities as utils
import numpy as np
from solarpy import irradiance_on_plane, daylight_hours
from ambiance import Atmosphere
from datetime import datetime, timedelta
from scipy.optimize import brute, minimize, differential_evolution, shgo
from dataclasses import dataclass, field

    

design_vars_0 = [3.0, 2.0]
bounds = [[0.1, 5.0], [0.1, 5.0]]
ranges = (slice(1.0, 4.0, 0.5), slice(1.0, 4.0, 0.5))  # Define the ranges for M_Sw and Mbat_Sw

user_defined_parameters = user.get_user_defined_parameters()
azores_flight_envelope = user.get_azores_flight_envelope(dday = 5)
solar_cell_efficiency = 0.15

result_brute = brute(utils.obj_fun, ranges, args= (azores_flight_envelope, user_defined_parameters, solar_cell_efficiency), finish=None) # I need more constraints
log_brute = "log_brute.csv"
with open(log_brute, 'w') as file:
    file.write(f"Optimal M_Sw: {result_brute[0]:.2f} kg, Optimal Mbat_Sw: {result_brute[1]:.2f} kg: Points Validated: {-utils.obj_fun(result_brute, azores_flight_envelope, user_defined_parameters, solar_cell_efficiency=0.15)}")


result_dif = differential_evolution(utils.obj_fun, bounds, args= (azores_flight_envelope, user_defined_parameters, solar_cell_efficiency), polish=False) # I need more constraints
log_diff = "log_differential_evolution.csv"
with open(log_diff, 'w') as file:
    file.write(f"Optimal M_Sw: {result_dif.x[0]:.2f} kg, Optimal Mbat_Sw: {result_dif.x[1]:.2f} kg: Points Validated: {-utils.obj_fun([result_dif.x[0], result_dif.x[1]], azores_flight_envelope, user_defined_parameters, solar_cell_efficiency=0.15)}")


result_shgo = differential_evolution(utils.obj_fun, bounds, args= (azores_flight_envelope, user_defined_parameters, solar_cell_efficiency), polish=False) # I need more constraints
log_shgo = "log_shgo.csv"
with open(log_shgo, 'w') as file:
    file.write(f"Optimal M_Sw: {result_shgo.x[0]:.2f} kg, Optimal Mbat_Sw: {result_shgo.x[1]:.2f} kg: Points Validated: {-utils.obj_fun([result_shgo.x[0], result_shgo.x[1]], azores_flight_envelope, user_defined_parameters, solar_cell_efficiency=0.15)}")




result_minimize = minimize(fun = utils.obj_fun, 
                  x0 = design_vars_0, 
                  args= (azores_flight_envelope, user_defined_parameters, solar_cell_efficiency), 
                  bounds = bounds,
                  options = {"ftol": 1e-02, "gtol": 1e-05, "eps": 1e-1, "maxfun": 15000, "maxiter": 15000}, 
                  ) 

log_minimize = "log_minimize.csv"
with open(log_minimize, 'w') as file:
    file.write(f"Optimal M_Sw: {result_minimize.x[0]:.2f} kg, Optimal Mbat_Sw: {result_minimize.x[1]:.2f} kg: Points Validated: {-utils.obj_fun([result_minimize.x[0], result_minimize.x[1]], azores_flight_envelope, user_defined_parameters, solar_cell_efficiency=0.15)}")




