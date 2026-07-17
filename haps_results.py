import user_defined_parameters as user
import utilities as utils
import numpy as np
from solarpy import irradiance_on_plane, daylight_hours
from ambiance import Atmosphere
from datetime import datetime, timedelta
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

if __name__ == '__main__':

    mass_properties = user.get_mass_properties()
    user_defined_parameters = user.get_user_defined_parameters()
    time_and_location_parameters = user.get_time_location()
    global_time_and_location_parameters = user.get_global_time_location()
    azores_time_and_location_parameters = user.get_azores_time_location()

    wing_loading_array = np.linspace(0, 5, 1000) # -> loading (kg/m^2)

    utils.feasibility_study(wing_loading_array, mass_properties, time_and_location_parameters, user_defined_parameters, solar_cell_efficiency=0.15)

    count_max_days_in_a_year = utils.obj_fun([3.1, 2.1], azores_time_and_location_parameters, user_defined_parameters, solar_cell_efficiency=0.15)
    print(f'Days in a Year: {-count_max_days_in_a_year}')

    utils.filtering_yearly_mean_power_contour(mass_properties, global_time_and_location_parameters, user_defined_parameters, solar_cell_efficiency=0.15)

