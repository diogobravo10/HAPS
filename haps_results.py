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

    payload_mass = 5 # [kg]
    mass_properties = user.get_mass_properties(M_Sw = 2.9, Mbat_Sw = 2.0)
    time_and_location = user.get_time_location(day = datetime(2027, 6, 21, 0, 0), lat = 33, h = 23000)


    user_defined_parameters = user.get_user_defined_parameters()
    global_time_and_location = user.get_global_time_location(dday = 5) # dday -> resolution
    azores_flight_envelope = user.get_azores_flight_envelope(dday = 1) # dday -> resolution

    wing_loading_array = np.linspace(0, 5, 1000) # -> loading (kg/m^2)


    utils.mission_planning(payload_mass = 5.0, optimum_mass_properties = mass_properties, time_and_location=time_and_location, flight_envelope_time_and_location = azores_flight_envelope, user_params = user_defined_parameters, solar_cell_efficiency=0.15)
    # utils.mission_planning(Sw = 20.0, optimum_mass_properties = mass_properties, flight_envelope_time_and_location = azores_flight_envelope, user_params = user_defined_parameters)
    
    
    utils.feasibility_study(wing_loading_array, mass_properties, time_and_location, user_defined_parameters, solar_cell_efficiency=0.15)

    count_max_days_in_a_year = utils.obj_fun([mass_properties.M_Sw, mass_properties.Mbat_Sw], azores_flight_envelope, user_defined_parameters, solar_cell_efficiency=0.15)
    print(f'Days in a Year: {-count_max_days_in_a_year}')

    utils.filtering_yearly_mean_power_contour(mass_properties, global_time_and_location, user_defined_parameters, solar_cell_efficiency=0.15)

    a=1