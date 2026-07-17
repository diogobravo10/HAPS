import utilities as utils
import numpy as np
from solarpy import irradiance_on_plane, daylight_hours
from ambiance import Atmosphere
from datetime import datetime, timedelta
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

    
@dataclass(slots=True)
class MassProperties:
    M_Sw = 3.1 # 2.6, 3.1
    Mbat_Sw = 2.1 # 1.8, 2.1

    def __getitem__(self, key):
        return getattr(self, key)
    
@dataclass(slots=True)
class TimeLocation:
    day = datetime(2027, 6, 21, 0, 0)
    h = [23000]  # altitude in meters
    lat = 33 # latitude in degrees
    
    def __getitem__(self, key):
        return getattr(self, key)


@dataclass(slots=True)
class GlobalTimeLocation:
    h = [23000]  # altitude in meters
    start_date = datetime(2027, 1, 1, 0, 0)
    end_date = datetime(2028, 1, 1, 0, 0)
    dday = 5
    N_lat = 60
    S_lat = -60
    dlat = 5
    
    def __getitem__(self, key):
        return getattr(self, key)
    

@dataclass(slots=True)
class AzoresTimeLocation:
    h = [23000]  # altitude in meters
    start_date = datetime(2027, 1, 1, 0, 0)
    end_date = datetime(2028, 1, 1, 0, 0)
    dday = 5
    N_lat = 43
    S_lat = 33
    dlat = 2
    
    def __getitem__(self, key):
        return getattr(self, key)

@dataclass(slots=True)
class UserDefinedParamters:
    carrying_ability: float = 0.2 # -> historical guideline
    mb : float = 450 # [Wh/Kg] -> energy density LS-battery
    k_prop: float = 0.0045 # [kg/W] -> propeller
    mu_m: float = 0.6 # effficiency propulsion system
    mu_e: float = 0.9 # efficiency energy management system
    mu_LS: float = 0.9 # efficiency LS-battery
    CL: float = 1.5
    CD: float = 0.0708
    g: float = 9.81 # gravitational acceleration
    
    def __getitem__(self, key):
        return getattr(self, key)
    


if __name__ == '__main__':

    user_defined_parameters = UserDefinedParamters()
    time_and_location_parameters = TimeLocation()
    global_time_and_location_parameters = GlobalTimeLocation()
    azores_time_and_location_parameters = AzoresTimeLocation()
    mass_properties = MassProperties()

    wing_loading_array = np.linspace(0, 5, 1000) # -> loading (kg/m^2)


    utils.feasibility_study(wing_loading_array, mass_properties, time_and_location_parameters, user_defined_parameters, solar_cell_efficiency=0.15)

    count_max_days_in_a_year = utils.obj_fun([3.1, 2.1], azores_time_and_location_parameters, user_defined_parameters, solar_cell_efficiency=0.15)
    print(f'Days in a Year: {count_max_days_in_a_year}')


    utils.filtering_yearly_mean_power_contour(mass_properties, global_time_and_location_parameters, user_defined_parameters, solar_cell_efficiency=0.15)

