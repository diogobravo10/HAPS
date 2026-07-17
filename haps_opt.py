import utilities as utils
import numpy as np
from solarpy import irradiance_on_plane, daylight_hours
from ambiance import Atmosphere
from datetime import datetime, timedelta
from scipy.optimize import brute
from dataclasses import dataclass, field

    
@dataclass(slots=True)
class MassProperties:
    M_Sw = 2.9 # 2.6, 3.1
    Mbat_Sw = 1.8 # 1.8, 2.1

    def __getitem__(self, key):
        return getattr(self, key)
    
@dataclass(slots=True)
class GlobalTimeLocation:
    h = 20000  # altitude in meters
    start_date = datetime(2027, 1, 1, 0, 0)
    end_date = datetime(2028, 1, 1, 0, 0)
    dday = 5
    N_lat = 60
    S_lat = -60
    dlat = 5
    
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
    


ranges = (slice(1.0, 4.0, 0.2), slice(1.0, 4.0, 0.2))  # Define the ranges for M_Sw and Mbat_Sw

user_defined_parameters = UserDefinedParamters()
global_time_and_location_parameters = GlobalTimeLocation()
mass_properties = MassProperties()
solar_cell_efficiency = 0.15

result = brute(utils.obj_fun, ranges, args= (mass_properties, global_time_and_location_parameters, user_defined_parameters, solar_cell_efficiency), finish=None) # I need more constraints

print(f"Optimal M_Sw: {result[0]:.2f} kg, Optimal Mbat_Sw: {result[1]:.2f} kg: Points Validated: {-utils.obj_fun(result)}")



