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
    M_Sw = 3.1
    Mbat_Sw = 2.1

    def __getitem__(self, key):
        return getattr(self, key)
    
@dataclass(slots=True)
class TimeLocation:
    day = datetime(2027, 6, 21, 0, 0)
    h = 20000  # altitude in meters
    lat = 33 # latitude in degrees
    
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
    mass_properties = MassProperties()

    wing_loading_array = np.linspace(0, 5, 1000) # -> loading (kg/m^2)


    utils.feasibility_study(wing_loading_array, mass_properties, time_and_location_parameters, user_defined_parameters)