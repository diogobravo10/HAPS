from dataclasses import dataclass
from datetime import datetime


#############################################################################
#                                                                           #
#                                                                           #
#   Initialize DataClasses                                                  #
#                                                                           #
#                                                                           #
#############################################################################

    
@dataclass(slots=True)
class MassProperties:
    M_Sw: float
    Mbat_Sw: float

    def __getitem__(self, key):
        return getattr(self, key)
    
@dataclass(slots=True)
class TimeLocation:
    day: datetime
    h: list[float] 
    lat: float
    
    def __getitem__(self, key):
        return getattr(self, key)

@dataclass(slots=True)
class GlobalTimeLocation:
    h: list[float]
    start_date: datetime
    end_date: datetime
    dday: int
    N_lat: float
    S_lat: float
    dlat: int
    
    def __getitem__(self, key):
        return getattr(self, key)
    
@dataclass(slots=True)
class UserDefinedParamters:
    carrying_ability: float
    mb : float
    k_prop: float
    mu_m: float
    mu_e: float
    mu_LS: float
    CL: float
    CD: float
    g: float
    
    def __getitem__(self, key):
        return getattr(self, key)
    



#############################################################################
#                                                                           #
#                                                                           #
#   Parameter Definition                                                    #
#                                                                           #
#                                                                           #
#############################################################################


def get_mass_properties(M_Sw, Mbat_Sw):

    m_sw = M_Sw
    mbat_sw = Mbat_Sw

    mass = MassProperties(
            M_Sw = m_sw,
            Mbat_Sw = mbat_sw
    ) 

    return mass


def get_user_defined_parameters():

    carrying_ability = 0.2 # -> historical guideline
    mb = 450 # [Wh/Kg] -> energy density LS-battery
    k_prop = 0.0045 # [kg/W] -> propeller
    mu_m = 0.6 # effficiency propulsion system
    mu_e = 0.9 # efficiency energy management system
    mu_LS = 0.9 # efficiency LS-battery
    CL = 1.5
    CD = 0.0708
    g = 9.81 # gravitational acceleration

    user_params = UserDefinedParamters(
    carrying_ability = carrying_ability,
    mb = mb,
    k_prop = k_prop,
    mu_m = mu_m,
    mu_e = mu_e,
    mu_LS = mu_LS,
    CL = CL,
    CD = CD,
    g = g
    )

    return user_params


def get_time_location():

    day = datetime(2027, 6, 21, 0, 0)
    h = [23000]  # altitude in meters
    lat = 33 # latitude in degrees

    t_l = TimeLocation(
    day = day,
    h = h,
    lat = lat
    )

    return t_l


def get_global_time_location(dday):

    h = [20000]  # altitude in meters
    start_date = datetime(2027, 1, 1, 0, 0)
    end_date = datetime(2028, 1, 1, 0, 0)
    dday = dday
    N_lat = 60
    S_lat = -60
    dlat = 5

    g_t_l = GlobalTimeLocation(
        h = h,
        start_date = start_date,
        end_date = end_date,
        dday = dday,
        N_lat = N_lat,
        S_lat = S_lat,
        dlat = dlat
    )

    return g_t_l


def get_azores_time_location(dday):

    h = [20000]  # altitude in meters
    start_date = datetime(2027, 1, 1, 0, 0)
    end_date = datetime(2028, 1, 1, 0, 0)
    dday = dday
    N_lat = 43
    S_lat = 33
    dlat = 2

    g_t_l = GlobalTimeLocation(
        h = h,
        start_date = start_date,
        end_date = end_date,
        dday = dday,
        N_lat = N_lat,
        S_lat = S_lat,
        dlat = dlat
    )

    return g_t_l

