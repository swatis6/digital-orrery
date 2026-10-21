'''Constants for the entire system. All constants should be defined here, 
and imported to other modules.'''

import numpy as np

#Earth (EGM-96 for gravity // WSG-84 for gps values) 
MU_EARTH = 3.986004418e14 # m^3/s^2, gravitational parameter
R_EARTH = 6378137.0 # m, equatorial radius
J2_EARTH = 1.08262668e-3 # -, oblateness coefficient
OMEGA_EARTH = 7.2921159e-5 # rad/s, rotation rate

#General Consts
G0 = 9.80665
SECONDS_PER_DAY = 86400.0
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi