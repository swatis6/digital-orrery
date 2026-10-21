'''Sun position and eclipse geometry for power, sun sensors, and plots'''

from __future__ import annotations

from datetime import datetime

import numpy as np

from orrery.constants import R_EARTH
from orrery.frames import J2000_JD, julian_date

AU = 1.495978707e11  # one astronomical unit in (m)

def sun_position_eci(epoch: datetime, t: np.ndarray | float = 0.0) -> np.ndarray:
    '''Earth-to-Sun vector in ECI (m)
    Accurate to about 0.01 deg in direction (Vallado, Algorithm 29), which is good enough for now
    Returns (3,) for scalar t; returns (N, 3) for an array.'''
    jd = julian_date(epoch) + np.asarray(t, dtype=float) / 86400.0
    T = (jd - J2000_JD) / 36525.0

    mean_lon = np.radians(280.460 + 36000.771 * T) #mean long - where the Sun would be on the ecliptic if Earth’s orbit was a perfect circle
    M = np.radians(357.5291092 + 35999.05034 * T) #mean anomaly - how far Earth is through its orbit, measured from perihelion 
    ecl_lon = mean_lon + np.radians(1.914666471 * np.sin(M) + 0.019994643 * np.sin(2 * M)) #ecliptic longitude - eq of center adds or subtracts up to ~1.9° to account for uneven speed
    dist = (1.000140612 - 0.016708617 * np.cos(M) - 0.000139589 * np.cos(2 * M)) * AU
    obliquity = np.radians(23.439291 - 0.0130042 * T) #obliquity of epplictic (ie the tilt of the earth)

    r = np.stack([dist * np.cos(ecl_lon),
                  dist * np.cos(obliquity) * np.sin(ecl_lon),
                  dist * np.sin(obliquity) * np.sin(ecl_lon)], axis=-1) #place sun in ecliptic plane and tilt by obliquity around x axis
    return r


def in_eclipse(r_sat: np.ndarray, r_sun: np.ndarray, body_radius: float = R_EARTH) -> np.ndarray:
    '''Cylindrical shadow model: True where the spacecraft is in Earth's shadow.
    r_sat and r_sun are (3,) or (N, 3), ECI. Ignores the penumbra (only a few seconds per eclipse in LEO)
    Eventually will upgrade to a cone.'''
    r_sat = np.atleast_2d(r_sat)
    s_hat = np.atleast_2d(r_sun) / np.linalg.norm(r_sun, axis=-1, keepdims=True) #sun direction unit vector
    along = np.sum(r_sat * s_hat, axis=1) #positive = sunlit, negative = eclipse
    perp = np.linalg.norm(r_sat - along[:, None] * s_hat, axis=1) #calculates position perpendical to sun-earth vector by removing along part from position
    return (along < 0.0) & (perp < body_radius) #in shadow if sat is behind earth AND if its in the shadow of cylinder 