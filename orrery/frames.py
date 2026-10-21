'''Reference-frame conversions & time systems go here.

Conventions:
- ECI: interrial frame w/ z-axis along earth's spin axis, used by the propagator
- ECEF: earth-fixed frame that rotates w/ the earth
ECI to ECEF conversion is a single rotation by the Greenwich Mean Sidereal Time.
this ignores precession, notation, and polar motion, and it treats UTC as UT1. 
Accurate enough for ground tracks, access geom, and ADCS ref vectors for the time being.'''

from __future__ import annotations
from datetime import datetime, timezone
import numpy as np

from orrery.constants import F_EARTH, R_EARTH

J2000_JD = 2451545.0 #julian date - continuous count of days since 4713 BC
J2000 = datetime(2000, 1, 1, 12, 0, 0, 0, tzinfo=timezone.utc)

def julian_date(dt:datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return 2440587.5 + dt.timestamp() / 86400.0 #2440587.5 is the Julian Date of the Unix epoch, and adding it shifts the count to the Julian Date scale. The .5 is there because Julian days begin at noon


def gmst(jd:np.ndarray | float) -> float:
    '''greenwich mean std time in radians - the angle the Earth has rotated relative to the stars, 
    measured at the Greenwich meridian. Also the rotation angle between ECI and ECEF
    Sidereal day = one full spin relative to the stars, is about 23 h 56 m 4 s; 4 mins shorter than a solar day'''
    T = (np.asarray(jd) - J2000_JD) / 36525.0
    seconds = (67310.54841 #GMST at j2000
               + (876600.0 * 3600.0 + #number of seconds in a century
                  8640184.812866) #extra rotation per century
                * T + 0.093104 * T**2 - 6.2e-6 * T**3) #IAU 1982 GMST formula
    return np.mod(seconds, 86400.0) * (2.0 * np.pi / 86400.0) #keeps only the position within the current rotation, then convert to radians

def gmst_at(epoch: datetime, t: np.ndarray | float) -> np.ndarray | float:
    '''GMST (rad) at t seconds after epoch'''
    return gmst(julian_date(epoch) + np.asarray(t) / 86400.0)


def _rot_z(vectors: np.ndarray, angle: np.ndarray | float) -> np.ndarray:
    '''Rotate vectors (3,) or (N, 3) about z by -angle (into a frame rotated by +angle).'''
    v = np.atleast_2d(vectors) #vector of shape (3,) becomes (1, 3)
    c, s = np.cos(angle), np.sin(angle)
    out = np.column_stack((c * v[:, 0] + s * v[:, 1],
                           -s * v[:, 0] + c * v[:, 1],
                           v[:, 2])) #standard rotation matrix
    return out if np.ndim(vectors) == 2 else out[0] #out[0] to unwrap (1,3) back into (3,)

def eci_to_ecef(r_eci: np.ndarray, epoch: datetime, t: np.ndarray | float) -> np.ndarray:
    '''Position(s) from ECI to ECEF'''
    return _rot_z(r_eci, gmst_at(epoch, t))

def ecef_to_eci(r_ecef: np.ndarray, epoch: datetime, t: np.ndarray | float) -> np.ndarray:
    '''Position(s) from ECEF to ECI (inverse of eci_to_ecef)'''
    return _rot_z(r_ecef, -np.asarray(gmst_at(epoch, t)))


def ecef_to_latlon(r_ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''Geocentric latitude and longitude in degrees (longitude in [-180, 180)).
    Geocentric latitude differs from geodetic (map) latitude by max ~0.19 deg, fine for now'''
    r = np.atleast_2d(r_ecef)
    lat = np.degrees(np.arctan2(r[:, 2], np.hypot(r[:, 0], r[:, 1]))) #arctan2 used to handle the poles where the distance is 0 without dividing by 0
    lon = np.degrees(np.arctan2(r[:, 1], r[:, 0])) #arctan2 used to consider the signs of both x and y to work out which quadrant the point is in
    return lat, lon

def latlon_to_ecef(lat_deg: float, lon_deg: float, radius: float) -> np.ndarray:
    '''ECEF position of a point on a sphere of given radius.'''
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return radius * np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
            #spherical to cartesian: x = cos(lat)*cos(lon); y = cos(lat)*sin(lon); z = sin(lat)


def ecef_to_geodetic(r_ecef: np.ndarray, a: float = R_EARTH, f: float = F_EARTH
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Geodetic latitude (deg), longitude (deg), and height above the WGS-84 ellipsoid (m).
    Geodetic latitude is the angle of the ellipsoid's surface normal (map latitude).
    Converges to under 1 mm for anything near Earth.
    Returns arrays of length N for (N, 3) input, length 1 for (3,) input.'''
    r = np.atleast_2d(r_ecef)
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    e2 = f * (2.0 - f) #eccentricity squared of the ellipsoid
    p = np.hypot(x, y) #distance from the spin axis
    lat = np.arctan2(z, p * (1.0 - e2))
    for _ in range(5):
        sin_lat = np.sin(lat)
        n = a / np.sqrt(1.0 - e2 * sin_lat**2) #radius of curvature in vertical
        lat = np.arctan2(z + e2 * n * sin_lat, p)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    h = p * cos_lat + z * sin_lat - a * np.sqrt(1.0 - e2 * sin_lat**2) #height that stays fine at the poles
    lon = np.degrees(np.arctan2(y, x))
    return np.degrees(lat), lon, h