'''Dynamics module for the digital orrery project. Contains functions and classes for calculating gravitational 
and atmospheric forces on the spacecraft.

Includes: point mass gravity, j2 gravity, exponential atmosphere model, NRLMSIS atmosphere model,
drag acceleration, and a force model class that combines all forces.

Atmosphere models all have a density(r, t) method: r is the ECI position (m) and t is seconds since
the run's epoch. Models that don't depend on time ignore t.'''

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy as np

from orrery.constants import J2_EARTH, MU_EARTH, OMEGA_EARTH, R_EARTH
from orrery.frames import ecef_to_geodetic, eci_to_ecef
from orrery.spacecraft import Spacecraft

#Gravity --------------

def point_mass_gravity(r: np.ndarray, mu: float) -> np.ndarray:
    '''a = -mu*r / |r|^3 ; two body accel returned as a matrix'''
    r_norm = np.linalg.norm(r)
    return -mu*r/r_norm**3

def j2_gravity(r: np.ndarray, mu: float, radius: float, j2: float) -> np.ndarray:
    '''take z2=5*z/|r|^2 
    --> a = (-3/2*j2*mu*r / |r|^3)([x*(1-z2), y*(1-z2), z*(3-z2)]) ;
    j2 perterbating acceleration returned as a matrix
    assumes frame's z-axis is bodys spin axis'''
    x, y, z = r
    r_norm = np.linalg.norm(r)
    factor = -1.5*j2*mu*radius**2/r_norm**5
    z2 = 5.0*(z/r_norm)**2
    return factor*np.array([x*(1.0-z2), y*(1.0-z2), z*(3.0-z2)])

#Atmosphere/Drag --------------
@dataclass
class ExponentialAtmosphere:
    '''initial exponential atmosphere model carried over from
    prior project. inaccurate, and will be improved soon.'''
    rho0: float = 1.225 #kg/m^3, atm density at sea level
    scale_height: float = 8500.0
    body_rad: float = R_EARTH

    def density(self, r:np.ndarray, t:float = 0.0) -> float:
        '''returns density at a given position vector of r (t is accepted but not used)'''
        r_norm = np.linalg.norm(r)
        altitude = max(r_norm - self.body_rad, 0.0) #clamps when below surface/sea lvl
        return self.rho0*np.exp(-altitude/self.scale_height)

@dataclass
class NRLMSISAtmosphere:
    '''NRLMSIS 2.1 empirical atmosphere through pymsis package. Density depends on altitude,
    latitude, longitude, time of day, season, solar activity (F10.7), and geomagnetic activity (Ap).

    epoch: UTC datetime at t=0, must match the epoch passed to propagate()
    f107: daily 10.7 cm solar radio flux for the previous day (solar flux units)
    f107a: 81-day average of F10.7
    ap: daily geomagnetic Ap index
    ~F10.7 ~70 at solar minimum, ~150 moderate, ~200+ at solar maximum; Ap ~4 quiet, 15+ active.

    Set f107, f107a and ap all to None to use the observed/predicted values for each date instead.
    pymsis downloads those from CelesTrak and dates with no data raise an error.'''
    epoch: datetime
    f107: float | None = 150.0
    f107a: float | None = 150.0
    ap: float | None = 15.0
    _epoch64: np.datetime64 = field(init=False, repr=False)

    def __post_init__(self):
        import pymsis #imported here so everything else works even if pymsis isn't installed
        self._pymsis = pymsis
        values = (self.f107, self.f107a, self.ap)
        if any(v is None for v in values) and not all(v is None for v in values):
            raise ValueError("set f107, f107a and ap all to numbers, or all to None")
        epoch = self.epoch if self.epoch.tzinfo is not None else self.epoch.replace(tzinfo=timezone.utc)
        self._epoch64 = np.datetime64(epoch.astimezone(timezone.utc).replace(tzinfo=None), "us")

    def density(self, r: np.ndarray, t: float = 0.0) -> float:
        '''returns density (kg/m^3) at ECI position r, t seconds after epoch'''
        r_ecef = eci_to_ecef(r, self.epoch, t) #msis needs earth-fixed location
        lat, lon, h = ecef_to_geodetic(r_ecef) #msis wants geodetic lat and height above the ellipsoid
        alt_km = max(h[0], 0.0) / 1e3 #clamps below surface
        date = self._epoch64 + np.timedelta64(int(round(t * 1e6)), "us")
        if self.f107 is None:
            out = self._pymsis.calculate(date, lon[0], lat[0], alt_km)
        else:
            out = self._pymsis.calculate(date, lon[0], lat[0], alt_km,
                                         self.f107, self.f107a, [[self.ap] * 7])
        rho = float(np.ravel(out)[0]) #first output variable is total mass density in kg/m^3
        if not np.isfinite(rho):
            raise ValueError(f"NRLMSIS returned no density at t={t:.1f} s (missing space weather data?)")
        return rho

def drag_accel (r:np.ndarray, v:np.ndarray, density:float, ballistic_coeff:float, omega_body:float)-> np.ndarray:
    '''a = -.5*tho*|v_relative_to_atm|*v_relative_to_atm / BC
    v_rel = v - omega x r; velocity relative to atmosphere that co-rotates w/ central body'''
    omega_vec = np.array([0.0, 0.0, omega_body])
    v_rel = v - np.cross(omega_vec, r)
    v_rel_norm = np.linalg.norm(v_rel)
    return -0.5 * density * v_rel_norm * v_rel / ballistic_coeff

#Force model --------------
@dataclass
class ForceModel:
    mu: float = MU_EARTH
    radius: float = R_EARTH
    j2: float = J2_EARTH
    omega_body: float = OMEGA_EARTH
    use_j2: bool = True
    atmosphere: ExponentialAtmosphere | NRLMSISAtmosphere | None = None
    spacecraft: Spacecraft | None = None

    def acceleration(self, t:float, r:np.ndarray, v:np.ndarray) -> np.ndarray:
        '''a = gravity + j2 + drag ;combines all sources of acceleration into a single vector.'''
        a= point_mass_gravity(r, self.mu)
        if self.use_j2:
            a=a+j2_gravity(r, self.mu, self.radius, self.j2)
        if self.atmosphere is not None:
            rho = self.atmosphere.density(r, t)
            a=a+drag_accel(r, v, rho, self.spacecraft.ballistic_coefficient, self.omega_body)
        return a

    def __call__(self, t:float, y:np.ndarray) -> np.ndarray:
        '''call like a function to return dydt = [v, a] for ODE integration.'''
        r, v = y[:3], y[3:] #state vector is sliced into [x, y, z] [vx, vy, vz]
        return np.concatenate((v, self.acceleration(t, r, v))) #returns dydt = [vx, vy, vz, ax, ay, az]