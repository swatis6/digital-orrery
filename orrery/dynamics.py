'''Dynamics module for the digital orrery project. Contains functions and classes for calculating gravitational 
and atmospheric forces on the spacecraft.

Includes: point mass gravity, j2 gravity, exponential atmosphere model, drag acceleration, and a force model 
class that combines all forces.'''

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from orrery.constants import J2_EARTH, MU_EARTH, OMEGA_EARTH, R_EARTH
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

    def density(self, r:np.ndarray) -> float:
        '''returns density at a given position vector of r'''
        r_norm = np.linalg.norm(r)
        altitude = max(r_norm - self.body_rad, 0.0) #clamps when below surface/sea lvl
        return self.rho0*np.exp(-altitude/self.scale_height)

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
    atmosphere: ExponentialAtmosphere | None = None
    spacecraft: Spacecraft | None = None

    def acceleration(self, t:float, r:np.ndarray, v:np.ndarray) -> np.ndarray:
        '''a = gravity + j2 + drag ;combines all sources of acceleration into a single vector.'''
        a= point_mass_gravity(r, self.mu)
        if self.use_j2:
            a=a+j2_gravity(r, self.mu, self.radius, self.j2)
        if self.atmosphere is not None:
            rho = self.atmosphere.density(r)
            a=a+drag_accel(r, v, rho, self.spacecraft.ballistic_coefficient, self.omega_body)
        return a

    def __call__(self, t:float, y:np.ndarray) -> np.ndarray:
        '''call like a function to return dydt = [v, a] for ODE integration.'''
        r, v = y[:3], y[3:] #state vector is sliced into [x, y, z] [vx, vy, vz]
        return np.concatenate((v, self.acceleration(t, r, v))) #returns dydt = [vx, vy, vz, ax, ay, az]