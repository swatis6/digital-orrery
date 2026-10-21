'''Orbital mechanics helpers that accept a single state or array of states'''

from __future__ import annotations
import numpy as np
from orrery.constants import MU_EARTH

def circular_velocity(r_mag: float, mu: float = MU_EARTH) -> float:
    '''speed of circular orbit at radius r_mag in m/s'''
    return np.sqrt(mu/r_mag)

def orbital_period(a:float, mu: float = MU_EARTH) -> float:
    '''period of an orbit with a semi-major axis of a in seconds'''
    return 2.0*np.pi*np.sqrt(a**3/mu)

def specific_energy(r: np.ndarray, v:np.ndarray, mu: float = MU_EARTH) -> np.ndarray:
    '''two body specific orbital energy
    v^2/2 - mu/r, in J/kg.
    With shape (3,), it returns a single number, with shape (N, 3), it computes 
    the length of each row and returns N distances.'''
    r_norm = np.linalg.norm(r, axis=-1) #compute along the last axis
    v_sq = np.sum(np.asarray(v)**2, axis = -1)
    return 0.5 * v_sq - mu / r_norm

def specific_angular_moment(r: np.ndarray, v:np.ndarray) -> np.ndarray:
    '''h = r x v in m^2/s'''
    return np.cross (r, v)