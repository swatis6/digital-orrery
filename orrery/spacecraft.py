'''Lists all spacecraft constants and params. Later, adjustable parameters will be
moved to a config file.'''

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass #nicer than having to define self.property over and over; generates __init__, __repr__, and __eq__
class Spacecraft:
    name: str = 'satellite'
    mass: float = 4.0 # kg
    cd: float = 2.2 # drag coeff
    area: float = 0.03 # m^2 of ram facing area
    inertia: np.ndarray | None = None #sets up 3x3 moi in kg m^2

    def __post_init__(self):
        '''checks the inertia tensor is physically possible'''
        if self.inertia is None:
            return
        self.inertia = np.asarray(self.inertia, dtype=float)
        if self.inertia.shape != (3, 3):
            raise ValueError("inertia must be a 3x3 matrix in kg m^2")
        if not np.allclose(self.inertia, self.inertia.T):
            raise ValueError("inertia must be symmetric (Ixy = Iyx, Ixz = Izx, Iyz = Izy)")
        principal = np.linalg.eigvalsh(self.inertia) #principal moments, smallest first
        if principal[0] <= 0.0 or principal[0] + principal[1] < principal[2] * (1 - 1e-9):
            raise ValueError(f"inertia isn't physically possible (principal moments {principal}); "
                             "check signs of the off-diagonal terms")

    @property 
    def ballistic_coefficient(self) -> float: #return a float value.
        return self.mass/(self.cd*self.area) #BC=m/(Cd*A) in kg/m^2.