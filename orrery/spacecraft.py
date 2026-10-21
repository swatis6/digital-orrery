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

    @property 
    def ballistic_coefficient(self) -> float: #return a float value.
        return self.mass/(self.cd*self.area) #BC=m/(Cd*A) in kg/m^2.