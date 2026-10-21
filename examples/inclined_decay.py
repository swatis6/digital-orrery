'''Orbital decay of a 3U CubeSat from 200 km at 51.6 deg inclination, with J2 and drag.

To test more features
'''

from datetime import datetime, timezone
import matplotlib.pyplot as plt
import numpy as np

from orrery.visualization import basic_plots as viz
from orrery.constants import DEG2RAD, J2_EARTH, MU_EARTH, R_EARTH, RAD2DEG, SECONDS_PER_DAY
from orrery.dynamics import ExponentialAtmosphere, ForceModel
from orrery.orbits import circular_velocity, orbital_period
from orrery.propagate import altitude_event, propagate
from orrery.spacecraft import Spacecraft
from orrery.sun import in_eclipse, sun_position_eci

# Settings
EPOCH = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
START_ALTITUDE = 200e3 
INCLINATION_DEG = 51.6 
REENTRY_ALTITUDE = 100e3 
USE_J2 = True

def initial_state(altitude: float, inc_deg: float) -> np.ndarray:
    '''Circular orbit starting at the ascending node (RAAN = 0).'''
    r = R_EARTH + altitude
    vc = circular_velocity(r)
    i = inc_deg * DEG2RAD
    return np.array([r, 0.0, 0.0, 0.0, vc * np.cos(i), vc * np.sin(i)])


def raan_deg(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    '''Right ascension of the ascending node at each state, in degrees.
    The node vector is z x h = (-h_y, h_x, 0), where h = r x v.'''
    h = np.cross(r, v)
    return np.degrees(np.unwrap(np.arctan2(h[:, 0], -h[:, 1])))


def main():
    sc = Spacecraft(name="TesterSat", mass=4.0, cd=2.2, area=0.03)
    y0 = initial_state(START_ALTITUDE, INCLINATION_DEG)

    model = ForceModel(use_j2=USE_J2, atmosphere=ExponentialAtmosphere(), spacecraft=sc)
    traj = propagate(model, y0, (0.0, 60 * SECONDS_PER_DAY), epoch=EPOCH,
                     events=[altitude_event(REENTRY_ALTITUDE)])

    # Summary
    days = traj.t[-1] / SECONDS_PER_DAY
    period = orbital_period(R_EARTH + START_ALTITUDE)
    samples = traj.resample(5000)
    eclipse = in_eclipse(samples.r, sun_position_eci(EPOCH, samples.t)).mean()
    node_drift = raan_deg(samples.r, samples.v)
    node_rate = (node_drift[-1] - node_drift[0]) / days 
    a0 = R_EARTH + START_ALTITUDE 
    theory = (-1.5 * np.sqrt(MU_EARTH / a0**3) * J2_EARTH * (R_EARTH / a0)**2
              * np.cos(INCLINATION_DEG * DEG2RAD) * RAD2DEG * SECONDS_PER_DAY)

    if traj.stopped_by_event:
        print(f"{sc.name} reached {REENTRY_ALTITUDE / 1e3:.0f} km after {days:.2f} days "
              f"(about {traj.t[-1] / period:.0f} orbits).")
    else:
        print(f"{sc.name} still in orbit after {days:.0f} days.")
    print(f"Time in eclipse: {eclipse:.0%}")
    print(f"Orbit plane turned {node_drift[-1] - node_drift[0]:+.2f} deg "
          f"({node_rate:+.2f} deg/day; J2 theory at {START_ALTITUDE / 1e3:.0f} km: "
          f"{theory if USE_J2 else 0.0:+.2f} deg/day)")

    # Plots
    label = f"{START_ALTITUDE / 1e3:.0f} km, {INCLINATION_DEG}°" + (", J2" if USE_J2 else "")
    viz.plot_altitude(traj, reentry_altitude=REENTRY_ALTITUDE,
                      title=f"{sc.name} orbital decay · {label}")
    viz.plot_ground_track(traj, title=f"{sc.name} ground track until re-entry · {label}")
    viz.plot_orbit_3d(traj, show_hidden=True, title=f"{sc.name} decay · {label}")
    plt.show()


if __name__ == "__main__":
    main()