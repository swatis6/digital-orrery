'''Orbital decay of a 3U CubeSat from 200 km'''

from datetime import datetime, timezone
import matplotlib.pyplot as plt
import numpy as np

from orrery.visualization import basic_plots as viz
from orrery.constants import R_EARTH, SECONDS_PER_DAY
from orrery.dynamics import ExponentialAtmosphere, ForceModel
from orrery.orbits import circular_velocity
from orrery.propagate import altitude_event, propagate
from orrery.spacecraft import Spacecraft

EPOCH = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)


def main():
    sc = Spacecraft(name="TesterSat", mass=4.0, cd=2.2, area=0.03)

    # Circular equatorial orbit at 200 km
    r0 = R_EARTH + 200e3
    y0 = np.array([r0, 0.0, 0.0, 0.0, circular_velocity(r0), 0.0])

    model = ForceModel(use_j2=True, atmosphere=ExponentialAtmosphere(), spacecraft=sc)
    traj = propagate(
        model, y0, (0.0, 60 * SECONDS_PER_DAY), epoch=EPOCH,
        events=[altitude_event(1e3)],   # stop at 100 km
    )

    if traj.stopped_by_event:
        print(f"{sc.name} reached 100 km after {traj.t[-1] / SECONDS_PER_DAY:.2f} days.")
    else:
        print(f"{sc.name} still in orbit after {traj.t[-1] / SECONDS_PER_DAY:.0f} days.")

    viz.plot_altitude(traj, reentry_altitude=100e3,
                      title=f"{sc.name} orbital decay (single-exponential atmosphere)")
    viz.plot_ground_track(traj, title=f"{sc.name} ground track until re-entry")
    viz.plot_orbit_3d(traj, title=f"{sc.name} decay")
    plt.show()


if __name__ == "__main__":
    main()
