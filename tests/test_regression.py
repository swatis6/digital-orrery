'''Regression baselines: re-entry times of the two current examples.

These numbers record what the code does TODAY, not what is physically correct.
They exist so refactors (Phase 1, the simulator) can't change results by accident.
When you change the physics on purpose (Phase 2, new atmosphere), these will fail:
re-run, check the new numbers make sense, and update the baselines.
'''

from datetime import datetime, timezone

import numpy as np
from numpy.testing import assert_allclose

from orrery.constants import DEG2RAD, R_EARTH, SECONDS_PER_DAY
from orrery.dynamics import ExponentialAtmosphere, ForceModel
from orrery.orbits import circular_velocity
from orrery.propagate import altitude_event, propagate
from orrery.spacecraft import Spacecraft

EPOCH = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)

# Baselines (seconds from start to 100 km), recorded 2026-10-21
LEO_DECAY_REENTRY_S = 41819.90 # examples/leo_decay.py (0.484 days)
INCLINED_DECAY_REENTRY_S = 62941.52 # examples/inclined_decay.py (0.728 days)


def _reentry_time(inc_deg: float) -> float:
    sc = Spacecraft(name="TesterSat", mass=4.0, cd=2.2, area=0.03)
    r0 = R_EARTH + 200e3
    vc = circular_velocity(r0)
    i = inc_deg * DEG2RAD
    y0 = np.array([r0, 0.0, 0.0, 0.0, vc * np.cos(i), vc * np.sin(i)])
    model = ForceModel(use_j2=True, atmosphere=ExponentialAtmosphere(), spacecraft=sc)
    traj = propagate(model, y0, (0.0, 60 * SECONDS_PER_DAY), epoch=EPOCH,
                     events=[altitude_event(100e3)])
    assert traj.stopped_by_event
    return traj.t[-1]


def test_leo_decay_baseline():
    assert_allclose(_reentry_time(0.0), LEO_DECAY_REENTRY_S, rtol=1e-5)


def test_inclined_decay_baseline():
    assert_allclose(_reentry_time(51.6), INCLINED_DECAY_REENTRY_S, rtol=1e-5)
