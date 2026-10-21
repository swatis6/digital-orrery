'''NRLMSIS tests. Skips automatically if pymsis isn't installed.'''

from datetime import datetime, timezone

import numpy as np
import pytest
from numpy.testing import assert_allclose

pymsis = pytest.importorskip("pymsis")

from orrery.constants import R_EARTH
from orrery.dynamics import NRLMSISAtmosphere
from orrery.frames import ecef_to_geodetic, eci_to_ecef
from orrery.sun import sun_position_eci

EPOCH = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)


def test_matches_direct_pymsis_call():
    '''ECI -> geodetic, epoch + t -> date, and units.'''
    r, t = np.array([4000e3, -3000e3, 4500e3]), 3600.0
    lat, lon, h = ecef_to_geodetic(eci_to_ecef(r, EPOCH, t))
    expected = pymsis.calculate(np.datetime64("2026-10-15T13:00:00"), lon[0], lat[0],
                                h[0] / 1e3, 120.0, 110.0, [[8.0] * 7])
    atm = NRLMSISAtmosphere(epoch=EPOCH, f107=120.0, f107a=110.0, ap=8.0)
    assert_allclose(atm.density(r, t), np.ravel(expected)[0], rtol=1e-10)


def test_dayside_denser_than_nightside():
    '''If longitude or time was handled wrong, day and night would get mixed up.'''
    s_hat = sun_position_eci(EPOCH) / np.linalg.norm(sun_position_eci(EPOCH))
    r_day = (R_EARTH + 416e3) * s_hat
    atm = NRLMSISAtmosphere(epoch=EPOCH)
    assert atm.density(r_day) > 1.5 * atm.density(-r_day)