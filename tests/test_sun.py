'''Tests for sun.py: Sun position and cylindrical eclipse model.'''

from datetime import datetime, timezone

import numpy as np
from numpy.testing import assert_allclose

from orrery.constants import R_EARTH
from orrery.sun import AU, in_eclipse, sun_position_eci


def _declination_deg(r_sun):
    return np.degrees(np.arcsin(r_sun[2] / np.linalg.norm(r_sun)))


def test_sun_distance_is_about_one_au():
    r = sun_position_eci(datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc))
    assert 0.98 * AU < np.linalg.norm(r) < 1.02 * AU


def test_sun_closest_in_january_farthest_in_july():
    d_jan = np.linalg.norm(sun_position_eci(datetime(2026, 1, 3, tzinfo=timezone.utc)))
    d_jul = np.linalg.norm(sun_position_eci(datetime(2026, 7, 4, tzinfo=timezone.utc)))
    assert d_jan < d_jul


def test_declination_at_june_solstice():
    r = sun_position_eci(datetime(2026, 6, 21, 8, 0, tzinfo=timezone.utc))
    assert_allclose(_declination_deg(r), 23.44, atol=0.05)


def test_declination_near_zero_at_march_equinox():
    r = sun_position_eci(datetime(2026, 3, 20, 14, 46, tzinfo=timezone.utc))
    assert abs(_declination_deg(r)) < 0.05


def test_sun_position_array_shape():
    r = sun_position_eci(datetime(2026, 10, 15, tzinfo=timezone.utc), np.arange(10) * 3600.0)
    assert r.shape == (10, 3)


def test_eclipse_geometry():
    sun = np.array([AU, 0.0, 0.0])
    r = R_EARTH + 400e3
    sats = np.array([
        [-r, 0.0, 0.0], # directly behind Earth -> shadow
        [r, 0.0, 0.0], # between Earth and Sun -> lit
        [0.0, r, 0.0], # beside Earth -> lit
        [-r, 0.9 * R_EARTH, 0.0], # behind, inside the cylinder -> shadow
        [-r, 1.1 * R_EARTH, 0.0], # behind, outside the cylinder -> lit
    ])
    assert list(in_eclipse(sats, sun)) == [True, False, False, True, False]


def test_eclipse_fraction_of_noon_midnight_orbit():
    '''A circular orbit whose plane contains the Sun is in shadow for about
    (1/pi)*arcsin(R/r) of each orbit, which is about 38% at 400 km.'''
    r = R_EARTH + 400e3
    theta = np.linspace(0.0, 2 * np.pi, 20000, endpoint=False)
    sats = np.column_stack((r * np.cos(theta), r * np.sin(theta), np.zeros_like(theta)))
    fraction = in_eclipse(sats, np.array([AU, 0.0, 0.0])).mean()
    assert_allclose(fraction, np.arcsin(R_EARTH / r) / np.pi, atol=1e-3)
