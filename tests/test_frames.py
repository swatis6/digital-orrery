'''Tests for frames.py: Julian date, GMST, ECI/ECEF rotation, lat/lon.'''

from datetime import datetime, timedelta, timezone

import numpy as np
from numpy.testing import assert_allclose

from orrery.frames import (J2000, J2000_JD, ecef_to_eci, ecef_to_latlon, eci_to_ecef,
                           gmst, gmst_at, julian_date, latlon_to_ecef)

EPOCH = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
SIDEREAL_DAY = 86164.0905  # s


def test_julian_date_at_j2000():
    assert_allclose(julian_date(J2000), J2000_JD, atol=1e-9)


def test_naive_datetime_is_treated_as_utc():
    naive = datetime(2026, 10, 15, 12, 0)
    assert julian_date(naive) == julian_date(EPOCH)


def test_gmst_at_j2000():
    '''GMST at J2000 is 280.46061837 deg (IAU 1982).'''
    assert_allclose(np.degrees(gmst(J2000_JD)), 280.46061837, atol=1e-6)


def test_gmst_repeats_after_one_sidereal_day():
    diff = gmst_at(EPOCH, SIDEREAL_DAY) - gmst_at(EPOCH, 0.0)
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    assert abs(np.degrees(diff)) < 1e-3


def test_gmst_rate_matches_earth_rotation():
    from orrery.constants import OMEGA_EARTH
    rate = (gmst_at(EPOCH, 60.0) - gmst_at(EPOCH, 0.0)) / 60.0
    assert_allclose(rate, OMEGA_EARTH, rtol=1e-6)


def test_eci_ecef_round_trip_single_vector():
    r = np.array([7000e3, -1200e3, 3000e3])
    back = ecef_to_eci(eci_to_ecef(r, EPOCH, 1234.5), EPOCH, 1234.5)
    assert back.shape == (3,)
    assert_allclose(back, r, atol=1e-6)


def test_eci_ecef_round_trip_array():
    rng = np.random.default_rng(0)
    r = rng.normal(size=(50, 3)) * 7000e3
    t = np.linspace(0.0, 86400.0, 50)
    back = ecef_to_eci(eci_to_ecef(r, EPOCH, t), EPOCH, t)
    assert back.shape == (50, 3)
    assert_allclose(back, r, atol=1e-6)


def test_rotation_keeps_length_and_z():
    r = np.array([7000e3, -1200e3, 3000e3])
    r_ecef = eci_to_ecef(r, EPOCH, 500.0)
    assert_allclose(np.linalg.norm(r_ecef), np.linalg.norm(r), rtol=1e-14)
    assert r_ecef[2] == r[2]


def test_eci_x_axis_lands_at_minus_gmst_longitude():
    '''A point on the ECI x-axis (vernal equinox) sits at longitude -GMST in ECEF.'''
    r_ecef = eci_to_ecef(np.array([7000e3, 0.0, 0.0]), EPOCH, 0.0)
    _, lon = ecef_to_latlon(r_ecef)
    expected = (-np.degrees(gmst_at(EPOCH, 0.0)) + 180.0) % 360.0 - 180.0
    assert_allclose(lon[0], expected, atol=1e-9)


def test_latlon_round_trip():
    for lat, lon in [(0.0, 0.0), (37.94, -75.47), (-51.6, 170.0), (89.0, -179.0)]:
        lat_out, lon_out = ecef_to_latlon(latlon_to_ecef(lat, lon, 6378137.0))
        assert_allclose(lat_out[0], lat, atol=1e-9)
        assert_allclose(lon_out[0], lon, atol=1e-9)


def test_north_pole_latitude():
    lat, _ = ecef_to_latlon(np.array([0.0, 0.0, 6378137.0]))
    assert_allclose(lat[0], 90.0)
