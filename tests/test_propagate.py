'''Tests for propagate.py and orbits.py: integrators, events, Trajectory, orbit helpers.'''

import numpy as np
import pytest
from numpy.testing import assert_allclose

from orrery.constants import DEG2RAD, J2_EARTH, MU_EARTH, R_EARTH, SECONDS_PER_DAY
from orrery.dynamics import ForceModel
from orrery.orbits import (circular_velocity, orbital_period, specific_angular_moment,
                           specific_energy)
from orrery.propagate import altitude_event, propagate

R0 = R_EARTH + 400e3


def circular_state(r: float, inc_deg: float = 0.0) -> np.ndarray:
    vc = circular_velocity(r)
    i = inc_deg * DEG2RAD
    return np.array([r, 0.0, 0.0, 0.0, vc * np.cos(i), vc * np.sin(i)])


def eccentric_state(r_perigee: float, e: float) -> np.ndarray:
    '''Start at perigee of an orbit with eccentricity e.'''
    v_p = np.sqrt(MU_EARTH * (1 + e) / r_perigee)
    return np.array([r_perigee, 0.0, 0.0, 0.0, v_p, 0.0])


# Orbit helpers --------------

def test_geostationary_period_is_one_sidereal_day():
    a_geo = 42164.17e3
    assert_allclose(orbital_period(a_geo), 86164.09, rtol=1e-5)


def test_circular_orbit_energy_is_minus_mu_over_2a():
    y = circular_state(R0)
    assert_allclose(specific_energy(y[:3], y[3:]), -MU_EARTH / (2 * R0), rtol=1e-12)


def test_orbit_helpers_accept_arrays():
    y = np.vstack([circular_state(R0), circular_state(R0 + 100e3)])
    assert specific_energy(y[:, :3], y[:, 3:]).shape == (2,)
    assert specific_angular_moment(y[:, :3], y[:, 3:]).shape == (2, 3)


# Two-body propagation --------------

def test_adaptive_circular_orbit_returns_to_start():
    y0 = circular_state(R0, 30.0)
    traj = propagate(ForceModel(use_j2=False), y0, (0.0, orbital_period(R0)))
    assert np.linalg.norm(traj.y[-1, :3] - y0[:3]) < 1.0 


def test_adaptive_eccentric_orbit_conserves_energy_and_momentum():
    y0 = eccentric_state(R0, 0.3)
    a = R0 / (1 - 0.3)
    traj = propagate(ForceModel(use_j2=False), y0, (0.0, 5 * orbital_period(a)))
    energy = specific_energy(traj.r, traj.v)
    h = np.linalg.norm(specific_angular_moment(traj.r, traj.v), axis=1)
    assert_allclose(energy, energy[0], rtol=1e-8)
    assert_allclose(h, h[0], rtol=1e-8)


def test_rk4_matches_adaptive():
    y0 = circular_state(R0, 51.6)
    t_end = orbital_period(R0)
    model = ForceModel(use_j2=True)
    adaptive = propagate(model, y0, (0.0, t_end))
    rk4 = propagate(model, y0, (0.0, t_end), method="RK4", dt=10.0)
    assert_allclose(rk4.t[-1], t_end)
    assert np.linalg.norm(rk4.y[-1, :3] - adaptive.y[-1, :3]) < 1.0  # meters


def test_rk4_without_dt_raises():
    with pytest.raises(ValueError):
        propagate(ForceModel(), circular_state(R0), (0.0, 100.0), method="RK4")


# J2 --------------

def test_j2_conserves_polar_angular_momentum():
    '''J2 is symmetric about the z-axis, so h_z stays constant even though h tilts.'''
    y0 = circular_state(R0, 51.6)
    traj = propagate(ForceModel(use_j2=True), y0, (0.0, 0.5 * SECONDS_PER_DAY))
    hz = specific_angular_moment(traj.r, traj.v)[:, 2]
    assert_allclose(hz, hz[0], rtol=1e-9)


def test_j2_node_regression_matches_theory():
    '''RAAN drift over one day should match -1.5*n*J2*(R/a)^2*cos(i) within 1%.
    (The start isn't exactly a mean circular orbit, so it won't match perfectly.)'''
    inc = 51.6
    y0 = circular_state(R0, inc)
    traj = propagate(ForceModel(use_j2=True), y0, (0.0, SECONDS_PER_DAY)).resample(4000)
    h = specific_angular_moment(traj.r, traj.v)
    raan = np.unwrap(np.arctan2(h[:, 0], -h[:, 1]))
    measured = (raan[-1] - raan[0]) / SECONDS_PER_DAY
    theory = -1.5 * np.sqrt(MU_EARTH / R0**3) * J2_EARTH * (R_EARTH / R0)**2 * np.cos(inc * DEG2RAD)
    assert_allclose(measured, theory, rtol=0.01)


# Events and Trajectory --------------

def test_altitude_event_stops_adaptive_run_at_altitude():
    '''Start at 500 km slightly slower than circular, so the orbit dips below 300 km.
    The run should stop right at 300 km.'''
    y0 = np.array([R_EARTH + 500e3, 0.0, 0.0, 0.0, circular_velocity(R_EARTH + 500e3) * 0.97, 0.0])
    traj = propagate(ForceModel(use_j2=False), y0, (0.0, SECONDS_PER_DAY),
                     events=[altitude_event(300e3)])
    assert traj.stopped_by_event
    alt = np.linalg.norm(traj.y[-1, :3]) - R_EARTH
    assert_allclose(alt, 300e3, atol=1.0)


def test_altitude_event_stops_rk4_run_within_one_step():
    y0 = np.array([R_EARTH + 500e3, 0.0, 0.0, 0.0, circular_velocity(R_EARTH + 500e3) * 0.97, 0.0])
    traj = propagate(ForceModel(use_j2=False), y0, (0.0, SECONDS_PER_DAY),
                     method="RK4", dt=10.0, events=[altitude_event(300e3)])
    assert traj.stopped_by_event
    alt_end = np.linalg.norm(traj.y[-1, :3]) - R_EARTH
    alt_before = np.linalg.norm(traj.y[-2, :3]) - R_EARTH
    assert alt_end <= 300e3 < alt_before


def test_run_without_event_completes():
    traj = propagate(ForceModel(use_j2=False), circular_state(R0), (0.0, 1000.0),
                     events=[altitude_event(100e3)])
    assert not traj.stopped_by_event
    assert_allclose(traj.t[-1], 1000.0)


def test_resample_and_at():
    traj = propagate(ForceModel(use_j2=False), circular_state(R0), (0.0, 3000.0))
    s = traj.resample(101)
    assert s.t.shape == (101,) and s.y.shape == (101, 6)
    assert_allclose(s.y[0], traj.y[0], atol=1e-6)
    assert_allclose(s.y[-1], traj.y[-1], atol=1e-6)
    radii = np.linalg.norm(s.r, axis=1)
    assert_allclose(radii, R0, rtol=1e-8)


def test_rk4_trajectory_uses_linear_interpolation():
    traj = propagate(ForceModel(use_j2=False), circular_state(R0), (0.0, 100.0),
                     method="RK4", dt=10.0)
    assert traj.interpolant is None
    mid = traj.at(5.0)[0]
    assert_allclose(mid, 0.5 * (traj.y[0] + traj.y[1]))
