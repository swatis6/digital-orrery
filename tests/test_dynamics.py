'''Tests for dynamics.py and spacecraft.py: gravity, J2, atmosphere, drag, force model.'''

import numpy as np
from numpy.testing import assert_allclose

from orrery.constants import J2_EARTH, MU_EARTH, OMEGA_EARTH, R_EARTH
from orrery.dynamics import (ExponentialAtmosphere, ForceModel, drag_accel,
                             j2_gravity, point_mass_gravity)
from orrery.spacecraft import Spacecraft

R_LEO = R_EARTH + 400e3


# Gravity --------------

def test_point_mass_magnitude_and_direction():
    r = np.array([R_LEO, 0.0, 0.0])
    a = point_mass_gravity(r, MU_EARTH)
    assert_allclose(np.linalg.norm(a), MU_EARTH / R_LEO**2, rtol=1e-12)
    assert_allclose(a / np.linalg.norm(a), -r / R_LEO, atol=1e-15)


def test_j2_at_equator_pulls_inward():
    '''On the equator J2 adds extra inward pull of 1.5*J2*mu*R^2/r^4.'''
    a = j2_gravity(np.array([R_LEO, 0.0, 0.0]), MU_EARTH, R_EARTH, J2_EARTH)
    expected = -1.5 * J2_EARTH * MU_EARTH * R_EARTH**2 / R_LEO**4
    assert_allclose(a, [expected, 0.0, 0.0], atol=1e-15)


def test_j2_at_pole_pushes_outward():
    '''Over the pole J2 reduces gravity: +3*J2*mu*R^2/r^4 along +z.'''
    a = j2_gravity(np.array([0.0, 0.0, R_LEO]), MU_EARTH, R_EARTH, J2_EARTH)
    expected = 3.0 * J2_EARTH * MU_EARTH * R_EARTH**2 / R_LEO**4
    assert_allclose(a, [0.0, 0.0, expected], atol=1e-15)


def test_j2_is_small_compared_to_point_mass():
    r = np.array([R_LEO, 0.0, 0.0])
    ratio = np.linalg.norm(j2_gravity(r, MU_EARTH, R_EARTH, J2_EARTH)) / np.linalg.norm(
        point_mass_gravity(r, MU_EARTH))
    assert 1e-4 < ratio < 2e-3


# Atmosphere and drag --------------

def test_density_at_sea_level_and_one_scale_height():
    atm = ExponentialAtmosphere()
    assert_allclose(atm.density(np.array([R_EARTH, 0.0, 0.0])), atm.rho0)
    r_h = np.array([R_EARTH + atm.scale_height, 0.0, 0.0])
    assert_allclose(atm.density(r_h), atm.rho0 / np.e, rtol=1e-12)


def test_density_clamped_below_surface():
    atm = ExponentialAtmosphere()
    assert atm.density(np.array([R_EARTH - 1000.0, 0.0, 0.0])) == atm.rho0


def test_density_decreases_with_altitude():
    atm = ExponentialAtmosphere()
    alts = [100e3, 200e3, 300e3]
    rhos = [atm.density(np.array([R_EARTH + h, 0.0, 0.0])) for h in alts]
    assert rhos[0] > rhos[1] > rhos[2] > 0.0


def test_drag_opposes_velocity_relative_to_atmosphere():
    r = np.array([R_LEO, 0.0, 0.0])
    v = np.array([0.0, 7600.0, 0.0])
    a = drag_accel(r, v, density=1e-11, ballistic_coeff=60.0, omega_body=OMEGA_EARTH)
    v_rel = v - np.cross([0.0, 0.0, OMEGA_EARTH], r)
    assert np.dot(a, v_rel) < 0.0
    assert_allclose(np.cross(a, v_rel), 0.0, atol=1e-20)


def test_no_drag_when_moving_with_atmosphere():
    r = np.array([R_LEO, 0.0, 0.0])
    v = np.cross([0.0, 0.0, OMEGA_EARTH], r)
    a = drag_accel(r, v, density=1e-11, ballistic_coeff=60.0, omega_body=OMEGA_EARTH)
    assert_allclose(a, 0.0, atol=1e-20)


def test_ballistic_coefficient():
    sc = Spacecraft(mass=4.0, cd=2.2, area=0.03)
    assert_allclose(sc.ballistic_coefficient, 4.0 / (2.2 * 0.03))


# Force model --------------

def test_force_model_without_j2_or_drag_is_point_mass():
    r = np.array([R_LEO, 1000.0, 2000.0])
    v = np.array([0.0, 7600.0, 0.0])
    model = ForceModel(use_j2=False)
    assert_allclose(model.acceleration(0.0, r, v), point_mass_gravity(r, MU_EARTH))


def test_force_model_sums_all_terms():
    r = np.array([R_LEO, 1000.0, 2000.0])
    v = np.array([0.0, 7600.0, 100.0])
    atm, sc = ExponentialAtmosphere(), Spacecraft()
    model = ForceModel(use_j2=True, atmosphere=atm, spacecraft=sc)
    expected = (point_mass_gravity(r, MU_EARTH)
                + j2_gravity(r, MU_EARTH, R_EARTH, J2_EARTH)
                + drag_accel(r, v, atm.density(r), sc.ballistic_coefficient, OMEGA_EARTH))
    assert_allclose(model.acceleration(0.0, r, v), expected, rtol=1e-14)


def test_force_model_call_returns_velocity_then_acceleration():
    r = np.array([R_LEO, 0.0, 0.0])
    v = np.array([0.0, 7600.0, 0.0])
    model = ForceModel(use_j2=False)
    dydt = model(0.0, np.concatenate((r, v)))
    assert dydt.shape == (6,)
    assert_allclose(dydt[:3], v)
    assert_allclose(dydt[3:], model.acceleration(0.0, r, v))


def test_force_model_passes_time_to_atmosphere():
    class RecordingAtmosphere:
        def __init__(self):
            self.times = []

        def density(self, r, t=0.0):
            self.times.append(t)
            return 1e-12

    atm = RecordingAtmosphere()
    ForceModel(atmosphere=atm, spacecraft=Spacecraft())(4321.0, np.array([R_LEO, 0, 0, 0, 7600.0, 0]))
    assert atm.times == [4321.0]
