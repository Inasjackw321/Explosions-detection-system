import numpy as np
import pytest

from gulfseis import physics
from gulfseis.config import VelocityModel
from gulfseis.geo import azimuth_deg, destination, haversine_km


def test_haversine_one_degree_of_latitude():
    assert haversine_km(0, 0, 1, 0) == pytest.approx(111.19, abs=0.01)


def test_destination_roundtrip():
    lat, lon = destination(26.0, 52.0, 70.0, 500.0)
    assert haversine_km(26.0, 52.0, lat, lon) == pytest.approx(500.0, abs=0.1)
    assert azimuth_deg(26.0, 52.0, lat, lon) == pytest.approx(70.0, abs=0.1)


def test_richter_definition():
    # ML 3 at 100 km = 1 mm on a real (x2080) Wood-Anderson
    amp = physics.wa_amplitude_for_ml(3.0, 100.0)
    assert amp * physics.WA_HISTORICAL_GAIN == pytest.approx(1e6, rel=0.01)
    assert physics.local_magnitude(amp, 100.0) == pytest.approx(3.0)


def test_pn_overtakes_pg_at_regional_distance():
    vm = VelocityModel()
    assert physics.p_time(50, 0, vm) == pytest.approx(50 / vm.vp)
    assert physics.p_time(400, 0, vm) < 400 / vm.vp
    assert np.isinf(physics.head_wave_time(20, 0, vm.vp, vm.vpn, vm.moho))


def test_sp_rule():
    vm = VelocityModel()
    d = 200.0
    dt = float(physics.s_time(d, 0, vm) - physics.p_time(d, 0, vm))
    assert physics.sp_distance_km(dt, vm.vp, vm.vs) == pytest.approx(d, rel=0.01)


def test_beirut_calibration():
    # ML 3.3 surface blast -> several hundred tonnes TNT
    y = physics.explosion_yield_tons(3.3, physics.SURFACE_BLAST_COUPLING)
    assert 300 < y < 1200


def test_sound_speed():
    assert physics.sound_speed_km_s(20) == pytest.approx(0.343, abs=0.001)


def test_wood_anderson_gain_at_high_frequency():
    fs = 100.0
    t = np.arange(0, 20, 1 / fs)
    f = 10.0
    vel = 1e-6 * 2 * np.pi * f * np.cos(2 * np.pi * f * t)  # exact derivative of disp
    wa = physics.velocity_to_wood_anderson(vel, fs)
    assert np.max(np.abs(wa[500:-500])) == pytest.approx(1e-6, rel=0.05)
