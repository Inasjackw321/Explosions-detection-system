"""Pipeline tests on simulated recordings (tests/sim.py - test-only data)."""

import pytest

import sim
from gulfseis.geo import haversine_km
from gulfseis.pipeline import analyze

E = sim.ScenarioEvent


@pytest.fixture(scope="module")
def default_run():
    _, wfs, cat, truth = sim.generate(seed=7)
    return analyze(wfs, catalog=cat), truth


def test_default_scenario(default_run):
    res, truth = default_run
    labels = [(e.label, e.in_region) for e in res.events]
    # the explosion is on the coast (inside the area); the Zagros quake is inland
    # (outside the area); the Hindu Kush quake is far away
    assert labels == [("Likely explosion", True), ("Likely earthquake", False), ("Distant earthquake", False)]


def test_locations_and_magnitudes(default_run):
    res, truth = default_run
    for ev, t in zip(res.events[:2], truth[:2]):
        assert haversine_km(ev.lat, ev.lon, t.lat, t.lon) < 15
        assert ev.ml == pytest.approx(t.magnitude, abs=0.3)


def test_distant_event(default_run):
    d = default_run[0].events[2]
    assert d.kind == "distant" and 40 < d.back_azimuth < 70 and d.catalog_match is not None


def test_explosion_has_air_wave(default_run):
    res, _ = default_run
    assert any(q.channel == "HDF" for q in res.events[0].air_picks)
    assert not res.events[1].air_picks


def test_noise_only_gives_no_events():
    _, wfs, cat, _ = sim.generate([], seed=3)
    res = analyze(wfs, catalog=cat)
    assert res.events == []
    assert all(q.phase == "noise" for q in res.noise_picks)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_small_kuwait_blast_is_found(seed):
    """ML 2.2 near Kuwait City: too few stations for a full location, but the
    infrasound / few-station detectors must still flag it as an explosion."""
    _, wfs, cat, _ = sim.generate([E("explosion", 100, 29.2, 48.0, 0, 2.2)], seed=seed)
    res = analyze(wfs, catalog=cat)
    found = [e for e in res.events if e.in_region and "explosion" in e.label.lower()]
    assert found
    assert all("earthquake" not in e.label.lower() for e in res.events)


@pytest.mark.parametrize("seed", [0, 1])
def test_quake_outside_area_is_not_reported_inside(seed):
    _, wfs, cat, _ = sim.generate([E("earthquake", 100, 25.0, 56.0, 15, 3.0, strike=30)], seed=seed)
    res = analyze(wfs, catalog=cat)
    assert not [e for e in res.events if e.in_region]
