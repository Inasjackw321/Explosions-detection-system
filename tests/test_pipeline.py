import pytest

from gulfseis import synthetic
from gulfseis.geo import haversine_km
from gulfseis.pipeline import analyze
from gulfseis.synthetic import ScenarioEvent


@pytest.fixture(scope="module")
def default_run():
    _, wfs, cat, truth = synthetic.generate(seed=7)
    return analyze(wfs, catalog=cat), truth


def test_default_scenario_finds_all_three(default_run):
    res, truth = default_run
    labels = [e.label for e in res.events]
    assert labels == ["Likely explosion", "Likely earthquake", "Distant earthquake"]


def test_locations_and_magnitudes(default_run):
    res, truth = default_run
    for ev, t in zip(res.events[:2], truth[:2]):
        assert haversine_km(ev.lat, ev.lon, t.lat, t.lon) < 15
        assert ev.ml == pytest.approx(t.magnitude, abs=0.3)
    assert res.events[0].depth <= 3
    assert res.events[1].depth >= 5


def test_distant_event_direction(default_run):
    res, _ = default_run
    d = res.events[2]
    assert d.kind == "distant"
    assert 40 < d.back_azimuth < 70           # Hindu Kush is NE of the Gulf
    assert d.catalog_match is not None


def test_explosion_has_air_wave(default_run):
    res, _ = default_run
    assert any(q.channel == "HDF" for q in res.events[0].air_picks)
    assert not res.events[1].air_picks


def test_noise_only_gives_no_events():
    _, wfs, cat, _ = synthetic.generate([], seed=3)
    assert analyze(wfs, catalog=cat).events == []


@pytest.mark.parametrize("seed", [0, 1])
def test_custom_explosion_and_earthquake(seed):
    evs = [ScenarioEvent("explosion", 100, 24.0, 56.5, 0, 2.8),
           ScenarioEvent("earthquake", 700, 27.3, 56.0, 18, 4.0, strike=70)]
    _, wfs, cat, _ = synthetic.generate(evs, seed=seed)
    res = analyze(wfs, catalog=cat)
    assert [e.label for e in res.events] == ["Likely explosion", "Likely earthquake"]
