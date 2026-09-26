"""Region polygon + the real-data path (FDSN download -> response removal ->
chunked monitoring) exercised against a fake FDSN server."""

import numpy as np
import pytest

import fake_fdsn
import sim
from gulfseis import data_sources, monitor
from gulfseis import region as reg
from gulfseis.config import Settings
from gulfseis.geo import haversine_km

E = sim.ScenarioEvent


def test_polygon():
    assert reg.contains(29.37, 47.98)          # Kuwait City
    assert reg.contains(26.5, 52.0)            # middle of the Gulf
    assert reg.contains(25.0, 57.3)            # Gulf of Oman off Fujairah
    assert not reg.contains(29.59, 52.58)      # Shiraz (inland)
    assert not reg.contains(24.71, 46.68)      # Riyadh
    assert reg.distance_km(29.59, 52.58) == pytest.approx(116, abs=15)
    assert 250_000 < reg.area_km2() < 450_000


@pytest.fixture(scope="module")
def fake_server():
    evs = [E("explosion", 1500, 29.25, 47.95, 0, 2.4), E("explosion", 4300, 25.5, 51.3, 0, 3.0),
           E("earthquake", 6000, 27.6, 52.3, 10, 3.3, strike=40)]
    _, wfs, _, _ = sim.generate(evs, duration=7800, seed=1)
    client = fake_fdsn.FakeClient(wfs)
    old = data_sources._client
    data_sources._client = lambda key, timeout=120: client
    yield evs, client
    data_sources._client = old


def test_station_search_keeps_area_stations(fake_server):
    s = Settings()
    f = data_sources.DataFetcher(["Raspberry Shake (AM citizen network)"], s, log=lambda *a: None)
    stations = f.discover(0, 1)
    ids = {x.id for x in stations}
    assert "XX.KUWT" in ids                    # Kuwait must never be dropped
    assert "XX.RIYD" not in ids                # Riyadh is > 150 km outside the area
    kuwt = next(x for x in stations if x.id == "XX.KUWT")
    assert kuwt.infrasound_channel == "HDF" and kuwt.kind == "Raspberry Shake & Boom"


def test_download_converts_to_physical_units(fake_server):
    s = Settings()
    f = data_sources.DataFetcher(["Raspberry Shake (AM citizen network)"], s, log=lambda *a: None)
    f.discover(0, 1)
    t0 = sim.default_start()
    wfs = f.fetch(t0 + 100, t0 + 400)
    seis = [w for w in wfs if w.units == "m/s"]
    assert seis and all(w.fs <= 50.0 for w in wfs)
    assert all(np.std(w.data) < 1e-4 for w in seis)       # m/s, not counts
    assert all(f.status[w.station.id].ok for w in wfs)


def test_chunked_monitor_finds_all_events(fake_server):
    evs, _ = fake_server
    s = Settings()
    f = data_sources.DataFetcher(["Raspberry Shake (AM citizen network)"], s, log=lambda *a: None)
    f.discover(0, 1)
    t0 = sim.default_start()
    r = monitor.run(f, t0 + 60, t0 + 7200, s, [], keep_full_below_s=0)   # force 1-hour chunks
    assert len(r.events) == 3
    for ev, truth in zip(r.events, evs):
        assert ev.in_region
        assert abs(ev.origin_time - (t0 + truth.t)) < 10
        assert haversine_km(ev.lat, ev.lon, truth.lat, truth.lon) < 30
    assert [("explosion" in e.label.lower()) for e in r.events] == [True, True, False]
    assert set(r.snippets) == {e.id for e in r.events}
