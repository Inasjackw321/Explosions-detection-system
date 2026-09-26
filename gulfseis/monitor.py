"""Scan a long time period in overlapping chunks (download -> analyze -> keep).

Long periods (hours) are cut into chunks of `chunk_s` seconds.  Each chunk is
downloaded with extra data after it (`overlap_s`) so that slow air waves of an
explosion near the end of a chunk are still found, and extra data before it
(`pad_s`) so the STA/LTA has a noise history.  Only events whose origin time
falls inside the chunk proper are kept, so nothing is counted twice.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import physics
from .config import Settings
from .geo import haversine_km
from .models import Waveform
from .pipeline import AnalysisResult, StationTrace, analyze


@dataclass
class MonitorResult:
    events: list
    picks: list
    infra_picks: list
    settings: Settings
    t_start: float
    t_end: float
    catalog: list = field(default_factory=list)
    traces: dict | None = None            # full traces (short periods only)
    snippets: dict = field(default_factory=dict)   # event id -> {station id: StationTrace}
    overview: dict = field(default_factory=dict)   # station id -> (t0, dt, rms array)
    status: dict = field(default_factory=dict)     # station id -> StationStatus
    triggers_total: dict = field(default_factory=dict)
    messages: list = field(default_factory=list)
    focus: tuple | None = None            # (lat, lon, time) of an event the user asked to check

    @property
    def in_region(self):
        return [e for e in self.events if e.in_region]

    @property
    def outside(self):
        return [e for e in self.events if not e.in_region]

    @property
    def noise_picks(self):
        return [p for p in self.picks + self.infra_picks if p.event_id is None]

    def traces_for(self, ev=None) -> dict:
        if self.traces is not None:
            return self.traces
        return self.snippets.get(ev.id, {}) if ev is not None else {}


def _slice(w: Waveform, arr, t1, t2):
    i1 = max(w.index(t1), 0)
    i2 = min(w.index(t2), len(arr))
    return i1, (arr[i1:i2].astype(np.float32) if arr is not None and i2 > i1 else None)


def snip(st: StationTrace, t1, t2, t2_infra=None) -> StationTrace:
    """A light copy of a station's processed data for [t1, t2] (plotting only)."""
    out = StationTrace(st.station)
    if st.seismic is not None:
        w = st.seismic
        i1, f = _slice(w, st.filtered, t1, t2)
        _, c = _slice(w, st.cft, t1, t2)
        if f is not None:
            out.seismic = Waveform(w.station, w.channel, w.starttime + i1 / w.fs, w.fs, f, w.units)
            out.filtered, out.cft = f, c
    if st.infrasound is not None:
        w = st.infrasound
        i1, f = _slice(w, st.infra_filtered, t1, t2_infra or t2)
        _, c = _slice(w, st.infra_cft, t1, t2_infra or t2)
        if f is not None:
            out.infrasound = Waveform(w.station, w.channel, w.starttime + i1 / w.fs, w.fs, f, w.units)
            out.infra_filtered, out.infra_cft = f, c
    return out


def _overview(st: StationTrace, t1, t2, dt=10.0):
    """RMS ground motion in dt-second bins (for the timeline heat map)."""
    if st.seismic is None or st.filtered is None:
        return None
    w = st.seismic
    i1, i2 = max(w.index(t1), 0), min(w.index(t2), len(st.filtered))
    n = int(dt * w.fs)
    x = st.filtered[i1:i2]
    m = (len(x) // n) * n
    if m == 0:
        return None
    rms = np.sqrt(np.mean(x[:m].reshape(-1, n) ** 2, axis=1)).astype(np.float32)
    return (w.starttime + i1 / w.fs, dt, rms)


def run(fetcher, t_start, t_end, settings: Settings, catalog=None, progress=None,
        chunk_s=3600.0, overlap_s=1500.0, pad_s=180.0, keep_full_below_s=7200.0,
        max_snippets=60, focus=None) -> MonitorResult:
    catalog = catalog or []
    p, vm = settings.detection, settings.velocity
    total = t_end - t_start
    single = total <= keep_full_below_s
    if single:
        chunks = [(t_start, t_end)]
    else:
        edges = list(np.arange(t_start, t_end, chunk_s)) + [t_end]
        chunks = list(zip(edges[:-1], edges[1:]))
    out = MonitorResult([], [], [], settings, t_start, t_end, catalog, focus=focus)
    now = time.time() - 30
    n_ev = 0
    for k, (cs, ce) in enumerate(chunks):
        def prog(f, text, k=k):
            if progress:
                progress((k + f) / len(chunks), f"Part {k + 1}/{len(chunks)}: {text}")

        fetch_end = min(ce if single else ce + overlap_s, now)
        wfs = fetcher.fetch(cs - pad_s, fetch_end, progress=lambda f, t: prog(0.7 * f, t))
        if not wfs:
            out.messages.append(f"{_hm(cs)}-{_hm(ce)}: no data downloaded")
            continue
        res: AnalysisResult = analyze(wfs, settings, catalog, progress=lambda f, t: prog(0.7 + 0.3 * f, t))
        keep = [e for e in res.events if cs - (pad_s if k == 0 else 0) <= e.origin_time < ce]
        # renumber events so ids are unique over the whole period
        mapping = {}
        for e in keep:
            n_ev += 1
            mapping[e.id] = f"EV{n_ev:03d}"
            e.id = mapping[e.id]
        for q in res.picks + res.infra_picks:
            if q.event_id is not None and q.event_id not in mapping.values():
                q.event_id = mapping.get(q.event_id, "__other__")
        lo = cs - (pad_s if k == 0 else 0)
        ids = set(mapping.values())

        def mine(q):   # picks of events kept by a neighbouring part are stored there
            return q.event_id in ids or (q.event_id is None and lo <= q.time < ce)

        out.picks += [q for q in res.picks if mine(q)]
        out.infra_picks += [q for q in res.infra_picks if mine(q)]
        out.events += keep
        for sid, st in res.traces.items():
            out.triggers_total[sid] = out.triggers_total.get(sid, 0) + st.triggers_total + st.infra_triggers_total
            ov = _overview(st, cs, ce)
            if ov is not None:
                if sid in out.overview:
                    t0, dt, arr = out.overview[sid]
                    gap = int(round((ov[0] - (t0 + len(arr) * dt)) / dt))
                    arr = np.concatenate([arr, np.full(max(gap, 0), np.nan, np.float32), ov[2]])
                    out.overview[sid] = (t0, dt, arr)
                else:
                    out.overview[sid] = ov
        if single:
            out.traces = res.traces
        else:
            for e in keep:
                if len(out.snippets) >= max_snippets:
                    break
                far = max((float(haversine_km(e.lat, e.lon, st.station.lat, st.station.lon))
                           for st in res.traces.values()), default=100.0)
                t2 = e.origin_time + float(physics.s_time(far, e.depth, vm)) + 60.0
                t2i = e.origin_time + p.max_air_range_km / p.celerity_min + 30.0
                out.snippets[e.id] = {sid: snip(st, e.origin_time - 60.0, t2, t2i)
                                      for sid, st in res.traces.items()}
        out.messages += res.messages
    out.events.sort(key=lambda e: e.origin_time)
    out.status = dict(fetcher.status)
    return out


def _hm(t):
    return time.strftime("%H:%M", time.gmtime(t))
