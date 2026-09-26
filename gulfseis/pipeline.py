"""Full processing chain: detect -> associate -> locate -> size -> classify."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import discrimination as disc
from . import physics
from . import region as reg
from .config import Settings
from .detection import aic_pick, bandpass, detect_station, sta_lta
from .geo import KM_PER_DEG, haversine_km
from .location import associate_grid, locate, locate_distant, plane_wave_fit
from .models import CatalogEvent, DetectedEvent, Pick, Station, Waveform


@dataclass
class StationTrace:
    """Processed data for one station (kept for plotting)."""
    station: Station
    seismic: Waveform | None = None
    filtered: np.ndarray | None = None
    cft: np.ndarray | None = None
    infrasound: Waveform | None = None
    infra_filtered: np.ndarray | None = None
    infra_cft: np.ndarray | None = None
    triggers_total: int = 0        # all STA/LTA triggers, before the rate limit
    infra_triggers_total: int = 0


@dataclass
class AnalysisResult:
    traces: dict                  # station id -> StationTrace
    picks: list                   # seismic trigger picks
    events: list                  # DetectedEvent
    catalog: list = field(default_factory=list)
    settings: Settings | None = None
    t_start: float = 0.0
    t_end: float = 0.0
    messages: list = field(default_factory=list)
    infra_picks: list = field(default_factory=list)   # air-pressure trigger picks

    @property
    def stations(self):
        return [t.station for t in self.traces.values()]

    @property
    def unassociated(self):
        return [p for p in self.picks if p.event_id is None]

    @property
    def noise_picks(self):
        return [p for p in self.picks + self.infra_picks if p.event_id is None]


def _limit_rate(picks, hours, max_rate):
    """Very noisy stations: keep only the strongest triggers (max_rate per hour)."""
    n_max = max(int(round(max_rate * max(hours, 0.1))), 5)
    if len(picks) <= n_max:
        return picks
    keep = sorted(picks, key=lambda q: -q.peak_ratio)[:n_max]
    return sorted(keep, key=lambda q: q.time)


def analyze(waveforms: list[Waveform], settings: Settings | None = None,
            catalog: list[CatalogEvent] | None = None, progress=None) -> AnalysisResult:
    """Detect, associate, locate and classify everything in one time window."""
    settings = settings or Settings()
    p = settings.detection
    vm = settings.velocity
    catalog = catalog or []

    # ---- 1. single-station detection (seismic + infrasound) ---------------
    traces: dict[str, StationTrace] = {}
    for w in waveforms:
        st = traces.setdefault(w.station.id, StationTrace(w.station))
        if w.is_infrasound:
            st.infrasound = w
        else:
            st.seismic = w
    infra_params = replace(p, sta=1.0, lta=30.0, trigger_on=p.infra_trigger_on, trigger_off=1.5)
    all_picks, infra_picks = [], []
    for k, st in enumerate(traces.values()):
        if progress:
            progress(k / max(len(traces), 1), f"STA/LTA on {st.station.id}")
        if st.seismic is not None:
            w = st.seismic
            picks, st.filtered, st.cft = detect_station(w, p)
            st.triggers_total = len(picks)
            all_picks.extend(_limit_rate(picks, (w.endtime - w.starttime) / 3600, p.max_trigger_rate))
        if st.infrasound is not None:
            w = st.infrasound
            fmin, fmax = p.infrasound_band
            ipicks, st.infra_filtered, st.infra_cft = detect_station(w, infra_params, fmin, fmax)
            st.infra_triggers_total = len(ipicks)
            infra_picks.extend(_limit_rate(ipicks, (w.endtime - w.starttime) / 3600, p.max_trigger_rate))
    all_picks.sort(key=lambda q: q.time)
    infra_picks.sort(key=lambda q: q.time)

    seismic = [t.seismic for t in traces.values() if t.seismic is not None]
    every = [w for w in waveforms]
    t_start = min((w.starttime for w in every), default=0.0)
    t_end = max((w.endtime for w in every), default=0.0)
    res = AnalysisResult(traces, all_picks, [], catalog, settings, t_start, t_end,
                         infra_picks=infra_picks)
    if progress:
        progress(1.0, "Associating picks into events")

    # ---- 2. network events (>= min_stations) + distant earthquakes --------
    n_ev = 0
    max_span = _max_station_span(seismic) / vm.vs + 10.0
    for i, seed in enumerate(all_picks):
        if seed.event_id is not None:
            continue
        window, per_sta = [], {}
        for q in all_picks[i:]:
            if q.time - seed.time > max_span:
                break
            if q.event_id is None and per_sta.get(q.station.id, 0) < 3:
                per_sta[q.station.id] = per_sta.get(q.station.id, 0) + 1
                window.append(q)
        # stations whose first trigger could be the same wave (moveout <= distance / Vp)
        cands = {seed.station.id: seed}
        for q in window[1:]:
            if q.station.id in cands:
                continue
            d = float(haversine_km(seed.station.lat, seed.station.lon, q.station.lat, q.station.lon))
            if q.time - seed.time <= d / vm.vp + 3.0:
                cands[q.station.id] = q
        if len(cands) < p.min_stations:
            continue
        dist_ev = None
        if len(cands) >= max(p.min_stations, 5):
            dist_ev = _try_distant(list(cands.values()), settings, catalog, n_ev + 1)
        local_ev = _try_local(window, settings, n_ev + 1)
        ev = local_ev or dist_ev
        if local_ev and dist_ev:
            # both hypotheses fit: keep the one explaining more first arrivals
            n_loc = sum(ph == "P" for ph in local_ev.phases)
            ev = local_ev if n_loc >= dist_ev.n_stations - 1 else dist_ev
        if ev is None:
            continue
        for q, ph, r in zip(ev.picks, ev.phases, ev.residuals):
            q.phase, q.residual = ph, r
        n_ev += 1
        for q in ev.picks:
            q.event_id = ev.id
        if ev.kind == "local":
            _refine_local(ev, res)
            ev.in_region = bool(reg.contains(ev.lat, ev.lon, settings.polygon)) or \
                reg.distance_km(ev.lat, ev.lon, settings.polygon) <= ev.error_km
        else:
            ev.in_region = False
            _consume_distant(ev, res)
        res.events.append(ev)

    # ---- 3. size + classify the network events (after all of them exist, so
    #         one event's air-wave search cannot steal another event's picks)
    for ev in res.events:
        if ev.kind == "local":
            _measure_and_classify(ev, res)

    # ---- 4. small-network events (2-3 stations, source must be in the area)
    n_small = len(res.events)
    n_ev = _small_events(res, n_ev)
    for ev in res.events[n_small:]:
        _measure_and_classify(ev, res)

    # ---- 5. air-pressure (infrasound) events -------------------------------
    n_ev = _acoustic_events(res, n_ev)

    # ---- 6. whatever is left is local noise --------------------------------
    for q in res.picks + res.infra_picks:
        if q.event_id is None:
            q.phase = "noise"
    res.events.sort(key=lambda e: e.origin_time)
    return res


# ---------------------------------------------------------------------------
def _max_station_span(waveforms) -> float:
    if len(waveforms) < 2:
        return 0.0
    lat = np.array([w.station.lat for w in waveforms])
    lon = np.array([w.station.lon for w in waveforms])
    return float(np.max(haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])))


def _try_local(window: list[Pick], settings: Settings, num: int) -> DetectedEvent | None:
    """Associate picks around window[0] (the seed) and locate them.

    1. phase-free grid association decides which picks are P, S or unrelated;
    2. least-squares location on those picks, dropping the worst misfit until
       every residual is within tolerance.
    """
    p, vm = settings.detection, settings.velocity
    if len({q.station.id for q in window}) < p.min_stations:
        return None
    # the seed (window[0]) is taken as the first P; if it turns out to be a noise
    # burst the association is abandoned and the next pick becomes the seed
    args = ([q.station.lat for q in window], [q.station.lon for q in window],
            [q.time for q in window], p.pick_sigma, vm, settings.region)
    phases, inlier, cost = associate_grid([0], *args)
    # would the picks be explained better with the seed treated as noise?
    nxt = next((k for k, q in enumerate(window) if q.station.id != window[0].station.id), None)
    if nxt is not None:
        _, inl2, cost2 = associate_grid([nxt], *args)
        if cost2 < cost and not inl2[0]:
            return None
    # at most one P and one S per station (keep the first of each)
    chosen, seen = [], set()
    for q, ph, ok in zip(window, phases, inlier):
        if ok and (q.station.id, ph) not in seen:
            seen.add((q.station.id, ph))
            chosen.append((q, str(ph)))

    def n_sta(c):
        return len({q.station.id for q, _ in c})

    limit = max(3.0 * p.pick_sigma, 1.5)
    # 4 unknowns (lat, lon, depth, origin time) -> need at least 5 picks so that
    # the misfit can reveal a wrong association
    while n_sta(chosen) >= p.min_stations and len(chosen) >= 5:
        if sum(ph == "P" for _, ph in chosen) < 3:   # real events show P at >= 3 stations
            return None
        if chosen[0][0] is not window[0]:
            return None
        loc = locate([q.station.lat for q, _ in chosen], [q.station.lon for q, _ in chosen],
                     [q.time for q, _ in chosen], [ph for _, ph in chosen],
                     [p.pick_sigma if ph == "P" else 2 * p.pick_sigma for _, ph in chosen],
                     vm, settings.region)
        if loc.at_edge:
            return None
        norm = np.abs(loc.residuals) / np.array([1.0 if ph == "P" else 2.0 for _, ph in chosen])
        if loc.rms <= p.max_rms and norm.max() <= limit:
            break
        chosen.pop(int(np.argmax(norm)))
    else:
        return None
    ev = DetectedEvent(
        id=f"EV{num:03d}", kind="local", origin_time=loc.origin_time, lat=loc.lat, lon=loc.lon,
        depth=loc.depth, depth_min=loc.depth_min, depth_max=loc.depth_max,
        depth_constrained=loc.depth_constrained, error_km=loc.error_km, rms=loc.rms,
        picks=[q for q, _ in chosen])
    ev.phases = [ph for _, ph in chosen]
    ev.residuals = [float(r) for r in loc.residuals]
    return ev


def _try_distant(picks, settings, catalog, num) -> DetectedEvent | None:
    """Far-away earthquake: fit the P arrivals with the global travel-time table."""
    picks = list(picks)
    if len(picks) < 6:
        return None
    if _max_station_span([Waveform(q.station, "", 0, 1, np.zeros(1)) for q in picks]) < 300:
        return None
    n_min = max(5, int(np.ceil(0.8 * len(picks))))   # may reject at most 20% as noise
    while True:
        loc = locate_distant([q.station.lat for q in picks], [q.station.lon for q in picks],
                             [q.time for q in picks])
        if loc.rms <= 1.2 or len(picks) <= n_min:
            break
        picks.pop(int(np.argmax(np.abs(loc.residuals))))
    if loc.rms > 1.2:
        return None
    r = settings.region
    if (r["min_lat"] - 3 <= loc.lat <= r["max_lat"] + 3) and (r["min_lon"] - 3 <= loc.lon <= r["max_lon"] + 3):
        return None     # inside the local search area: the local locator handles it
    pw = plane_wave_fit([q.station.lat for q in picks], [q.station.lon for q in picks],
                        [q.time for q in picks])
    # a real distant P wave sweeps across the network at 8-30 km/s; near-simultaneous
    # noise bursts (infinite speed) or slow air waves are rejected
    if not 8.0 <= pw.app_velocity <= 30.0:
        return None
    ev = DetectedEvent(
        id=f"EV{num:03d}", kind="distant", origin_time=loc.origin_time, lat=loc.lat, lon=loc.lon,
        depth=0.0, rms=loc.rms, picks=picks, back_azimuth=loc.back_azimuth,
        app_velocity=pw.app_velocity, distance_deg=loc.distance_deg,
        error_km=max(100.0, 0.1 * loc.distance_deg * KM_PER_DEG),
        label="Distant earthquake", p_explosion=0.0)
    ev.phases = ["P"] * len(picks)
    ev.residuals = [float(r) for r in loc.residuals]
    # match against the global catalogue (our origin time assumes a surface
    # source, so a deep catalogue event appears up to depth/8.5 s earlier)
    for c in catalog:
        dt = c.time - loc.origin_time
        sep = float(haversine_km(c.lat, c.lon, loc.lat, loc.lon))
        if -30 <= dt <= 60 + c.depth / 8.5 and sep <= max(600.0, 0.2 * loc.distance_deg * KM_PER_DEG):
            ev.catalog_match = c
            ev.lat, ev.lon, ev.depth = c.lat, c.lon, c.depth
            ev.origin_time, ev.error_km = c.time, 0.0
            break
    where = (f"matched to {ev.catalog_match.source}: {ev.catalog_match.description or 'M%.1f' % ev.catalog_match.magnitude}"
             if ev.catalog_match else f"about {loc.distance_deg:.0f} degrees ({loc.distance_deg * KM_PER_DEG:,.0f} km) away")
    ev.summary = (f"Distant earthquake far outside the Gulf: P waves arrived from back-azimuth "
                  f"{loc.back_azimuth:.0f} deg, crossing the network at {pw.app_velocity:.1f} km/s ({where}).")
    return ev


def _consume_distant(ev: DetectedEvent, res: AnalysisResult):
    """Mark the P coda and S wave of a distant quake at each station so they don't
    make fake local events (only near the predicted times - local events in
    between are still found)."""
    for q in res.picks:
        if q.event_id is not None:
            continue
        dd = float(haversine_km(ev.lat, ev.lon, q.station.lat, q.station.lon)) / KM_PER_DEG
        tp = ev.origin_time + float(physics.tele_p_time(dd, ev.depth))
        ts = ev.origin_time + float(physics.tele_s_time(dd, ev.depth))
        if tp - 5 <= q.time <= tp + 90 or ts - 15 <= q.time <= ts + 120:
            q.event_id, q.phase = ev.id, "later phase"


# ---------------------------------------------------------------------------
def _refine_local(ev: DetectedEvent, res: AnalysisResult):
    """Attach missed P picks, add S picks, relocate, and absorb later re-triggers."""
    s = res.settings
    p, vm = s.detection, s.velocity
    tol_p = 3.0 * p.pick_sigma + 1.0

    def predicted(sta):
        d = float(haversine_km(ev.lat, ev.lon, sta.lat, sta.lon))
        return (ev.origin_time + float(physics.p_time(d, ev.depth, vm)),
                ev.origin_time + float(physics.s_time(d, ev.depth, vm)))

    # ---- 1. P picks at stations that were not in the seed cluster ---------
    has_p = {q.station.id for q in ev.picks if q.phase == "P"}
    has_s = {q.station.id for q in ev.picks if q.phase == "S"}
    for q in res.picks:
        if q.event_id is None and q.station.id not in has_p:
            tp, _ = predicted(q.station)
            if abs(q.time - tp) <= tol_p:
                q.event_id, q.phase = ev.id, "P"
                ev.picks.append(q)
                has_p.add(q.station.id)

    # ---- 2. S picks -----------------------------------------------------------
    s_picks = []
    for sid, st in res.traces.items():
        wf = st.seismic
        if wf is None:
            continue
        tp, ts = predicted(wf.station)
        if ts - tp < 3.0 or sid in has_s:
            continue
        tol = 2.0 + 0.04 * (ts - ev.origin_time)
        # (a) an existing trigger close to the predicted S time
        near = [q for q in res.picks if q.event_id is None and q.station.id == sid
                and abs(q.time - ts) <= tol]
        if near:
            q = min(near, key=lambda q: abs(q.time - ts))
            q.event_id, q.phase = ev.id, "S"
            s_picks.append(q)
            continue
        # (b) S stands out from the P coda on a short-window STA/LTA
        if sid not in has_p:
            continue
        i1, i2 = wf.index(ts - tol), wf.index(ts + tol)
        if i1 < 0 or i2 >= len(st.cft):
            continue
        seg = st.filtered[max(i1 - int(10 * wf.fs), 0):i2]
        cft = sta_lta(seg, wf.fs, 0.5, 5.0)[-(i2 - i1):]
        if not len(cft) or cft.max() < 2.5:
            continue
        j = int(np.argmax(cft))
        a = max(i1 + j - int(2 * wf.fs), 0)
        b = min(i1 + j + int(1 * wf.fs), len(st.filtered))
        t_s = wf.starttime + (a + aic_pick(st.filtered[a:b])) / wf.fs
        s_picks.append(Pick(wf.station, t_s, "S", snr=float(cft.max()), channel=wf.channel,
                            event_id=ev.id))

    # ---- 3. relocate with P + S --------------------------------------------
    def relocate(picks):
        return locate([q.station.lat for q in picks], [q.station.lon for q in picks],
                      [q.time for q in picks], [q.phase for q in picks],
                      [p.pick_sigma if q.phase == "P" else 2 * p.pick_sigma for q in picks],
                      vm, s.region)

    allp = ev.picks + s_picks
    loc = relocate(allp)
    bad = [q for q, r in zip(allp, loc.residuals) if abs(r) > max(3.0, 4 * p.pick_sigma)]
    if bad and len(allp) - len(bad) >= p.min_stations:
        for q in bad:
            q.event_id, q.phase = None, "?"
        allp = [q for q in allp if q not in bad]
        loc = relocate(allp)
    if not loc.at_edge:
        ev.lat, ev.lon, ev.depth, ev.origin_time = loc.lat, loc.lon, loc.depth, loc.origin_time
        ev.depth_min, ev.depth_max = loc.depth_min, loc.depth_max
        ev.depth_constrained, ev.error_km, ev.rms = loc.depth_constrained, loc.error_km, loc.rms
        ev.picks = allp
        for q, r in zip(allp, loc.residuals):
            q.residual = float(r)

    # ---- 4. absorb re-triggers between P and the end of the S coda ---------
    for q in res.picks:
        if q.event_id is None:
            tp, ts = predicted(q.station)
            if tp - 1.0 <= q.time <= ts + max(15.0, 0.6 * (ts - tp)):
                q.event_id, q.phase = ev.id, "coda"


def event_distance(ev: DetectedEvent, sta) -> float:
    """Epicentral distance (km); single-station events use the ring radius."""
    if ev.ring is not None and ev.ring[0].id == sta.id:
        return 0.5 * (ev.ring[1] + ev.ring[2])
    return float(haversine_km(ev.lat, ev.lon, sta.lat, sta.lon))


def _measure_and_classify(ev: DetectedEvent, res: AnalysisResult):
    """Magnitude, discrimination measurements, air-blast search, verdict."""
    vm = res.settings.velocity
    log_ps, pols = [], []
    for q in ev.picks:
        if q.phase != "P":
            continue
        st = res.traces[q.station.id]
        wf = st.seismic
        sta = wf.station
        d = event_distance(ev, sta)
        hypo = float(np.hypot(d, ev.depth))
        tp = q.time
        ts = ev.origin_time + float(physics.s_time(d, ev.depth, vm))
        sp = ts - tp
        pols.append(q.polarity if q.snr >= 8 else 0)
        # Wood-Anderson amplitude
        t1, t2 = tp - 30.0, ts + max(10.0, 0.5 * sp) + 10.0
        seg = wf.window(t1, t2)
        if len(seg) < wf.fs * 40:
            continue
        seg = bandpass(seg, wf.fs, 0.8, min(20.0, 0.45 * wf.fs))
        wa = physics.velocity_to_wood_anderson(seg, wf.fs)
        k0 = int((tp - 1 - t1) * wf.fs)
        noise = np.max(np.abs(wa[int(5 * wf.fs):k0])) if k0 > 5 * wf.fs else 0.0
        amp = float(np.max(np.abs(wa[k0:])))
        if amp > 2.5 * noise and hypo <= 1500:
            amp_nm = amp * 1e9
            ev.ml_stations.append({"station": sta.id, "site": sta.site, "dist_km": d, "hypo_km": hypo,
                                   "amp_nm": amp_nm, "ml": float(physics.local_magnitude(amp_nm, hypo))})
        # the P/S ratio needs well separated phases
        if sp < 2.0:
            continue
        hf = bandpass(seg, wf.fs, 4.0, min(12.0, 0.45 * wf.fs))
        kp1 = int((tp - 0.1 - t1) * wf.fs)
        kp2 = int((tp + min(0.45 * sp, 5.0) - t1) * wf.fs)
        ks1 = int((ts - 0.3 - t1) * wf.fs)
        ks2 = int((ts + min(0.45 * sp, 8.0) - t1) * wf.fs)
        noise_hf = np.max(np.abs(hf[int(5 * wf.fs):k0])) if k0 > 5 * wf.fs else 0.0
        ap = np.max(np.abs(hf[kp1:kp2])) if kp2 > kp1 else 0.0
        as_ = np.max(np.abs(hf[ks1:ks2])) if ks2 > ks1 else 0.0
        if ap > 3 * noise_hf and as_ > 1.5 * noise_hf and as_ > 0:
            log_ps.append(float(np.log10(ap / as_)))

    if ev.ml_stations:
        ev.ml = float(np.median([m["ml"] for m in ev.ml_stations]))

    # ---- air-blast search ---------------------------------------------------
    if ev.tier in ("acoustic", "single"):     # found from the air wave in the first place
        n_infra, n_seis_air, n_sensors = len(ev.air_picks), 0, len(ev.air_picks)
    else:
        n_infra, n_seis_air, n_sensors = _air_search(ev, res)

    evidence = [e for e in (
        disc.ps_ratio_evidence(log_ps),
        disc.depth_evidence(ev),
        disc.polarity_evidence(pols),
        disc.air_evidence(n_infra, n_seis_air, n_sensors),
        disc.catalog_evidence(ev, res.catalog),
        disc.time_of_day_evidence(ev),
    ) if e is not None]
    disc.classify(ev, evidence)
    _tier_label(ev)


def _air_search(ev: DetectedEvent, res: AnalysisResult):
    """Look for the explosion's air wave at every station within range."""
    p = res.settings.detection
    infra_hits, seis_cands, n_sensors = [], [], 0
    for sid, st in res.traces.items():
        sta = st.station
        d = float(haversine_km(ev.lat, ev.lon, sta.lat, sta.lon))
        if d > p.max_air_range_km or d < 1:
            continue
        w1 = ev.origin_time + d / p.celerity_max
        w2 = ev.origin_time + d / p.celerity_min
        if st.infrasound is not None:
            iw = st.infrasound
            i1, i2 = iw.index(w1), iw.index(w2)
            if i1 >= 0 and i2 < len(iw.data):
                n_sensors += 1
                seg = st.infra_cft[i1:i2]
                if len(seg) and seg.max() >= p.infra_trigger_on:
                    j = i1 + int(np.argmax(seg))
                    a = max(j - int(4 * iw.fs), 0)
                    t_air = iw.starttime + (a + aic_pick(st.infra_filtered[a:j + int(iw.fs)])) / iw.fs
                    amp = float(np.max(np.abs(st.infra_filtered[j - int(iw.fs):j + int(5 * iw.fs)])))
                    infra_hits.append(Pick(sta, t_air, "Air", snr=float(seg.max()), amplitude=amp,
                                           channel=iw.channel, event_id=ev.id))
                    for q in res.infra_picks:      # the matching infrasound trigger
                        if q.station.id == sid and abs(q.time - t_air) < 10 and q.event_id is None:
                            q.event_id, q.phase = ev.id, "Air"
        for q in res.picks:
            if q.event_id is None and q.station.id == sid and w1 <= q.time <= w2:
                seis_cands.append(q)
                break

    # accept seismic (air-coupled) candidates only if their celerities agree
    def celerity(q):
        d = float(haversine_km(ev.lat, ev.lon, q.station.lat, q.station.lon))
        return d / (q.time - ev.origin_time)

    ref = [celerity(q) for q in infra_hits]
    accepted = []
    if seis_cands:
        c = np.array([celerity(q) for q in seis_cands])
        centre = np.median(ref) if ref else np.median(c)
        ok = np.abs(c - centre) <= 0.02
        if ref or ok.sum() >= 3:
            accepted = [q for q, k in zip(seis_cands, ok) if k]
    for q in accepted:
        q.event_id, q.phase = ev.id, "Air"
    ev.air_picks = infra_hits + accepted
    return len(infra_hits), len(accepted), n_sensors


# ---------------------------------------------------------------------------
# Labels for detections made with few stations
# ---------------------------------------------------------------------------
def _tier_label(ev: DetectedEvent):
    """Fewer stations -> weaker claims ("Possible ..." instead of "Likely ...")."""
    if ev.tier == "network":
        return
    if ev.label == "Likely explosion":
        strong = ev.tier == "acoustic" and any(q.phase == "P" for q in ev.picks)
        ev.label = "Likely explosion" if strong else "Possible explosion"
    elif ev.label == "Likely earthquake":
        ev.label = "Possible earthquake"
    note = {"small": f"Detected by only {ev.n_stations} seismic stations: the location is an area "
                     "(shaded on the map), not a point.",
            "acoustic": "Located from the air-pressure (infrasound) arrivals.",
            "single": "Only one station: the distance comes from the delay between the ground "
                      "wave and the air wave; the direction is unknown (ring on the map)."}.get(ev.tier)
    if note and note not in ev.notes:
        ev.notes.append(note)
    if ev.tier in ("acoustic", "single") and not any(q.phase == "P" for q in ev.picks):
        ev.notes.append("Air wave only: thunder, sonic booms and other loud sounds look similar.")
    ev.summary = ev.summary.replace("Likely explosion (", f"{ev.label} (").replace(
        "Likely earthquake (", f"{ev.label} (")


def _subsample(lat, lon, n=1500):
    k = max(len(lat) // n, 1)
    return [float(x) for x in lat[::k]], [float(x) for x in lon[::k]]


# ---------------------------------------------------------------------------
# Small-network events: 2-3 seismic stations
# ---------------------------------------------------------------------------
def _small_events(res: AnalysisResult, n_ev: int) -> int:
    p, vm = res.settings.detection, res.settings.velocity
    free = [q for q in res.picks if q.event_id is None]
    for i, seed in enumerate(free):
        if seed.event_id is not None:
            continue
        group = {seed.station.id: seed}
        for q in free[i + 1:]:
            if q.time - seed.time > 200:
                break
            if q.event_id is not None or q.station.id in group:
                continue
            d = float(haversine_km(seed.station.lat, seed.station.lon, q.station.lat, q.station.lon))
            if q.time - seed.time <= d / vm.vp + 2.0:
                group[q.station.id] = q
        if len(group) < p.small_min_stations:
            continue
        ev = _try_small(list(group.values()), res, n_ev + 1)
        if ev is None:
            continue
        n_ev += 1
        res.events.append(ev)
    return n_ev


def _s_candidates(picks, res):
    """For each P pick, the next free trigger at the same station (possible S wave)."""
    out = {}
    for q in picks:
        later = [x for x in res.picks if x.station.id == q.station.id and x.event_id is None
                 and 1.5 <= x.time - q.time <= 90.0 and x.snr >= 3.0]
        if later:
            out[q.station.id] = min(later, key=lambda x: x.time)
    return out


def _feasible(picks, s_picks, res, tol):
    """Grid nodes inside the area whose P (and S-P) times explain the picks (surface source).

    P:    spread of (t_i - T_P,i) over stations <= tol
    S-P:  |(t_S - t_P) - (T_S - T_P)| <= 2 s + 5 %   at stations with an S candidate
    """
    vm = res.settings.velocity
    LA, LO = reg.grid(0.05, tuple(res.settings.polygon))
    st_lat = np.array([q.station.lat for q in picks])
    st_lon = np.array([q.station.lon for q in picks])
    t = np.array([q.time for q in picks]) - picks[0].time
    dist = haversine_km(LA[:, None], LO[:, None], st_lat[None, :], st_lon[None, :])
    tt = physics.p_time(dist, 0.0, vm)
    t0 = t[None, :] - tt
    spread = t0.max(axis=1) - t0.min(axis=1)
    ok = spread <= tol
    for k, q in enumerate(picks):
        sq = s_picks.get(q.station.id)
        if sq is not None:
            sp_pred = physics.s_time(dist[:, k], 0.0, vm) - tt[:, k]
            sp_obs = sq.time - q.time
            ok &= np.abs(sp_obs - sp_pred) <= 2.0 + 0.05 * sp_obs
    return LA, LO, t0, spread, ok


def _try_small(group, res, num):
    """2-3 stations.  The source must be inside the monitored area: if the arrival
    times can only be explained from outside it, nothing is reported."""
    p, vm = res.settings.detection, res.settings.velocity
    picks = sorted(group, key=lambda q: q.time)
    if len(picks) >= 3:
        picks = [q for q in picks if q.snr >= 4.0]
    if len(picks) < 2:
        return None
    if len(picks) == 2:
        a, b = picks
        d = float(haversine_km(a.station.lat, a.station.lon, b.station.lat, b.station.lon))
        # two stations can coincide by chance: demand strong, nearby signals
        if min(a.snr, b.snr) < p.min_snr_small or d > p.max_small_pair_km:
            return None
    s_picks = _s_candidates(picks, res)
    tol = 2 * p.pick_sigma + 0.8
    LA, LO, t0, spread, ok = _feasible(picks, s_picks, res, tol)
    if not ok.any():
        return None
    j = int(np.argmin(np.where(ok, spread, np.inf)))
    lat, lon = float(LA[j]), float(LO[j])
    err = float(max(np.max(haversine_km(lat, lon, LA[ok], LO[ok])), 5.0))
    # too poorly constrained to be useful (two P times alone = a long curve); two
    # stations are the weakest case, so they must pin the source down twice as well
    if err > (p.max_small_error_km if len(picks) >= 3 else 0.5 * p.max_small_error_km):
        return None
    origin = picks[0].time + float(np.mean(t0[j]))
    ev = DetectedEvent(id=f"EV{num:03d}", kind="local", origin_time=origin, lat=lat, lon=lon,
                       depth=0.0, depth_min=0.0, depth_max=40.0, depth_constrained=False,
                       error_km=err, rms=float(spread[j]) / 2, picks=list(picks), tier="small")
    ev.feasible_lat, ev.feasible_lon = _subsample(LA[ok], LO[ok])
    for q in picks:
        q.event_id, q.phase = ev.id, "P"
        d = float(haversine_km(lat, lon, q.station.lat, q.station.lon))
        q.residual = float(q.time - origin - physics.p_time(d, 0.0, vm))
    for sq in s_picks.values():
        sq.event_id, sq.phase = ev.id, "S"
        ev.picks.append(sq)
    # absorb S / coda re-triggers at every station
    for q in res.picks:
        if q.event_id is None:
            d = float(haversine_km(lat, lon, q.station.lat, q.station.lon))
            tp = origin + float(physics.p_time(d, 0.0, vm))
            ts = origin + float(physics.s_time(d, 0.0, vm))
            slack = err / vm.vs * 0.5 + 2.0
            if tp - 1.0 <= q.time <= ts + slack + max(15.0, 0.6 * (ts - tp)):
                q.event_id, q.phase = ev.id, "coda"
    return ev


# ---------------------------------------------------------------------------
# Air-pressure (infrasound) events
# ---------------------------------------------------------------------------
def _acoustic_events(res: AnalysisResult, n_ev: int) -> int:
    p = res.settings.detection
    for ev in res.events:
        _consume_air(ev, res)
    free = [q for q in res.infra_picks if q.event_id is None]
    for i, seed in enumerate(free):
        if seed.event_id is not None:
            continue
        group = {seed.station.id: seed}
        for q in free[i + 1:]:
            if q.time - seed.time > p.max_air_range_km / p.celerity_min:
                break
            if q.event_id is not None or q.station.id in group:
                continue
            d = float(haversine_km(seed.station.lat, seed.station.lon, q.station.lat, q.station.lon))
            if q.time - seed.time <= d / p.celerity_min + 5.0:
                group[q.station.id] = q
        ev = None
        if len(group) >= 2:
            ev = _try_acoustic(list(group.values()), res, n_ev + 1)
        if ev is None:
            ev = _try_single(seed, res, n_ev + 1)
        if ev is None:
            continue
        n_ev += 1
        _measure_and_classify(ev, res)
        _consume_air(ev, res)
        res.events.append(ev)
    return n_ev


def _consume_air(ev: DetectedEvent, res: AnalysisResult):
    """Triggers inside the air-wave window of an explosion-like event (the air wave
    reaching farther stations, reverberations, air-coupled ground shaking) belong
    to it and must not start new events."""
    p = res.settings.detection
    if ev.kind != "local" or ev.p_explosion < 0.5:
        return
    for q in res.infra_picks + res.picks:
        if q.event_id is None:
            d = event_distance(ev, q.station)
            if d <= 2 * p.max_air_range_km and \
                    ev.origin_time + d / p.celerity_max - 10 <= q.time <= ev.origin_time + d / p.celerity_min + 60:
                q.event_id, q.phase = ev.id, "Air (later)"


def _ground_wave(ev, res, slack):
    """Seismic P arrivals consistent with the source found from the air wave."""
    vm = res.settings.velocity
    found = []
    for sid, st in res.traces.items():
        if st.seismic is None:
            continue
        d = float(haversine_km(ev.lat, ev.lon, st.station.lat, st.station.lon))
        tp = ev.origin_time + float(physics.p_time(d, 0.0, vm))
        cands = [q for q in res.picks if q.station.id == sid and q.event_id is None
                 and abs(q.time - tp) <= slack]
        if cands:
            q = max(cands, key=lambda q: q.snr)
            q.event_id, q.phase, q.residual = ev.id, "P", float(q.time - tp)
            found.append(q)
    return found


def _try_acoustic(group, res, num):
    p, vm = res.settings.detection, res.settings.velocity
    picks = sorted((q for q in group if q.snr >= 5.0), key=lambda q: q.time)
    if len(picks) < 2 or max(q.snr for q in picks) < 8.0:
        return None
    LA, LO = reg.grid(0.05, tuple(res.settings.polygon))
    st_lat = np.array([q.station.lat for q in picks])
    st_lon = np.array([q.station.lon for q in picks])
    t = np.array([q.time for q in picks]) - picks[0].time
    d = haversine_km(LA[:, None], LO[:, None], st_lat[None, :], st_lon[None, :])
    # origin-time interval each station allows for celerities cmin..cmax
    lo_t = t[None, :] - d / p.celerity_min
    hi_t = t[None, :] - d / p.celerity_max
    ok = lo_t.max(axis=1) <= hi_t.min(axis=1) + 3.0
    ok &= d.max(axis=1) <= p.max_air_range_km
    if not ok.any():
        return None
    c_mid = 0.5 * (p.celerity_min + p.celerity_max)
    t0 = t[None, :] - d / c_mid
    spread = np.where(ok, t0.max(axis=1) - t0.min(axis=1), np.inf)
    j = int(np.argmin(spread))
    lat, lon = float(LA[j]), float(LO[j])
    err = float(max(np.max(haversine_km(lat, lon, LA[ok], LO[ok])), 5.0))
    ev = DetectedEvent(id=f"EV{num:03d}", kind="local", origin_time=picks[0].time + float(np.mean(t0[j])),
                       lat=lat, lon=lon, depth=0.0, depth_max=0.0, error_km=err,
                       rms=float(spread[j]) / 2, picks=[], tier="acoustic")
    ev.feasible_lat, ev.feasible_lon = _subsample(LA[ok], LO[ok])
    for q in picks:
        q.event_id, q.phase = ev.id, "Air"
    ev.air_picks = list(picks)
    ev.picks = _ground_wave(ev, res, 3.0 + min(err, 100.0) / vm.vp)
    return ev


def _try_single(seed, res, num):
    """One station with both a ground wave and a later, strong air wave.

    t_air - t_ground = d / c - d / Vp   ->   d = dt / (1/c - 1/Vp)
    """
    p, vm = res.settings.detection, res.settings.velocity
    if seed.snr < 8.0:
        return None
    st = res.traces[seed.station.id]
    if st.seismic is None:
        return None
    max_d = 150.0
    dt_max = max_d * (1 / p.celerity_min - 1 / vm.vp)
    cands = [q for q in res.picks if q.station.id == seed.station.id and q.snr >= 5.0
             and 2.0 <= seed.time - q.time <= dt_max and q.event_id is None]
    if len(cands) != 1:          # none, or ambiguous
        return None
    g = cands[0]
    dt = seed.time - g.time
    d_lo = dt / (1 / p.celerity_min - 1 / vm.vp)
    d_hi = dt / (1 / p.celerity_max - 1 / vm.vp)
    LA, LO = reg.grid(0.05, tuple(res.settings.polygon))
    dist = haversine_km(seed.station.lat, seed.station.lon, LA, LO)
    ok = (dist >= d_lo - 3) & (dist <= d_hi + 3)
    if not ok.any():
        return None
    d_mid = 0.5 * (d_lo + d_hi)
    ev = DetectedEvent(id=f"EV{num:03d}", kind="local", origin_time=g.time - d_mid / vm.vp,
                       lat=float(np.mean(LA[ok])), lon=float(np.mean(LO[ok])), depth=0.0,
                       error_km=d_hi, picks=[g], tier="single",
                       ring=(seed.station, float(d_lo), float(d_hi)))
    ev.feasible_lat, ev.feasible_lon = _subsample(LA[ok], LO[ok])
    g.event_id, g.phase = ev.id, "P"
    seed.event_id, seed.phase = ev.id, "Air"
    ev.air_picks = [seed]
    return ev
