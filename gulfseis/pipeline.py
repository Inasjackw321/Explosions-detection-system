"""Full processing chain: detect -> associate -> locate -> size -> classify."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import discrimination as disc
from . import physics
from .config import Settings
from .detection import aic_pick, bandpass, detect_station, sta_lta
from .geo import KM_PER_DEG, azimuth_deg, destination, haversine_km
from .location import associate_grid, locate, locate_distant, plane_wave_fit
from .models import CatalogEvent, DetectedEvent, Pick, Waveform


@dataclass
class StationTrace:
    """Processed data for one station (kept for plotting)."""
    seismic: Waveform
    filtered: np.ndarray
    cft: np.ndarray
    infrasound: Waveform | None = None
    infra_filtered: np.ndarray | None = None
    infra_cft: np.ndarray | None = None


@dataclass
class AnalysisResult:
    traces: dict                  # station id -> StationTrace
    picks: list                   # all seismic trigger picks
    events: list                  # DetectedEvent
    catalog: list = field(default_factory=list)
    settings: Settings | None = None
    t_start: float = 0.0
    t_end: float = 0.0
    messages: list = field(default_factory=list)

    @property
    def stations(self):
        return [t.seismic.station for t in self.traces.values()]

    @property
    def unassociated(self):
        return [p for p in self.picks if p.event_id is None]


def analyze(waveforms: list[Waveform], settings: Settings | None = None,
            catalog: list[CatalogEvent] | None = None, progress=None) -> AnalysisResult:
    settings = settings or Settings()
    p = settings.detection
    vm = settings.velocity
    catalog = catalog or []

    # ---- 1. single-station detection --------------------------------------
    traces: dict[str, StationTrace] = {}
    all_picks: list[Pick] = []
    seismic = [w for w in waveforms if not w.is_infrasound]
    infra = {w.station.id: w for w in waveforms if w.is_infrasound}
    for k, wf in enumerate(seismic):
        if progress:
            progress(k / max(len(seismic), 1), f"STA/LTA on {wf.station.id}")
        picks, filt, cft = detect_station(wf, p)
        st = StationTrace(wf, filt, cft)
        iw = infra.get(wf.station.id)
        if iw is not None:
            fmin, fmax = p.infrasound_band
            st.infrasound = iw
            st.infra_filtered = bandpass(iw.data, iw.fs, fmin, fmax)
            st.infra_cft = sta_lta(st.infra_filtered, iw.fs, 2.0, 30.0)
        traces[wf.station.id] = st
        all_picks.extend(picks)
    all_picks.sort(key=lambda q: q.time)

    t_start = min((w.starttime for w in seismic), default=0.0)
    t_end = max((w.endtime for w in seismic), default=0.0)
    res = AnalysisResult(traces, all_picks, [], catalog, settings, t_start, t_end)
    if progress:
        progress(1.0, "Associating picks into events")

    # ---- 2. association + location ---------------------------------------
    n_ev = 0
    max_span = _max_station_span(seismic) / vm.vs + 10.0
    for i, seed in enumerate(all_picks):
        if seed.event_id is not None:
            continue
        window = [q for q in all_picks[i:] if q.event_id is None and q.time - seed.time <= max_span]
        # distant earthquakes first: a plane wave crossing the whole network
        # (earliest pick per station, moveout <= distance / Vp)
        cands = {seed.station.id: seed}
        for q in window[1:]:
            if q.station.id in cands:
                continue
            d = float(haversine_km(seed.station.lat, seed.station.lon, q.station.lat, q.station.lon))
            if q.time - seed.time <= d / vm.vp + 3.0:
                cands[q.station.id] = q
        dist_ev = None
        if len(cands) >= max(p.min_stations, 5):
            dist_ev = _try_distant(list(cands.values()), settings, catalog, n_ev + 1)
        local_ev = _try_local(window, settings, n_ev + 1)
        ev = local_ev or dist_ev
        if local_ev and dist_ev:
            # both hypotheses fit: keep the one explaining more first arrivals
            n_loc = sum(ph == "P" for ph in local_ev.phases)
            ev = local_ev if n_loc >= dist_ev.n_stations - 1 else dist_ev
        if ev is not None:
            for q, ph, r in zip(ev.picks, ev.phases, ev.residuals):
                q.phase, q.residual = ph, r
        if ev is None:
            continue
        n_ev += 1
        for q in ev.picks:
            q.event_id = ev.id
        if ev.kind == "local":
            _refine_local(ev, res)
        else:
            _consume_distant(ev, res)
        res.events.append(ev)

    # ---- 3. size + classify (after all events exist, so that one event's
    #         air-wave search cannot steal another event's P picks) ----------
    for ev in res.events:
        if ev.kind == "local":
            _measure_and_classify(ev, res)
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
    """Mark later phases (S, surface waves) of a distant quake so they don't make fake events."""
    dist = ev.distance_deg or 30.0
    dt_sp = float(physics.tele_s_time(dist) - physics.tele_p_time(dist))
    t_first = min(q.time for q in ev.picks)
    t_last = t_first + dt_sp + 400.0
    for q in res.picks:
        if q.event_id is None and t_first <= q.time <= t_last:
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
        d = float(haversine_km(ev.lat, ev.lon, sta.lat, sta.lon))
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
    if ev.n_stations < 4:
        ev.notes.append("Only 3 stations: location is exactly determined, so errors cannot be checked.")


def _air_search(ev: DetectedEvent, res: AnalysisResult):
    """Look for the explosion's air wave at every station within range."""
    p = res.settings.detection
    infra_hits, seis_cands, n_sensors = [], [], 0
    for sid, st in res.traces.items():
        sta = st.seismic.station
        d = float(haversine_km(ev.lat, ev.lon, sta.lat, sta.lon))
        if d > p.max_air_range_km or d < 5:
            continue
        w1 = ev.origin_time + d / p.celerity_max
        w2 = ev.origin_time + d / p.celerity_min
        if st.infrasound is not None:
            iw = st.infrasound
            i1, i2 = iw.index(w1), iw.index(w2)
            if i1 >= 0 and i2 < len(iw.data):
                n_sensors += 1
                seg = st.infra_cft[i1:i2]
                if len(seg) and seg.max() >= p.trigger_on:
                    j = i1 + int(np.argmax(seg))
                    a = max(j - int(4 * iw.fs), 0)
                    t_air = iw.starttime + (a + aic_pick(st.infra_filtered[a:j + int(iw.fs)])) / iw.fs
                    amp = float(np.max(np.abs(st.infra_filtered[j - int(iw.fs):j + int(5 * iw.fs)])))
                    infra_hits.append(Pick(sta, t_air, "Air", snr=float(seg.max()), amplitude=amp,
                                           channel=iw.channel, event_id=ev.id))
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
