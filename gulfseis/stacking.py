"""Network stacking ("brightness") detector.

Instead of asking every station to trigger on its own, the STA/LTA traces of
all stations are shifted by the travel time from a trial source and added up.
A real event makes the shifted traces line up, so their sum stands out even if
each station alone is below its trigger level; noise at one station does not
line up with the others.

For a trial source x and origin time t:

    c_i(t) = clip( R_i(t) - q_i , 0, 5 )          station excess STA/LTA
                                                  (q_i = its 90th percentile, so
                                                  noisy stations are not favoured)
    s_i(x, t) = c_i(t + T_P,i(x)) + c_i(t + T_S,i(x))   (P and S must BOTH line up at
                                                  the right source, which pins the location)
    B(x, t) = sum_i s_i(x, t) - max_i s_i(x, t)   brightness

Subtracting the largest term means one station alone can never make an event:
at least two stations must contribute.  A detection is a peak of
max_x B(x, t) above  max(threshold, median + k * 1.4826 * MAD), supported by
at least 3 stations, or by 2 stations that each show BOTH a P and an S wave
(two P or S times alone can always be matched by some trial source).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d

from . import physics
from . import region as reg
from .geo import haversine_km

CF_RATE = 2.0      # Hz, sampling of the stacked characteristic functions
CF_CAP = 5.0
SUPPORT = 2.0      # a station "supports" a source if its excess STA/LTA there is >= this


@dataclass
class StackDetection:
    time: float            # origin time (POSIX s)
    lat: float
    lon: float
    brightness: float
    threshold: float
    error_km: float
    contributions: dict    # station id -> (s_i, "P"/"S")
    area_lat: list
    area_lon: list


@dataclass
class StackTrace:
    t0: float
    rate: float
    bmax: np.ndarray       # max over trial sources of B, per time sample
    threshold: float


def _station_cf(st, t_start, n, p):
    """Excess STA/LTA of one station, resampled to CF_RATE on the common time axis."""
    w, cft = st.seismic, st.cft
    q = float(np.percentile(cft[cft > 0], 90)) if np.any(cft > 0) else 1.0
    exc = np.clip(cft - max(q, 1.0), 0.0, CF_CAP)
    # the start of a record (LTA still filling, filter start-up) and its end are unreliable
    exc[:int((2 * p.lta + p.sta + 20.0) * w.fs)] = 0.0
    exc[max(len(exc) - int(5 * w.fs), 0):] = 0.0
    k = max(int(round(w.fs / CF_RATE)), 1)
    m = (len(exc) // k) * k
    down = exc[:m].reshape(-1, k).max(axis=1)
    out = np.zeros(n)
    i0 = int(round((w.starttime - t_start) * CF_RATE))
    a, b = max(i0, 0), min(i0 + len(down), n)
    if b > a:
        out[a:b] = down[a - i0:b - i0]
    # tolerate travel-time model errors of +-1.5 s
    return maximum_filter1d(out, size=int(3 * CF_RATE) + 1)


def _mask_known(C, sts, t_start, known_local, known_distant, p, vm):
    """Zero the parts of each station's trace already explained by known events:
    their P-S wave train and, for explosion-like events, the later air wave."""
    n = C.shape[1]

    def zero(i, t1, t2):
        a, b = int((t1 - t_start) * CF_RATE), int((t2 - t_start) * CF_RATE) + 1
        C[i, max(a, 0):max(min(b, n), 0)] = 0.0

    for i, t in enumerate(sts):
        for e in known_local:
            d = float(haversine_km(e.lat, e.lon, t.station.lat, t.station.lon))
            tp = e.origin_time + float(physics.p_time(d, e.depth, vm))
            ts = e.origin_time + float(physics.s_time(d, e.depth, vm))
            zero(i, tp - 3 - e.error_km / vm.vp, ts + 20 + max(0.6 * (ts - tp), e.error_km / vm.vs))
            if e.p_explosion >= 0.5 and d <= 2 * p.max_air_range_km:
                zero(i, e.origin_time + d / p.celerity_max - 15, e.origin_time + d / p.celerity_min + 60)
        for e in known_distant:
            dd = d = float(haversine_km(e.lat, e.lon, t.station.lat, t.station.lon)) / 111.19
            tp = e.origin_time + float(physics.tele_p_time(dd, e.depth))
            ts = e.origin_time + float(physics.tele_s_time(d, e.depth))
            zero(i, tp - 5, tp + 120)
            zero(i, ts - 20, ts + 150)


def detect(res, known_events=(), known_distant=()):
    """Run the stacking detector on an AnalysisResult; returns (detections, StackTrace)."""
    s = res.settings
    p, vm = s.detection, s.velocity
    sts = [t for t in res.traces.values() if t.seismic is not None and t.cft is not None]
    if len(sts) < 2:
        return [], None
    t_start = min(t.seismic.starttime for t in sts)
    t_end = max(t.seismic.endtime for t in sts)
    n = int((t_end - t_start) * CF_RATE) + 1
    # trial sources extend well beyond the area, so that an event just outside it
    # is located outside (and then ignored) instead of being forced inside
    LA, LO = reg.grid(p.stack_step_deg, tuple(s.polygon), float(s.area_margin_km) + 300.0)
    st_lat = np.array([t.station.lat for t in sts])
    st_lon = np.array([t.station.lon for t in sts])
    D = haversine_km(LA[:, None], LO[:, None], st_lat[None, :], st_lon[None, :])
    use = D <= p.stack_max_dist_km
    sP = np.round(physics.p_time(D, 0.0, vm) * CF_RATE).astype(int)
    sS = np.round(physics.s_time(D, 0.0, vm) * CF_RATE).astype(int)
    pad = int(sS[use].max()) + 2 if use.any() else 2
    C = np.zeros((len(sts), n + pad))
    for i, t in enumerate(sts):
        C[i, :n] = _station_cf(t, t_start, n, p)
    _mask_known(C, sts, t_start, known_events, known_distant, p, vm)

    bmax = np.zeros(n)
    arg = np.zeros(n, int)
    for j in range(len(LA)):
        idx = np.flatnonzero(use[j])
        if len(idx) < 2:
            continue
        rows = (np.stack([C[i, sP[j, i]:sP[j, i] + n] for i in idx])
                + np.stack([C[i, sS[j, i]:sS[j, i] + n] for i in idx]))
        b = rows.sum(axis=0) - rows.max(axis=0)
        better = b > bmax
        bmax[better] = b[better]
        arg[better] = j

    med = float(np.median(bmax))
    mad = float(np.median(np.abs(bmax - med))) * 1.4826
    thr = max(p.stack_threshold, med + p.stack_mad_factor * mad)
    trace = StackTrace(t_start, CF_RATE, bmax.astype(np.float32), thr)

    # peaks, at least 90 s apart, strongest first
    cand = np.flatnonzero(bmax >= thr)
    peaks = []
    for k in cand[np.argsort(-bmax[cand])]:
        if all(abs(k - q) > 90 * CF_RATE for q in peaks):
            peaks.append(int(k))
    out = []
    for k in sorted(peaks):
        t0 = t_start + k / CF_RATE
        j = int(arg[k])
        # stations that support this source, and the source area at this time
        contrib, both = {}, 0
        for i in np.flatnonzero(use[j]):
            cp, cs = C[i, sP[j, i] + k], C[i, sS[j, i] + k]
            if max(cp, cs) >= SUPPORT:
                contrib[sts[i].station.id] = (float(cp + cs), "P" if cp >= cs else "S")
                both += cp >= SUPPORT and cs >= SUPPORT
        if not (len(contrib) >= 3 or (len(contrib) == 2 and both == 2)):
            continue
        bk = np.zeros(len(LA))
        for jj in range(len(LA)):
            idx = np.flatnonzero(use[jj])
            if len(idx) >= 2:
                v = C[idx, sP[jj, idx] + k] + C[idx, sS[jj, idx] + k]
                bk[jj] = v.sum() - v.max()
        area = bk >= 0.8 * bmax[k]
        err = float(max(np.max(haversine_km(LA[j], LO[j], LA[area], LO[area])), 10.0))
        if not reg.in_area(float(LA[j]), float(LO[j]), s.area_margin_km, s.polygon):
            continue
        if any(abs(e.origin_time - t0) < 60 and haversine_km(e.lat, e.lon, LA[j], LO[j]) < 200 + e.error_km
               for e in known_events):
            continue
        sub = max(int(area.sum()) // 800, 1)
        out.append(StackDetection(t0, float(LA[j]), float(LO[j]), float(bmax[k]), thr, err, contrib,
                                  [float(x) for x in LA[area][::sub]], [float(x) for x in LO[area][::sub]]))
    return out, trace
