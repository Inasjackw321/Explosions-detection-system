"""Event location.

* Local/regional events: grid search over (lat, lon, depth) minimising the
  weighted travel-time residuals; the origin time is solved analytically.
* Distant events: plane-wave fit across the network -> apparent velocity and
  back-azimuth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import physics
from .config import VelocityModel
from .geo import haversine_km, local_xy_km

DEPTHS_COARSE = np.array([0.0, 3.0, 6.0, 10.0, 15.0, 20.0, 30.0])
DEPTHS_FINE = np.arange(0.0, 41.0, 1.0)
CHI2_95_2DOF = 5.99   # 95% confidence, 2 unknowns (lat, lon)
CHI2_95_1DOF = 3.84   # 95% confidence, 1 unknown (depth)


@dataclass
class Location:
    lat: float
    lon: float
    depth: float
    origin_time: float
    rms: float
    residuals: np.ndarray
    error_km: float
    depth_min: float
    depth_max: float
    at_edge: bool

    @property
    def depth_constrained(self) -> bool:
        return (self.depth_max - self.depth_min) < 15.0


def _travel_times(dist, depth, phases, vm):
    tp = physics.p_time(dist, depth, vm)
    ts = physics.s_time(dist, depth, vm)
    return np.where(phases == "S", ts, tp)


def _evaluate(lat_g, lon_g, depths, st_lat, st_lon, t_obs, phases, w, vm, robust=False):
    """Return chi2[depth, node], t0[depth, node] for a set of grid nodes.

    robust=True uses the median for the origin time and caps every squared
    normalised residual at 3^2, so a few wrong picks cannot drag the solution.
    """
    dist = haversine_km(lat_g[:, None], lon_g[:, None], st_lat[None, :], st_lon[None, :])
    chi2 = np.empty((len(depths), len(lat_g)))
    t0 = np.empty_like(chi2)
    for k, h in enumerate(depths):
        tt = _travel_times(dist, h, phases[None, :], vm)
        if robust:
            t0k = np.median(t_obs - tt, axis=1)
            r = t_obs - tt - t0k[:, None]
            chi2[k] = np.sum(np.minimum(w * r ** 2, 9.0), axis=1)
        else:
            # weighted mean gives the least-squares origin time analytically
            t0k = np.sum(w * (t_obs - tt), axis=1) / np.sum(w)
            r = t_obs - tt - t0k[:, None]
            chi2[k] = np.sum(w * r ** 2, axis=1)
        t0[k] = t0k
    return chi2, t0


def locate(st_lat, st_lon, t_obs, phases, sigmas, vm: VelocityModel, region, margin=3.0,
           robust=False):
    """Grid-search hypocentre.

    Minimises chi^2 = sum_i ((t_obs_i - t0 - T_i(x, y, z)) / sigma_i)^2 with
    t0 = weighted mean of (t_obs_i - T_i).
    """
    st_lat = np.asarray(st_lat, float)
    st_lon = np.asarray(st_lon, float)
    t_ref = float(np.min(t_obs))
    t_obs = np.asarray(t_obs, float) - t_ref
    phases = np.asarray(phases)
    w = 1.0 / np.asarray(sigmas, float) ** 2

    lat_min, lat_max = region["min_lat"] - margin, region["max_lat"] + margin
    lon_min, lon_max = region["min_lon"] - margin, region["max_lon"] + margin

    # --- coarse grid (0.2 deg) ---
    la = np.arange(lat_min, lat_max + 1e-9, 0.2)
    lo = np.arange(lon_min, lon_max + 1e-9, 0.2)
    LA, LO = [a.ravel() for a in np.meshgrid(la, lo, indexing="ij")]
    chi2, _ = _evaluate(LA, LO, DEPTHS_COARSE, st_lat, st_lon, t_obs, phases, w, vm, robust)
    k, j = np.unravel_index(np.argmin(chi2), chi2.shape)
    blat, blon = LA[j], LO[j]

    # --- fine grid (0.02 deg, +-1.2 deg, 1 km depth) ---
    la = np.arange(blat - 1.2, blat + 1.2 + 1e-9, 0.02)
    lo = np.arange(blon - 1.2, blon + 1.2 + 1e-9, 0.02)
    LA, LO = [a.ravel() for a in np.meshgrid(la, lo, indexing="ij")]
    chi2, t0 = _evaluate(LA, LO, DEPTHS_FINE, st_lat, st_lon, t_obs, phases, w, vm, robust)
    k, j = np.unravel_index(np.argmin(chi2), chi2.shape)
    best = chi2[k, j]
    lat, lon, depth = float(LA[j]), float(LO[j]), float(DEPTHS_FINE[k])

    # residuals at the optimum
    d = haversine_km(lat, lon, st_lat, st_lon)
    tt = _travel_times(d, depth, phases, vm)
    res = t_obs - t0[k, j] - tt
    n = len(t_obs)
    rms = float(np.sqrt(np.mean(res ** 2)))

    # Scale chi2 by the reduced chi2 when the fit is worse than the assumed pick
    # errors (so the confidence region reflects the real misfit).
    dof = max(n - 4, 1)
    scale = max(best / dof, 1.0)
    # 95% epicentre confidence region at the best depth
    inside = chi2[k] <= best + CHI2_95_2DOF * scale
    dist_in = haversine_km(lat, lon, LA[inside], LO[inside])
    err = float(max(np.max(dist_in) if dist_in.size else 0.0, 2.0))
    # depth confidence interval
    best_per_depth = chi2.min(axis=1)
    ok = best_per_depth <= best + CHI2_95_1DOF * scale
    dmin, dmax = float(DEPTHS_FINE[ok].min()), float(DEPTHS_FINE[ok].max())
    edge = bool(abs(lat - la[0]) < 0.03 or abs(lat - la[-1]) < 0.03
                or abs(lon - lo[0]) < 0.03 or abs(lon - lo[-1]) < 0.03
                or lat <= lat_min + 0.05 or lat >= lat_max - 0.05
                or lon <= lon_min + 0.05 or lon >= lon_max - 0.05)
    if edge:
        err = max(err, 150.0)
    return Location(lat, lon, depth, t_ref + float(t0[k, j]), rms, res, err, dmin, dmax, edge)


@dataclass
class PlaneWave:
    back_azimuth: float     # direction the waves come FROM (deg from north)
    app_velocity: float     # km/s across the network
    rms: float
    t_ref: float
    residuals: np.ndarray


def plane_wave_fit(st_lat, st_lon, t_obs):
    """Least-squares plane wave: t_i = t0 + px * x_i + py * y_i.

    Apparent velocity = 1/|p|; back-azimuth = direction of -p.
    """
    st_lat = np.asarray(st_lat, float)
    st_lon = np.asarray(st_lon, float)
    t = np.asarray(t_obs, float)
    lat0, lon0 = st_lat.mean(), st_lon.mean()
    x, y = local_xy_km(st_lat, st_lon, lat0, lon0)
    A = np.column_stack([np.ones_like(x), x, y])
    sol, *_ = np.linalg.lstsq(A, t - t.min(), rcond=None)
    t0, px, py = sol
    res = (t - t.min()) - A @ sol
    p = float(np.hypot(px, py))
    vapp = 1.0 / p if p > 1e-6 else np.inf
    baz = float((np.degrees(np.arctan2(-px, -py)) + 360.0) % 360.0)
    return PlaneWave(baz, vapp, float(np.sqrt(np.mean(res ** 2))), t.min() + t0, res)


def associate_grid(anchors, st_lat, st_lon, t_obs, sigma, vm: VelocityModel, region,
                   margin=3.0):
    """Phase-free association of a group of picks.

    One pick (the anchor) is assumed to be a first P arrival.  For every trial
    source (lat, lon, depth) the origin time then follows from the anchor,
        t0 = t_anchor - T_P(anchor station),
    and every other pick is explained as either P or S, whichever fits better:
        r_i = min(|t_i - t0 - T_P,i|, |t_i - t0 - T_S,i|)
    The misfit sum_i min((r_i / sigma_i)^2, 9) rewards explaining as many
    picks as possible.  Several anchors are tried (in case the first pick is
    a noise burst).  Returns (phases, inlier_mask, cost) for the best trial source.
    """
    st_lat = np.asarray(st_lat, float)
    st_lon = np.asarray(st_lon, float)
    t_all = np.asarray(t_obs, float)
    la = np.arange(region["min_lat"] - margin, region["max_lat"] + margin + 1e-9, 0.2)
    lo = np.arange(region["min_lon"] - margin, region["max_lon"] + margin + 1e-9, 0.2)
    LA, LO = [a.ravel() for a in np.meshgrid(la, lo, indexing="ij")]
    dist = haversine_km(LA[:, None], LO[:, None], st_lat[None, :], st_lon[None, :])
    tts = [(physics.p_time(dist, h, vm), physics.s_time(dist, h, vm)) for h in DEPTHS_COARSE]
    best = (np.inf, None)
    for a in anchors:
        t = t_all - t_all[a]
        for tp, ts in tts:
            t0 = -tp[:, a]
            rp = t[None, :] - t0[:, None] - tp
            rs = t[None, :] - t0[:, None] - ts
            cost = np.minimum(np.minimum((rp / sigma) ** 2, 9.0),
                              np.minimum((rs / (2 * sigma)) ** 2, 9.0)).sum(axis=1)
            j = int(np.argmin(cost))
            if cost[j] < best[0]:
                best = (cost[j], (rp[j], rs[j]))
    rp, rs = best[1]
    use_s = (rs / 2.0) ** 2 < rp ** 2
    phases = np.where(use_s, "S", "P")
    tol = max(3.0 * sigma, 3.0)
    inlier = np.where(use_s, np.abs(rs) <= 2 * tol, np.abs(rp) <= tol)
    return phases, inlier, float(best[0])


@dataclass
class DistantLocation:
    lat: float
    lon: float
    distance_deg: float     # from the network centre
    back_azimuth: float     # from the network centre towards the source
    origin_time: float
    rms: float
    residuals: np.ndarray


def locate_distant(st_lat, st_lon, t_obs, min_deg=5.0, max_deg=100.0):
    """Locate a far-away source with the global (iasp91) P travel-time table.

    Grid search over the globe: for each trial epicentre, T_i = T_P(Delta_i)
    and t0 = mean(t_i - T_i).  The wavefront curvature across the network
    fixes the distance, the arrival order fixes the direction.
    """
    from .geo import KM_PER_DEG, azimuth_deg

    st_lat = np.asarray(st_lat, float)
    st_lon = np.asarray(st_lon, float)
    t_ref = float(np.min(t_obs))
    t = np.asarray(t_obs, float) - t_ref
    lat0, lon0 = st_lat.mean(), st_lon.mean()

    def search(la, lo):
        LA, LO = [a.ravel() for a in np.meshgrid(la, lo, indexing="ij")]
        dc = haversine_km(lat0, lon0, LA, LO) / KM_PER_DEG
        keep = (dc >= min_deg) & (dc <= max_deg)
        LA, LO = LA[keep], LO[keep]
        d = haversine_km(LA[:, None], LO[:, None], st_lat[None, :], st_lon[None, :]) / KM_PER_DEG
        tt = physics.tele_p_time(d)
        t0 = np.mean(t[None, :] - tt, axis=1)
        r = t[None, :] - tt - t0[:, None]
        rms = np.sqrt(np.mean(r ** 2, axis=1))
        j = int(np.argmin(rms))
        return LA[j], LO[j], t0[j], rms[j], r[j]

    lat, lon, *_ = search(np.arange(-75, 80.1, 1.0), np.arange(-180, 180, 1.0))
    lat, lon, t0, rms, r = search(np.arange(lat - 2, lat + 2.01, 0.1), np.arange(lon - 2, lon + 2.01, 0.1))
    lon = (lon + 540) % 360 - 180
    return DistantLocation(float(lat), float(lon), float(haversine_km(lat0, lon0, lat, lon) / KM_PER_DEG),
                           float(azimuth_deg(lat0, lon0, lat, lon)), t_ref + float(t0), float(rms), r)
