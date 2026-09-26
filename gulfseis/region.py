"""The monitored area: the Persian Gulf and its coasts (traced from the user's map).

The polygon covers all of Kuwait, the head of the Gulf (Basrah, Faw,
Abadan, Bandar Mahshahr), runs down the Iranian coast to the Strait of Hormuz
and the Gulf of Oman (Sohar), back around Musandam and along the UAE, Qatar,
Bahrain and Saudi coasts to Kuwait.  Events up to `margin_km` outside the
outline also count as "in the area" (the outline was drawn by hand).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .geo import KM_PER_DEG, haversine_km

# (lat, lon) vertices, clockwise from the Kuwait / Saudi / Iraq border
GULF_POLYGON = [
    (28.53, 46.60), (29.10, 46.50), (30.10, 47.00), (30.75, 47.40), (31.00, 47.90),
    (30.75, 48.50), (30.65, 49.20), (30.45, 49.80), (30.25, 50.02),
    (30.20, 50.48), (29.98, 50.92), (29.64, 51.25), (28.97, 51.58), (28.59, 51.80),
    (28.20, 52.24), (27.71, 52.67), (27.08, 53.11), (26.75, 53.39), (26.78, 53.99),
    (26.88, 54.43), (27.15, 55.20), (27.32, 55.85), (27.30, 56.68), (27.03, 56.84),
    (26.64, 56.95), (26.24, 57.28), (25.75, 57.94), (25.05, 57.88), (24.55, 57.55),
    (24.10, 57.11), (24.75, 56.73), (25.45, 56.51), (25.75, 56.35), (26.05, 56.02),
    (26.14, 55.74), (25.65, 55.52), (25.05, 55.31), (24.45, 55.20), (23.95, 54.98),
    (23.80, 53.99), (23.80, 52.78), (23.85, 51.80), (24.35, 50.92), (24.85, 50.48),
    (25.35, 50.04), (25.75, 49.71), (26.05, 49.39), (26.64, 49.17), (27.42, 48.84),
    (28.20, 48.51), (28.53, 48.40),
]


def bbox(poly=GULF_POLYGON, pad_deg=0.0):
    lats = [p[0] for p in poly]
    lons = [p[1] for p in poly]
    return {"min_lat": min(lats) - pad_deg, "max_lat": max(lats) + pad_deg,
            "min_lon": min(lons) - pad_deg, "max_lon": max(lons) + pad_deg}


def contains(lat, lon, poly=GULF_POLYGON):
    """Point-in-polygon (ray casting), vectorised over lat/lon arrays."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    inside = np.zeros(np.broadcast(lat, lon).shape, bool)
    n = len(poly)
    for i in range(n):
        y1, x1 = poly[i]
        y2, x2 = poly[(i + 1) % n]
        crosses = (y1 > lat) != (y2 > lat)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (lon < x_at)
    return inside


@lru_cache(maxsize=4)
def _edge_points(poly=tuple(GULF_POLYGON), step_km=5.0):
    pts = []
    n = len(poly)
    for i in range(n):
        (a1, o1), (a2, o2) = poly[i], poly[(i + 1) % n]
        k = max(int(float(haversine_km(a1, o1, a2, o2)) / step_km), 1)
        for f in np.linspace(0, 1, k, endpoint=False):
            pts.append((a1 + f * (a2 - a1), o1 + f * (o2 - o1)))
    return np.array(pts)


def distance_km(lat, lon, poly=GULF_POLYGON):
    """0 inside the area, otherwise the distance (km) to its outline."""
    if contains(lat, lon, poly):
        return 0.0
    e = _edge_points(tuple(poly))
    return float(np.min(haversine_km(lat, lon, e[:, 0], e[:, 1])))


def in_area(lat, lon, margin_km=0.0, poly=GULF_POLYGON) -> bool:
    """Inside the outline, or within margin_km of it."""
    return bool(contains(lat, lon, poly)) or (margin_km > 0 and distance_km(lat, lon, poly) <= margin_km)


@lru_cache(maxsize=8)
def grid(step_deg=0.05, poly=tuple(GULF_POLYGON), margin_km=0.0):
    """Grid nodes (lat, lon arrays) inside the area (plus margin_km around it)."""
    pad = margin_km / KM_PER_DEG
    b = bbox(list(poly), pad)
    la = np.arange(b["min_lat"], b["max_lat"] + 1e-9, step_deg)
    lo = np.arange(b["min_lon"], b["max_lon"] + 1e-9, step_deg)
    LA, LO = [a.ravel() for a in np.meshgrid(la, lo, indexing="ij")]
    m = contains(LA, LO, list(poly))
    if margin_km > 0:
        e = _edge_points(tuple(poly))
        near = np.zeros_like(m)
        for k in range(0, len(LA), 4000):         # chunked to bound memory
            d = haversine_km(LA[k:k + 4000, None], LO[k:k + 4000, None], e[None, :, 0], e[None, :, 1])
            near[k:k + 4000] = d.min(axis=1) <= margin_km
        m |= near
    return LA[m], LO[m]


def area_km2(poly=GULF_POLYGON):
    la, lo = grid(0.05, tuple(poly))
    return float(len(la) * (0.05 * KM_PER_DEG) ** 2 * np.cos(np.radians(np.mean(la))))
