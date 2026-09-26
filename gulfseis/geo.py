"""Spherical-Earth geometry helpers (all vectorised with NumPy)."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0
KM_PER_DEG = np.pi * EARTH_RADIUS_KM / 180.0  # ~111.19 km


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km).

    d = 2R * asin( sqrt( sin^2(dphi/2) + cos(phi1) cos(phi2) sin^2(dlambda/2) ) )
    """
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def azimuth_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 to point 2, degrees clockwise from north."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlmb = np.radians(np.asarray(lon2) - np.asarray(lon1))
    x = np.sin(dlmb) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dlmb)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def destination(lat, lon, azimuth, dist_km):
    """Point reached travelling dist_km from (lat, lon) along an azimuth."""
    d = dist_km / EARTH_RADIUS_KM
    az = np.radians(azimuth)
    p1, l1 = np.radians(lat), np.radians(lon)
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(az))
    l2 = l1 + np.arctan2(np.sin(az) * np.sin(d) * np.cos(p1), np.cos(d) - np.sin(p1) * np.sin(p2))
    return float(np.degrees(p2)), float((np.degrees(l2) + 540.0) % 360.0 - 180.0)


def local_xy_km(lats, lons, lat0, lon0):
    """Flat-Earth projection (east, north) in km around (lat0, lon0)."""
    x = (np.asarray(lons) - lon0) * KM_PER_DEG * np.cos(np.radians(lat0))
    y = (np.asarray(lats) - lat0) * KM_PER_DEG
    return x, y


def circle_polygon(lat, lon, radius_km, n=72):
    """Lat/lon outline of a circle of radius_km (for error ellipses on maps)."""
    lats, lons = [], []
    for az in np.linspace(0, 360, n):
        la, lo = destination(lat, lon, az, radius_km)
        lats.append(la)
        lons.append(lo)
    return lats, lons
