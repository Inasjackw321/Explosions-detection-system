"""Reference places around the Gulf, used to describe event locations in words."""

from __future__ import annotations

from .geo import azimuth_deg, haversine_km

PLACES = [
    ("Kuwait City", 29.37, 47.98), ("Basrah", 30.51, 47.81), ("Ahvaz", 31.32, 48.67),
    ("Nasiriyah", 31.05, 46.26), ("Hafar Al Batin", 28.43, 45.97), ("Yasuj", 30.67, 51.59),
    ("Gachsaran", 30.36, 50.80), ("Bushehr", 28.97, 50.84), ("Borazjan", 29.27, 51.22),
    ("Shiraz", 29.59, 52.58), ("Firuzabad", 28.84, 52.57), ("Kazerun", 29.62, 51.65),
    ("Asaluyeh", 27.48, 52.61), ("Bandar Kangan", 27.84, 52.06), ("Lamerd", 27.34, 53.18),
    ("Lar", 27.68, 54.34), ("Jahrom", 28.50, 53.56), ("Darab", 28.75, 54.54),
    ("Bandar Abbas", 27.18, 56.27), ("Qeshm", 26.95, 56.27), ("Kish Island", 26.53, 53.98),
    ("Sirjan", 29.45, 55.68), ("Kerman", 30.28, 57.08), ("Hajiabad", 28.31, 55.90),
    ("Minab", 27.15, 57.08), ("Rafsanjan", 30.41, 55.99), ("Dubai", 25.20, 55.27),
    ("Abu Dhabi", 24.45, 54.38), ("Al Ain", 24.21, 55.74), ("Ras Al-Khaimah", 25.79, 55.94),
    ("Fujairah", 25.12, 56.33), ("Sohar", 24.35, 56.71), ("Muscat", 23.59, 58.41),
    ("Doha", 25.29, 51.53), ("Manama", 26.22, 50.59), ("Dammam", 26.43, 50.10),
    ("Al Jubail", 27.01, 49.66), ("Al Hofuf", 25.38, 49.59), ("Riyadh", 24.71, 46.68),
    ("Al Kharj", 24.16, 47.31), ("Buraydah", 26.33, 43.97), ("Haradh", 24.14, 49.07),
]

_COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def compass(az: float) -> str:
    return _COMPASS[int((az % 360) / 22.5 + 0.5) % 16]


def describe(lat: float, lon: float) -> str:
    """e.g. '23 km SW of Bushehr'."""
    name, plat, plon = min(PLACES, key=lambda p: float(haversine_km(lat, lon, p[1], p[2])))
    d = float(haversine_km(plat, plon, lat, lon))
    if d < 5:
        return f"near {name}"
    return f"{d:.0f} km {compass(float(azimuth_deg(plat, plon, lat, lon)))} of {name}"
