"""Explosion vs. earthquake discrimination.

Each physical clue is turned into an evidence score between -2 (looks like an
earthquake) and +2 (looks like an explosion), multiplied by a weight, summed
and turned into a probability with the logistic function:

    P(explosion) = 1 / (1 + exp(-(b + sum_i w_i * s_i)))
"""

from __future__ import annotations

import math

import numpy as np

from .models import CatalogEvent, DetectedEvent, Evidence
from .geo import haversine_km

PRIOR_BIAS = -0.3      # natural earthquakes are more common than explosions

WEIGHTS = {
    "P/S amplitude ratio": 1.2,
    "Source depth": 1.5,
    "First-motion polarity": 0.8,
    "Air-blast (infrasound) arrival": 1.3,
    "Catalogue match": 2.0,
    "Time of day": 0.3,
}


def logistic(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def ps_ratio_evidence(log_ps: list[float]) -> Evidence | None:
    if not log_ps:
        return None
    m = float(np.median(log_ps))
    score = float(np.clip((m - (-0.25)) / 0.2, -2, 2))
    return Evidence(
        "P/S amplitude ratio", f"log10(P/S) = {m:+.2f} (median of {len(log_ps)} stations)",
        score, WEIGHTS["P/S amplitude ratio"],
        "Explosions push the ground outward equally in all directions, making strong P waves "
        "and weak S waves (log P/S around 0 or above). Earthquakes are shear slip on a fault, "
        "so S waves are usually 2-5x bigger than P (log P/S around -0.4 to -0.6).")


def depth_evidence(ev: DetectedEvent) -> Evidence:
    if not ev.depth_constrained:
        return Evidence("Source depth", f"{ev.depth:.0f} km (poorly constrained: "
                        f"{ev.depth_min:.0f}-{ev.depth_max:.0f} km)", 0.0, WEIGHTS["Source depth"],
                        "The station geometry cannot pin down the depth, so depth is not used.")
    if ev.depth_min >= 6:
        score = -2.0
    elif ev.depth_max <= 3:
        score = 1.0
    else:
        score = float(np.clip(1.0 - (ev.depth - 2) / 3.0, -2, 1))
    return Evidence("Source depth", f"{ev.depth:.0f} km (95%: {ev.depth_min:.0f}-{ev.depth_max:.0f} km)",
                    score, WEIGHTS["Source depth"],
                    "Man-made explosions happen at or near the surface (< 2-3 km). "
                    "A source deeper than ~6 km can only be natural.")


def polarity_evidence(polarities: list[int]) -> Evidence | None:
    clear = [p for p in polarities if p != 0]
    if len(clear) < 3:
        return None
    up = sum(1 for p in clear if p > 0)
    frac = up / len(clear)
    conf = min(1.0, len(clear) / 6.0)
    if frac >= 0.9:
        score = 1.5 * conf
    elif frac <= 0.75:
        score = -1.5 * conf
    else:
        score = 0.0
    return Evidence("First-motion polarity", f"{up}/{len(clear)} stations moved UP first",
                    score, WEIGHTS["First-motion polarity"],
                    "An explosion pushes outward, so every station first moves UP (compression). "
                    "A fault slip pushes in two quadrants and pulls in the other two, so earthquakes "
                    "give a mix of up and down first motions.")


def air_evidence(n_infra: int, n_seismic_air: int, n_infra_sensors_in_range: int) -> Evidence:
    if n_infra >= 1 or n_seismic_air >= 3:
        score = 2.0 if n_infra >= 1 else 1.0
        meas = f"{n_infra} infrasound + {n_seismic_air} air-coupled seismic detections"
    elif n_infra_sensors_in_range >= 1:
        score, meas = -0.5, f"none on {n_infra_sensors_in_range} infrasound sensor(s) in range"
    else:
        score, meas = 0.0, "no infrasound sensor in range"
    return Evidence("Air-blast (infrasound) arrival", meas, score,
                    WEIGHTS["Air-blast (infrasound) arrival"],
                    "A surface or air explosion also sends a pressure wave through the atmosphere, "
                    "arriving much later at the speed of sound (~0.30 km/s). Earthquakes do not "
                    "produce this delayed air wave.")


def catalog_evidence(ev: DetectedEvent, catalog: list[CatalogEvent]) -> Evidence:
    best = None
    for c in catalog or []:
        if abs(c.time - ev.origin_time) <= 25 and haversine_km(c.lat, c.lon, ev.lat, ev.lon) <= 150:
            best = c
            break
    ev.catalog_match = best
    if best is None:
        return Evidence("Catalogue match", "not in the earthquake catalogue", 0.0,
                        WEIGHTS["Catalogue match"],
                        "Small events are often missing from catalogues, so no match is neutral.")
    t = best.event_type.lower()
    explo = any(k in t for k in ("explosion", "blast", "nuclear", "mining"))
    return Evidence("Catalogue match", f"{best.source}: M{best.magnitude:.1f} {best.event_type}",
                    2.0 if explo else -2.0, WEIGHTS["Catalogue match"],
                    "A seismological agency already reviewed this event and classified it.")


def time_of_day_evidence(ev: DetectedEvent) -> Evidence:
    h = (ev.origin_time / 3600.0 + ev.lon / 15.0) % 24
    day = 7 <= h <= 17.5
    return Evidence("Time of day", f"{int(h):02d}:{int((h % 1) * 60):02d} local solar time",
                    0.5 if day else -0.3, WEIGHTS["Time of day"],
                    "Quarry and construction blasts are done in working hours; earthquakes happen "
                    "at any hour. This is a weak clue only.")


def classify(ev: DetectedEvent, evidence: list[Evidence]):
    z = PRIOR_BIAS + sum(e.contribution for e in evidence)
    p = min(max(logistic(z), 0.01), 0.99)   # never claim certainty
    ev.evidence = evidence
    ev.p_explosion = p
    if p >= 0.7:
        ev.label = "Likely explosion"
    elif p <= 0.3:
        ev.label = "Likely earthquake"
    else:
        ev.label = "Uncertain"
    ranked = sorted([e for e in evidence if abs(e.contribution) > 0.05],
                    key=lambda e: -abs(e.contribution))
    pro = [e.name for e in ranked if e.contribution > 0][:3]
    con = [e.name for e in ranked if e.contribution < 0][:3]
    parts = []
    if pro:
        parts.append("explosion-like clues: " + ", ".join(pro))
    if con:
        parts.append("earthquake-like clues: " + ", ".join(con))
    ev.summary = f"{ev.label} ({p * 100:.0f}% explosion probability). " + "; ".join(parts) + "."
    return p
