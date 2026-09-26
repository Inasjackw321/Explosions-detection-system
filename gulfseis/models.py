"""Plain data containers shared by the whole package."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np


def utc(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def fmt_time(ts: float, with_date: bool = True) -> str:
    d = utc(ts)
    return d.strftime("%Y-%m-%d %H:%M:%S.") + f"{d.microsecond // 100000}" + " UTC" if with_date \
        else d.strftime("%H:%M:%S.") + f"{d.microsecond // 100000}"


@dataclass
class Station:
    network: str
    code: str
    lat: float
    lon: float
    elevation: float = 0.0
    location: str = ""
    channel: str = "EHZ"
    kind: str = "Raspberry Shake"      # "Raspberry Shake", "Raspberry Shake & Boom", "Broadband", "Simulated"
    source: str = ""                   # data centre
    infrasound_channel: str | None = None
    site: str = ""

    @property
    def id(self) -> str:
        return f"{self.network}.{self.code}"


@dataclass
class Waveform:
    station: Station
    channel: str
    starttime: float        # POSIX seconds, UTC
    fs: float               # sampling rate (Hz)
    data: np.ndarray        # m/s for seismic, Pa for infrasound
    units: str = "m/s"

    @property
    def is_infrasound(self) -> bool:
        return self.units == "Pa"

    @property
    def endtime(self) -> float:
        return self.starttime + (len(self.data) - 1) / self.fs

    def times(self) -> np.ndarray:
        return self.starttime + np.arange(len(self.data)) / self.fs

    def index(self, t: float) -> int:
        return int(round((t - self.starttime) * self.fs))

    def window(self, t1: float, t2: float) -> np.ndarray:
        i1 = max(self.index(t1), 0)
        i2 = min(self.index(t2), len(self.data))
        return self.data[i1:i2] if i2 > i1 else np.zeros(0)


@dataclass
class Pick:
    station: Station
    time: float              # POSIX seconds
    phase: str = "P"         # "P", "S", "Air", "?"
    snr: float = 0.0
    peak_ratio: float = 0.0  # max STA/LTA reached
    amplitude: float = 0.0   # peak filtered amplitude in the trigger (m/s or Pa)
    polarity: int = 0        # +1 up (compression), -1 down, 0 unclear
    duration: float = 0.0    # trigger length (s)
    channel: str = ""
    residual: float | None = None
    event_id: str | None = None


@dataclass
class CatalogEvent:
    time: float
    lat: float
    lon: float
    depth: float
    magnitude: float
    mag_type: str = ""
    event_type: str = "earthquake"
    source: str = ""
    description: str = ""


@dataclass
class Evidence:
    name: str
    measurement: str       # human readable measured value
    score: float           # -2 .. +2   (+ = explosion-like, - = earthquake-like)
    weight: float
    explanation: str

    @property
    def contribution(self) -> float:
        return self.score * self.weight


@dataclass
class DetectedEvent:
    id: str
    kind: str                       # "local" or "distant"
    origin_time: float
    lat: float
    lon: float
    depth: float = 0.0
    depth_min: float = 0.0
    depth_max: float = 0.0
    depth_constrained: bool = False
    error_km: float = 0.0
    rms: float = 0.0
    picks: list = field(default_factory=list)          # associated Pick objects (P & S)
    air_picks: list = field(default_factory=list)      # detected air-blast arrivals
    ml: float | None = None
    ml_stations: list = field(default_factory=list)    # dicts: station, dist, amp_nm, ml
    evidence: list = field(default_factory=list)
    p_explosion: float = 0.5
    label: str = "Uncertain"
    summary: str = ""
    # distant events
    back_azimuth: float | None = None
    app_velocity: float | None = None
    distance_deg: float | None = None
    catalog_match: CatalogEvent | None = None
    notes: list = field(default_factory=list)
    phases: list = field(default_factory=list)       # trial phase labels (association only)
    residuals: list = field(default_factory=list)
    # "network" (>= 4 stations, full location), "small" (2-3 stations, location = area),
    # "acoustic" (air-pressure sensors), "single" (one seismo-acoustic station), "distant"
    tier: str = "network"
    in_region: bool = True
    feasible_lat: list = field(default_factory=list)   # possible source area (small events)
    feasible_lon: list = field(default_factory=list)
    ring: tuple | None = None                          # (station, d_min_km, d_max_km) for single-station

    @property
    def stations(self) -> list:
        return sorted({p.station.id for p in self.picks})

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    @property
    def energy_j(self) -> float | None:
        from .physics import seismic_energy_joules
        return None if self.ml is None else float(seismic_energy_joules(self.ml))

    @property
    def icon(self) -> str:
        return {"Likely explosion": "💥", "Possible explosion": "💥", "Likely earthquake": "🌍",
                "Possible earthquake": "🌍", "Distant earthquake": "🌐"}.get(self.label, "❓")
