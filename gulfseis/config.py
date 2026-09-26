"""Default settings: region, velocity model, detection parameters, data centres."""

from __future__ import annotations

from dataclasses import dataclass, field

# Persian Gulf / Arabian Gulf study area (matches the map: Basrah -> Oman,
# Riyadh -> Kerman).  Degrees.
DEFAULT_REGION = {"min_lat": 22.0, "max_lat": 33.0, "min_lon": 45.0, "max_lon": 60.0}

# FDSN data centres that serve stations in / around the Gulf.  Keys are the
# ObsPy short names.
DATA_CENTERS = {
    "Raspberry Shake (AM citizen network)": "RASPISHAKE",
    "EarthScope / IRIS (global networks)": "EARTHSCOPE",
    "GEOFON (GFZ Potsdam)": "GEOFON",
}

# Vertical seismic channels in order of preference, and the Raspberry
# Shake & Boom infrasound (pressure) channel.
SEISMIC_CHANNEL_PREFERENCE = ["HHZ", "BHZ", "EHZ", "SHZ"]
INFRASOUND_CHANNELS = ["HDF", "BDF"]


@dataclass
class VelocityModel:
    """One crustal layer over a mantle half-space + the atmosphere.

    Values are typical for the Arabian platform / Zagros (crust ~42 km).
    """

    vp: float = 6.10          # crustal P velocity (km/s)
    vs: float = 3.52          # crustal S velocity (km/s)
    moho: float = 42.0        # crustal thickness (km)
    vpn: float = 8.05         # upper-mantle P velocity, Pn (km/s)
    vsn: float = 4.60         # upper-mantle S velocity, Sn (km/s)
    air_temp_c: float = 30.0  # near-surface air temperature for sound speed
    q: float = 300.0          # crustal quality factor (attenuation)

    @property
    def c_air(self) -> float:
        """Speed of sound in km/s: c = 331.3 * sqrt(1 + T/273.15) m/s."""
        return 0.3313 * (1.0 + self.air_temp_c / 273.15) ** 0.5


@dataclass
class DetectionParams:
    freqmin: float = 1.5        # bandpass low corner (Hz)
    freqmax: float = 10.0       # bandpass high corner (Hz)
    sta: float = 1.0            # short-term average window (s)
    lta: float = 20.0           # long-term average window (s)
    trigger_on: float = 4.0     # STA/LTA ratio that starts a trigger
    trigger_off: float = 1.5    # STA/LTA ratio that ends a trigger
    min_stations: int = 4       # stations needed to declare an event (3 = no error check)
    pick_sigma: float = 0.6     # assumed P pick uncertainty (s)
    max_rms: float = 2.0        # max RMS travel-time residual for a good location (s)
    infrasound_band: tuple = (1.0, 8.0)
    celerity_min: float = 0.26  # slowest plausible infrasound celerity (km/s)
    celerity_max: float = 0.36  # fastest plausible infrasound celerity (km/s)
    max_air_range_km: float = 450.0  # search for air-blasts up to this range


@dataclass
class Settings:
    region: dict = field(default_factory=lambda: dict(DEFAULT_REGION))
    velocity: VelocityModel = field(default_factory=VelocityModel)
    detection: DetectionParams = field(default_factory=DetectionParams)
