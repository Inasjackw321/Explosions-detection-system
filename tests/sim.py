"""Simulated recordings - used ONLY by the test suite (the app uses real data only).

Generates Raspberry-Shake-like and broadband seismograms (m/s) and
Raspberry Shake & Boom infrasound (Pa) at sites around the Gulf, containing
explosions, earthquakes, a distant earthquake and local noise bursts.
Network code "XX" marks every station as simulated.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from gulfseis import physics
from gulfseis.config import VelocityModel
from gulfseis.geo import azimuth_deg, haversine_km, KM_PER_DEG
from gulfseis.models import CatalogEvent, Station, Waveform

# code, site, lat, lon, kind, relative noise (cities are noisier)
SIM_SITES = [
    ("KUWT", "Kuwait City", 29.37, 47.98, "Raspberry Shake & Boom", 1.5),
    ("BSRA", "Basrah", 30.51, 47.81, "Raspberry Shake", 1.5),
    ("AHVZ", "Ahvaz", 31.32, 48.67, "Broadband", 1.0),
    ("YASJ", "Yasuj", 30.67, 51.59, "Raspberry Shake", 0.8),
    ("BUSH", "Bushehr", 28.97, 50.84, "Raspberry Shake & Boom", 1.0),
    ("SHRZ", "Shiraz", 29.59, 52.58, "Broadband", 0.8),
    ("LARR", "Lar", 27.68, 54.34, "Raspberry Shake", 0.8),
    ("BABS", "Bandar Abbas", 27.18, 56.27, "Broadband", 1.0),
    ("KRMN", "Kerman", 30.28, 57.08, "Broadband", 0.8),
    ("KISH", "Kish Island", 26.53, 53.98, "Raspberry Shake & Boom", 1.2),
    ("DUBI", "Dubai", 25.20, 55.27, "Raspberry Shake & Boom", 1.8),
    ("ABUD", "Abu Dhabi", 24.45, 54.38, "Raspberry Shake", 1.5),
    ("ALAN", "Al Ain", 24.21, 55.74, "Raspberry Shake", 0.8),
    ("SOHR", "Sohar", 24.35, 56.71, "Raspberry Shake", 1.0),
    ("DOHA", "Doha", 25.29, 51.53, "Raspberry Shake & Boom", 1.5),
    ("MANA", "Manama", 26.22, 50.59, "Raspberry Shake", 1.5),
    ("DAMM", "Dammam", 26.43, 50.10, "Raspberry Shake", 1.2),
    ("HOFF", "Al Hofuf", 25.38, 49.59, "Raspberry Shake", 0.9),
    ("RIYD", "Riyadh", 24.71, 46.68, "Broadband", 1.0),
]

SEISMIC_NOISE = 1.5e-7      # m/s rms in ~1-20 Hz for a quiet site
INFRA_NOISE = 0.03          # Pa rms (wind noise)
AIR_TO_GROUND = 1.2e-6      # m/s of ground motion per Pa of air pressure


@dataclass
class ScenarioEvent:
    kind: str            # "explosion", "earthquake" or "distant"
    t: float             # origin time, seconds after the start of the record
    lat: float
    lon: float
    depth: float
    magnitude: float     # ML for local events, mb for distant ones
    name: str = ""
    strike: float = 135.0


def default_scenario() -> list[ScenarioEvent]:
    return [
        ScenarioEvent("explosion", 120.0, 27.47, 52.61, 0.0, 2.9,
                      "Surface explosion on the south Iranian coast"),
        ScenarioEvent("earthquake", 560.0, 28.85, 52.55, 12.0, 3.7,
                      "Zagros earthquake (thrust belt)", strike=135.0),
        ScenarioEvent("distant", 890.0, 36.52, 70.95, 210.0, 5.7,
                      "Hindu Kush, Afghanistan"),
    ]


def default_start() -> float:
    return datetime(2026, 9, 26, 7, 0, 0, tzinfo=timezone.utc).timestamp()


def simulated_stations() -> list[Station]:
    out = []
    for code, site, lat, lon, kind, _ in SIM_SITES:
        bb = kind == "Broadband"
        out.append(Station(
            network="XX", code=code, lat=lat, lon=lon, channel="BHZ" if bb else "EHZ",
            kind=kind, source="Simulator", site=site,
            infrasound_channel="HDF" if "Boom" in kind else None,
        ))
    return out


# ---------------------------------------------------------------------------
def _shaped_noise(n, fs, rng, shape):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    x = np.fft.irfft(spec * shape(f), n)
    return x / (np.std(x) + 1e-30)


def _phase(fs, rng, fc, t_star, decay, polarity, n_total, lead_pulse=True):
    """One seismic phase: impulsive onset + band-limited coda, peak = 1."""
    n = n_total
    shape = lambda f: (f / (1 + (f / fc) ** 2)) * np.exp(-np.pi * f * t_star) * (f > 0.5)
    w = _shaped_noise(n, fs, rng, shape)
    t = np.arange(n) / fs
    fdom = min(max(fc, 1.0), 5.0, 0.4 * fs)   # onset pulse survives attenuation only < ~5 Hz
    rise = 0.5 / fdom
    env = np.where(t < rise, t / rise, np.exp(-(t - rise) / decay))
    x = w * env
    if lead_pulse:
        m = int(fs / (2 * fdom))
        pulse = np.zeros(n)
        pulse[:m] = np.sin(np.pi * np.arange(m) / m)
        x[:m] = 0.0
        x = x / (np.max(np.abs(x)) + 1e-30) * 0.7 + polarity * pulse
    return x / (np.max(np.abs(x)) + 1e-30)


def _add(dst, src, i0):
    if i0 >= len(dst) or i0 + len(src) <= 0:
        return
    a = max(i0, 0)
    b = min(i0 + len(src), len(dst))
    dst[a:b] += src[a - i0:b - i0]


def generate(events=None, duration=1500.0, noise_level=1.0, seed=7, start=None,
             vm: VelocityModel | None = None, glitches=4):
    """Return (stations, waveforms, catalog, events)."""
    vm = vm or VelocityModel()
    events = default_scenario() if events is None else events
    start = default_start() if start is None else start
    rng = np.random.default_rng(seed)
    stations = simulated_stations()
    noise_factor = {c: nf for c, _, _, _, _, nf in SIM_SITES}
    waveforms = []

    for sta in stations:
        fs = 40.0 if sta.channel == "BHZ" else 100.0
        n = int(duration * fs)
        sigma = SEISMIC_NOISE * noise_factor[sta.code] * noise_level
        seis = sigma * _shaped_noise(n, fs, rng, lambda f: (f > 0.3) / (1 + (f / 25) ** 4))
        seis += 1.5e-6 * _shaped_noise(n, fs, rng, lambda f: np.exp(-((f - 0.2) / 0.05) ** 2))  # microseism
        infra = None
        if sta.infrasound_channel:
            infra = INFRA_NOISE * noise_level * noise_factor[sta.code] * _shaped_noise(
                n, fs, rng, lambda f: (f > 0.2) / (1 + f) ** 1.0)

        for ev in events:
            dist = float(haversine_km(ev.lat, ev.lon, sta.lat, sta.lon))
            if ev.kind == "distant":
                _add_distant(seis, fs, rng, ev, dist, noise_level)
                continue
            _add_local(seis, fs, rng, ev, sta, dist, vm)
            if ev.kind == "explosion":
                _add_airblast(seis, infra, fs, rng, ev, dist)

        waveforms.append(Waveform(sta, sta.channel, start, fs, seis, "m/s"))
        if infra is not None:
            waveforms.append(Waveform(sta, sta.infrasound_channel, start, fs, infra, "Pa"))

    # a few single-station noise bursts (traffic, machinery, footsteps)
    seismic = [w for w in waveforms if not w.is_infrasound]
    for _ in range(glitches):
        w = seismic[rng.integers(len(seismic))]
        t0 = rng.uniform(60, duration - 60)
        m = int(3 * w.fs)
        burst = _phase(w.fs, rng, 12.0, 0.0, 0.6, 1, m, lead_pulse=False)
        _add(w.data, burst * SEISMIC_NOISE * 25 * noise_level, int(t0 * w.fs))

    catalog = [CatalogEvent(start + ev.t, ev.lat, ev.lon, ev.depth, ev.magnitude, "mb",
                            "earthquake", "Demo catalogue (USGS-like)", ev.name)
               for ev in events if ev.kind == "distant"]
    return stations, waveforms, catalog, events


def _add_local(seis, fs, rng, ev, sta, dist, vm):
    hypo = float(np.hypot(dist, ev.depth))
    tp = float(physics.p_time(dist, ev.depth, vm))
    ts = float(physics.s_time(dist, ev.depth, vm))
    fc = 10 ** (1.9 - 0.4 * ev.magnitude)
    if ev.kind == "explosion":
        fc *= 2.5                  # explosions: shorter source, richer in high frequencies
        polarity, p_over_s = 1, 1.2
    else:
        az = float(azimuth_deg(ev.lat, ev.lon, sta.lat, sta.lon))
        rad = np.sin(2 * np.radians(az - ev.strike))
        polarity = 1 if rad >= 0 else -1
        p_over_s = 0.22 * (0.5 + 0.5 * abs(rad)) + 0.04
    fc = min(fc, 0.4 * fs)
    t_star_p = tp / vm.q
    t_star_s = ts / vm.q
    n_p = int(min(ts - tp + 20, 120) * fs)
    n_s = int((40 + 0.15 * dist) * fs)
    s_wave = _phase(fs, rng, fc, t_star_s, 2.0 + 0.02 * dist, 1, n_s, lead_pulse=False)
    p_wave = _phase(fs, rng, fc, t_star_p, 1.0 + 0.01 * dist, polarity, n_p)
    # scale so that the S wave reproduces the requested ML on a Wood-Anderson
    target_m = physics.wa_amplitude_for_ml(ev.magnitude, hypo) * 1e-9
    wa_unit = np.max(np.abs(physics.velocity_to_wood_anderson(s_wave, fs)))
    scale = target_m / wa_unit
    # P amplitude relative to S measured in the high-frequency band
    _add(seis, s_wave * scale, int((ev.t + ts) * fs))
    _add(seis, p_wave * scale * p_over_s, int((ev.t + tp) * fs))


def _add_distant(seis, fs, rng, ev, dist, noise_level):
    deg = dist / KM_PER_DEG
    tp = float(physics.tele_p_time(deg, ev.depth))
    ts = float(physics.tele_s_time(deg, ev.depth))
    amp = 2.5e-6 * 10 ** (ev.magnitude - 5.7)
    p = _phase(fs, rng, 2.0, 0.6, 12.0, 1, int(90 * fs))
    s = _phase(fs, rng, 1.0, 1.5, 25.0, 1, int(200 * fs), lead_pulse=False)
    _add(seis, p * amp, int((ev.t + tp) * fs))
    _add(seis, s * amp * 1.8, int((ev.t + ts) * fs))


def _add_airblast(seis, infra, fs, rng, ev, dist):
    """Explosion air wave: N-shaped pressure pulse arriving at d / celerity."""
    if dist > 600:
        return
    celerity = rng.uniform(0.29, 0.33)
    ducting = 1.0 if rng.random() > 0.2 else 0.05     # some stations sit in a shadow zone
    p_amp = 20.0 * 10 ** (0.5 * (ev.magnitude - 3)) * (50.0 / max(dist, 5)) ** 1.2 * ducting
    t_arr = ev.t + dist / celerity
    dur = 0.4 + 0.002 * dist
    m = int((dur + 15) * fs)
    t = np.arange(m) / fs
    nwave = np.where(t < dur, 1 - 2 * t / dur, 0.0)
    tail = _phase(fs, rng, 4.0, 0.0, 3.0, 1, m, lead_pulse=False)
    pulse = nwave + 0.35 * np.roll(tail, int(dur * fs)) * (t >= dur)
    pulse /= np.max(np.abs(pulse))
    i0 = int(t_arr * fs)
    if infra is not None:
        _add(infra, pulse * p_amp, i0)
    ground = np.gradient(pulse) * fs / (2 * np.pi * 6.0)   # air-coupled ground motion (HF)
    _add(seis, ground / (np.max(np.abs(ground)) + 1e-30) * p_amp * AIR_TO_GROUND, i0)
