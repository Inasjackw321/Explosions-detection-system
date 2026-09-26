"""Physics formulas: travel times, magnitude, energy and explosive yield.

Every formula used by the app lives here so that the "Formulas" tab and the
processing code are guaranteed to be the same thing.
"""

from __future__ import annotations

import numpy as np

from .config import VelocityModel
from .geo import KM_PER_DEG

# ---------------------------------------------------------------------------
# Crustal travel times (1 layer over a half-space)
# ---------------------------------------------------------------------------


def direct_time(dist_km, depth_km, v):
    """Direct (Pg / Sg) wave: t = sqrt(d^2 + h^2) / v."""
    return np.sqrt(np.asarray(dist_km) ** 2 + np.asarray(depth_km) ** 2) / v


def head_wave_time(dist_km, depth_km, v1, v2, moho):
    """Moho head wave (Pn / Sn).

    t = d / v2 + (2H - h) * sqrt(1/v1^2 - 1/v2^2)

    Only exists beyond the critical distance x_c = (2H - h) * tan(i_c),
    where sin(i_c) = v1 / v2.  Returns +inf where it does not exist.
    """
    dist_km = np.asarray(dist_km, dtype=float)
    depth_km = np.minimum(np.asarray(depth_km, dtype=float), moho - 0.1)
    legs = 2 * moho - depth_km
    t = dist_km / v2 + legs * np.sqrt(1 / v1**2 - 1 / v2**2)
    ic = np.arcsin(v1 / v2)
    xc = legs * np.tan(ic)
    return np.where(dist_km >= xc, t, np.inf)


def p_time(dist_km, depth_km, vm: VelocityModel):
    """First-arriving P wave: min(Pg, Pn)."""
    return np.minimum(direct_time(dist_km, depth_km, vm.vp),
                      head_wave_time(dist_km, depth_km, vm.vp, vm.vpn, vm.moho))


def s_time(dist_km, depth_km, vm: VelocityModel):
    """First-arriving S wave: min(Sg, Sn)."""
    return np.minimum(direct_time(dist_km, depth_km, vm.vs),
                      head_wave_time(dist_km, depth_km, vm.vs, vm.vsn, vm.moho))


def air_time(dist_km, celerity_km_s):
    """Infrasound / air-blast arrival: t = d / c."""
    return np.asarray(dist_km) / celerity_km_s


def sound_speed_km_s(temp_c):
    """c = 331.3 * sqrt(1 + T/273.15) m/s (returned in km/s)."""
    return 0.3313 * np.sqrt(1.0 + np.asarray(temp_c) / 273.15)


def sp_distance_km(dt_sp, vp, vs):
    """Single-station distance from the S-P time: d = dt * vp*vs / (vp - vs)."""
    return dt_sp * vp * vs / (vp - vs)


# ---------------------------------------------------------------------------
# Teleseismic (distant) travel times - iasp91, surface focus.
# Columns: distance (deg), P time (s), P ray parameter (s/deg), S time (s)
# ---------------------------------------------------------------------------
_TELE = np.array([
    [0.5, 9.6, 19.17, 16.5], [1, 19.2, 19.17, 33.1], [2, 35.0, 13.75, 61.7],
    [3, 48.8, 13.75, 86.5], [4, 62.5, 13.75, 111.2], [5, 76.3, 13.74, 135.9],
    [6, 90.0, 13.74, 160.6], [8, 117.5, 13.72, 209.9], [10, 144.9, 13.70, 259.1],
    [12, 172.3, 13.67, 308.1], [15, 213.2, 13.63, 381.3], [18, 251.6, 12.33, 454.1],
    [20, 274.1, 10.90, 500.9], [25, 325.4, 9.10, 591.5], [30, 370.3, 8.85, 670.3],
    [35, 414.0, 8.62, 747.9], [40, 456.3, 8.30, 823.8], [45, 497.0, 7.96, 897.4],
    [50, 535.9, 7.60, 968.5], [55, 573.0, 7.24, 1037.0], [60, 608.3, 6.88, 1102.7],
    [65, 641.8, 6.51, 1165.7], [70, 673.4, 6.15, 1225.7], [75, 703.2, 5.78, 1282.9],
    [80, 731.2, 5.40, 1337.0], [85, 757.3, 5.02, 1388.1], [90, 781.3, 4.64, 1435.8],
    [95, 804.4, 4.55, 1480.1],
])


def tele_p_time(dist_deg, depth_km=0.0):
    """Approximate global P travel time (iasp91 table; depth correction ~h/8.5)."""
    return np.interp(dist_deg, _TELE[:, 0], _TELE[:, 1]) - np.asarray(depth_km) / 8.5


def tele_s_time(dist_deg, depth_km=0.0):
    return np.interp(dist_deg, _TELE[:, 0], _TELE[:, 3]) - np.asarray(depth_km) / 4.8


def distance_from_slowness(app_velocity_km_s):
    """Epicentral distance (deg) whose P ray parameter matches an apparent velocity.

    p [s/deg] = 111.19 / v_app.  Uses the far-distance branch (> 18 deg) where
    p decreases monotonically with distance.
    """
    p = KM_PER_DEG / app_velocity_km_s
    branch = _TELE[_TELE[:, 0] >= 18]
    # ray parameter decreases with distance -> reverse for np.interp
    return float(np.interp(p, branch[::-1, 2], branch[::-1, 0]))


# ---------------------------------------------------------------------------
# Magnitude
# ---------------------------------------------------------------------------
# Wood-Anderson torsion seismometer (IASPEI 2013): natural period 0.8 s,
# damping 0.7 -> two poles.  IASPEI's ML uses a WA-simulated record with
# static magnification 1 (the historical instrument magnified 2080x).
WA_POLES = np.array([-6.283 + 4.7124j, -6.283 - 4.7124j])
WA_HISTORICAL_GAIN = 2080.0


def wood_anderson_response(freqs, gain=1.0):
    """Complex WA displacement response H(s) = G * s^2 / ((s-p1)(s-p2)), s = i*2*pi*f."""
    s = 2j * np.pi * np.asarray(freqs)
    return gain * s**2 / ((s - WA_POLES[0]) * (s - WA_POLES[1]))


def velocity_to_wood_anderson(vel, fs):
    """Simulate a unit-magnification Wood-Anderson record (metres) from velocity (m/s).

    Done in the frequency domain: WA(f) = V(f) / (i*2*pi*f) * H(f).
    """
    n = len(vel)
    nfft = int(2 ** np.ceil(np.log2(max(n, 2) * 2)))
    spec = np.fft.rfft(vel - np.mean(vel), nfft)
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    s = 2j * np.pi * f
    s[0] = 1.0  # avoid 0/0, DC removed anyway
    h = wood_anderson_response(f) / s
    h[0] = 0.0
    return np.fft.irfft(spec * h, nfft)[:n]


def local_magnitude(amp_nm, hypo_km):
    """IASPEI / Hutton & Boore (1987) local magnitude.

    ML = log10(A) + 1.11 log10(R) + 0.00189 R - 2.09
    A = peak amplitude (nm) of the Wood-Anderson-simulated record with
        magnification 1, R = hypocentral distance in km.
    (Check: ML 3 at 100 km -> A = 480 nm, x2080 = 1 mm on a real WA: Richter's definition.)
    """
    r = np.maximum(np.asarray(hypo_km, dtype=float), 1.0)
    return np.log10(np.maximum(amp_nm, 1e-12)) + 1.11 * np.log10(r) + 0.00189 * r - 2.09


def wa_amplitude_for_ml(ml, hypo_km):
    """Inverse of local_magnitude: expected WA amplitude (nm)."""
    r = np.maximum(np.asarray(hypo_km, dtype=float), 1.0)
    return 10 ** (ml - 1.11 * np.log10(r) - 0.00189 * r + 2.09)


# ---------------------------------------------------------------------------
# Energy and yield
# ---------------------------------------------------------------------------
JOULES_PER_TON_TNT = 4.184e9


def seismic_energy_joules(magnitude):
    """Gutenberg-Richter energy: log10 E = 1.5 M + 4.8  (E in joules)."""
    return 10 ** (1.5 * np.asarray(magnitude) + 4.8)


def explosion_yield_tons(magnitude, coupling=1.0):
    """Explosive yield from magnitude.

    Fully coupled (underground, hard rock):  M = 4.45 + 0.75 log10(Y_kt)
      ->  Y_kt = 10 ** ((M - 4.45) / 0.75)
    Surface / air blasts couple only a small fraction of their energy into the
    ground, so the real yield is larger: Y = Y_coupled / coupling.
    Calibration: Beirut 2020 (~0.5-1.1 kt TNT) was ML ~3.3, which implies
    coupling ~0.05 for a surface blast.
    """
    y_kt = 10 ** ((np.asarray(magnitude) - 4.45) / 0.75)
    return y_kt * 1000.0 / coupling


SURFACE_BLAST_COUPLING = 0.05
