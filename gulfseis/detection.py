"""Signal processing & single-station detection.

* Butterworth band-pass filtering
* STA/LTA trigger (energy ratio)
* AIC onset picker (Maeda, 1985)
* First-motion polarity
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from .config import DetectionParams
from .models import Pick, Waveform


def bandpass(data, fs, fmin, fmax, order=4, zerophase=True):
    """Detrend, taper and Butterworth band-pass filter."""
    x = signal.detrend(np.asarray(data, dtype=float), type="linear")
    n = len(x)
    if n < 10:
        return x
    taper = signal.windows.tukey(n, alpha=min(0.05, 20 * fs / n))
    x = x * taper
    fmax = min(fmax, 0.45 * fs)
    if fmin >= fmax:
        return x
    sos = signal.butter(order, [fmin, fmax], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x) if zerophase else signal.sosfilt(sos, x)


def sta_lta(x, fs, sta_s, lta_s):
    """Classic STA/LTA on signal energy, LTA window directly before the STA window.

        STA(i) = 1/Ns * sum_{k=i-Ns+1..i} x_k^2
        LTA(i) = 1/Nl * sum_{k=i-Ns-Nl+1..i-Ns} x_k^2
        R(i)   = STA(i) / LTA(i)
    """
    ns = max(int(round(sta_s * fs)), 1)
    nl = max(int(round(lta_s * fs)), ns + 1)
    e = np.asarray(x, dtype=float) ** 2
    n = len(e)
    ratio = np.zeros(n)
    if n <= ns + nl:
        return ratio
    c = np.concatenate([[0.0], np.cumsum(e)])
    sta = np.zeros(n)
    sta[ns - 1:] = (c[ns:] - c[:-ns]) / ns
    lta = np.zeros(n)
    # LTA ending ns samples before i
    lta[ns + nl - 1:] = (c[nl:n - ns + 1] - c[:n - ns - nl + 1]) / nl
    floor = 0.05 * np.median(lta[ns + nl - 1:]) + 1e-30
    valid = np.arange(n) >= ns + nl - 1
    ratio[valid] = sta[valid] / np.maximum(lta[valid], floor)
    return ratio


def trigger_onsets(cft, on, off, min_samples=1):
    """Return [(i_on, i_off)] where cft rises above `on` until it falls below `off`."""
    above_on = cft >= on
    out = []
    i = 0
    n = len(cft)
    starts = np.flatnonzero(above_on[1:] & ~above_on[:-1]) + 1
    if n and above_on[0]:
        starts = np.concatenate([[0], starts])
    below_off = np.flatnonzero(cft < off)
    for s in starts:
        if s < i:
            continue
        k = np.searchsorted(below_off, s)
        e = below_off[k] if k < len(below_off) else n - 1
        if e - s >= min_samples:
            out.append((int(s), int(e)))
        i = e
    return out


def aic_pick(x):
    """Akaike Information Criterion onset picker (Maeda 1985).

        AIC(k) = k * log(var(x[0..k])) + (N - k - 1) * log(var(x[k+1..N-1]))

    The onset is where AIC is minimum (best split into 'noise' and 'signal').
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 10:
        return 0
    c1 = np.cumsum(x)
    c2 = np.cumsum(x * x)
    k = np.arange(1, n - 1)
    var1 = c2[k - 1] / k - (c1[k - 1] / k) ** 2
    m = n - k
    var2 = (c2[-1] - c2[k - 1]) / m - ((c1[-1] - c1[k - 1]) / m) ** 2
    eps = 1e-30
    aic = k * np.log(np.maximum(var1, eps)) + (n - k - 1) * np.log(np.maximum(var2, eps))
    # ignore the edges (variance of 1-2 samples is meaningless)
    edge = max(2, n // 50)
    aic[:edge] = np.inf
    aic[-edge:] = np.inf
    return int(k[np.argmin(aic)])


def first_motion(x_causal, i_pick, fs, noise_s=2.0, look_s=0.5, k=4.0):
    """Polarity of the first clear motion after the pick (+1 up, -1 down, 0 unclear)."""
    i0 = max(i_pick - int(noise_s * fs), 0)
    noise = x_causal[i0:i_pick]
    if len(noise) < 5:
        return 0
    sigma = np.std(noise)
    seg = x_causal[i_pick:i_pick + int(look_s * fs)]
    idx = np.flatnonzero(np.abs(seg) > k * sigma)
    if not len(idx):
        return 0
    return int(np.sign(seg[idx[0]]))


def detect_station(wf: Waveform, p: DetectionParams, fmin=None, fmax=None):
    """Run STA/LTA on one waveform and return (picks, filtered, cft)."""
    fs = wf.fs
    fmin = p.freqmin if fmin is None else fmin
    fmax = p.freqmax if fmax is None else fmax
    filt = bandpass(wf.data, fs, fmin, fmax)
    causal = bandpass(wf.data, fs, fmin, fmax, zerophase=False)
    cft = sta_lta(filt, fs, p.sta, p.lta)
    picks = []
    edge = int(3 * fs)
    for i_on, i_off in trigger_onsets(cft, p.trigger_on, p.trigger_off, int(0.3 * fs)):
        if i_on < edge or i_on > len(filt) - edge:
            continue
        # refine onset with AIC in a window around the trigger
        a = max(i_on - int(2.5 * fs), 0)
        b = min(i_on + int(1.0 * fs), len(filt))
        i_pick = a + aic_pick(filt[a:b])
        noise = filt[max(i_pick - int(p.lta * fs), 0):i_pick]
        sig = filt[i_pick:i_off + 1]
        if not len(sig) or not len(noise):
            continue
        noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-30
        amp = float(np.max(np.abs(sig)))
        picks.append(Pick(
            station=wf.station, time=wf.starttime + i_pick / fs, phase="?",
            snr=amp / noise_rms, peak_ratio=float(np.max(cft[i_on:i_off + 1])),
            amplitude=amp, polarity=first_motion(causal, i_pick, fs),
            duration=(i_off - i_on) / fs, channel=wf.channel,
        ))
    return picks, filt, cft
