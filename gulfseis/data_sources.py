"""Real data from public FDSN web services (needs ObsPy + internet).

* Raspberry Shake community network ("AM", data.raspberryshake.org)
  - RS1D/RS3D/RS4D: EHZ vertical geophone; RS&BOOM: EHZ + HDF infrasound;
    RBOOM: HDF only
* EarthScope (IRIS) - global and regional networks
* GEOFON (GFZ) - GE network and partners
* USGS / EMSC event catalogues for cross-checking.

Stations are searched in the box around the monitored polygon and kept when
they are inside it or within `station_buffer_km` of its outline.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np

from . import region as reg
from .config import DATA_CENTERS, INFRASOUND_CHANNELS, SEISMIC_CHANNEL_PREFERENCE
from .geo import KM_PER_DEG, haversine_km
from .models import CatalogEvent, Station, Waveform

TARGET_FS = 50.0      # Hz; enough for 1-20 Hz explosions and keeps long windows light


def obspy_available() -> bool:
    try:
        import obspy  # noqa: F401
        return True
    except ImportError:
        return False


def _client(key, timeout=120):
    from obspy.clients.fdsn import Client
    return Client(key, timeout=timeout)


@dataclass
class StationStatus:
    station: Station
    dist_to_area_km: float = 0.0
    ok: bool = False
    channels: list = field(default_factory=list)
    error: str = ""
    seconds_requested: float = 0.0
    seconds_with_data: float = 0.0

    @property
    def availability(self) -> float:
        """Fraction of the requested time that had data."""
        return self.seconds_with_data / self.seconds_requested if self.seconds_requested else 0.0


class DataFetcher:
    """Finds stations once, then downloads any time window (thread-safe caches)."""

    def __init__(self, center_labels, settings, include_infrasound=True, log=print):
        self.keys = [DATA_CENTERS[c] for c in center_labels]
        self.settings = settings
        self.include_infrasound = include_infrasound
        self.log = log
        self.clients = {}
        self.stations: list[Station] = []
        self.status: dict[str, StationStatus] = {}
        self._responses = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- stations
    def discover(self, t_start, t_end, max_stations=150):
        from obspy import UTCDateTime

        s = self.settings
        pad = s.station_buffer_km / KM_PER_DEG + 0.5
        box = reg.bbox(s.polygon, pad)
        chans = SEISMIC_CHANNEL_PREFERENCE + (INFRASOUND_CHANNELS if self.include_infrasound else [])
        found: dict[str, Station] = {}
        for key in self.keys:
            try:
                client = self.clients.get(key) or _client(key)
                self.clients[key] = client
                inv = client.get_stations(
                    network="AM" if key == "RASPISHAKE" else "*",
                    minlatitude=box["min_lat"], maxlatitude=box["max_lat"],
                    minlongitude=box["min_lon"], maxlongitude=box["max_lon"],
                    channel=",".join(chans), starttime=UTCDateTime(t_start),
                    endtime=UTCDateTime(t_end), level="channel")
            except Exception as exc:
                self.log(f"{key}: station search failed - {_short(exc)}")
                continue
            n0 = len(found)
            for net in inv:
                for sta in net:
                    sid = f"{net.code}.{sta.code}"
                    if sid in found:
                        continue
                    codes = {}
                    for c in sta.channels:      # prefer location "00" / "" and the first listed
                        codes.setdefault(c.code, c)
                    seis = next((c for c in SEISMIC_CHANNEL_PREFERENCE if c in codes), None)
                    infra = next((c for c in INFRASOUND_CHANNELS if c in codes), None) \
                        if self.include_infrasound else None
                    if seis is None and infra is None:
                        continue
                    d = reg.distance_km(sta.latitude, sta.longitude, s.polygon)
                    if d > s.station_buffer_km:
                        continue
                    is_rs = key == "RASPISHAKE" or net.code == "AM"
                    if is_rs:
                        kind = ("Raspberry Shake & Boom" if infra and seis else
                                "Raspberry Boom (infrasound only)" if infra else "Raspberry Shake")
                    else:
                        kind = "Broadband"
                    ch = codes[seis or infra]
                    found[sid] = Station(
                        network=net.code, code=sta.code, lat=sta.latitude, lon=sta.longitude,
                        elevation=sta.elevation, location=ch.location_code, channel=seis or "",
                        kind=kind, source=key, infrasound_channel=infra,
                        site=(sta.site.name or "") if sta.site else "")
                    self.status[sid] = StationStatus(found[sid], d)
            self.log(f"{key}: {len(found) - n0} stations in or near the area")
        stations = sorted(found.values(), key=lambda x: self.status[x.id].dist_to_area_km)
        if len(stations) > max_stations:
            self.log(f"Using the {max_stations} stations closest to the area (of {len(stations)}).")
            stations = stations[:max_stations]
        self.stations = stations
        return stations

    # ---------------------------------------------------------------- waveforms
    def _response(self, sta: Station, t1, t2):
        from obspy import UTCDateTime

        with self._lock:
            inv = self._responses.get(sta.id)
        if inv is None:
            chans = ",".join(c for c in (sta.channel, sta.infrasound_channel) if c)
            inv = self.clients[sta.source].get_stations(
                network=sta.network, station=sta.code, location=sta.location or "*", channel=chans,
                starttime=UTCDateTime(t1), endtime=UTCDateTime(t2), level="response")
            with self._lock:
                self._responses[sta.id] = inv
        return inv

    def _fetch_one(self, sta: Station, t1, t2):
        from obspy import UTCDateTime

        client = self.clients[sta.source]
        inv = self._response(sta, t1, t2)
        out, avail = [], []
        for cha, output, units in ((sta.channel, "VEL", "m/s"), (sta.infrasound_channel, "DEF", "Pa")):
            if not cha:
                continue
            st = client.get_waveforms(sta.network, sta.code, sta.location or "*", cha,
                                      UTCDateTime(t1), UTCDateTime(t2))
            if not len(st):
                continue
            fs = st[0].stats.sampling_rate
            n_have = sum(tr.stats.npts for tr in st)
            avail.append(min(n_have / max((t2 - t1) * fs, 1), 1.0))
            st.merge(method=1, fill_value="interpolate")   # bridge short gaps
            tr = st[0]
            tr.detrend("demean")
            tr.remove_response(inventory=inv, output=output, water_level=60,
                               pre_filt=(0.1, 0.3, 0.40 * fs, 0.48 * fs))
            if fs > TARGET_FS * 1.01:
                tr.filter("lowpass", freq=0.4 * TARGET_FS, corners=8, zerophase=True)
                factor = int(round(fs / TARGET_FS))
                if abs(fs / factor - TARGET_FS) < 1e-6:
                    tr.decimate(factor, no_filter=True)
                else:
                    tr.resample(TARGET_FS, no_filter=True)
            out.append(Waveform(sta, cha, tr.stats.starttime.timestamp, tr.stats.sampling_rate,
                                tr.data.astype(np.float64), units))
        return out, (float(np.mean(avail)) if avail else 0.0)

    def fetch(self, t1, t2, progress=None, workers=8):
        """Download + instrument-correct all stations for [t1, t2] (m/s and Pa)."""
        waveforms = []
        stations = [s for s in self.stations if s.source in self.clients]
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self._fetch_one, s, t1, t2): s for s in stations}
            for fut in as_completed(futs):
                s = futs[fut]
                stat = self.status[s.id]
                done += 1
                stat.seconds_requested += t2 - t1
                try:
                    w, a = fut.result()
                    waveforms.extend(w)
                    stat.seconds_with_data += a * (t2 - t1)
                    stat.channels = sorted({*stat.channels, *(x.channel for x in w)})
                    if w:
                        stat.ok = True
                        if stat.error == "no data in this time window":
                            stat.error = ""
                    elif not stat.ok:
                        stat.error = "no data in this time window"
                except Exception as exc:
                    if not stat.ok:
                        stat.error = _short(exc)
                if progress:
                    progress(done / max(len(futs), 1), f"Downloaded {done}/{len(futs)} stations")
        return waveforms


def _short(exc) -> str:
    msg = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    name = exc.__class__.__name__
    if "No data" in msg or "204" in msg or name == "FDSNNoDataException":
        return "no data available"
    return f"{name}: {msg[:140]}"


def fetch_catalog(settings, t_start, t_end, log=print, local_min_mag=1.5, global_min_mag=5.0):
    """Earthquakes/explosions from USGS and EMSC: small events near the area + large global ones."""
    from obspy import UTCDateTime

    box = reg.bbox(settings.polygon, 3.0)
    near = dict(minlatitude=box["min_lat"], maxlatitude=box["max_lat"],
                minlongitude=box["min_lon"], maxlongitude=box["max_lon"], minmagnitude=local_min_mag)
    events = []
    for key, q in (("USGS", near), ("USGS", dict(minmagnitude=global_min_mag)), ("EMSC", near)):
        try:
            cat = _client(key, 60).get_events(starttime=UTCDateTime(t_start - 3600),
                                              endtime=UTCDateTime(t_end), **q)
        except Exception as exc:
            if _short(exc) != "no data available":
                log(f"{key} catalogue: {_short(exc)}")
            continue
        for e in cat:
            o = e.preferred_origin() or (e.origins[0] if e.origins else None)
            m = e.preferred_magnitude() or (e.magnitudes[0] if e.magnitudes else None)
            if o is None:
                continue
            desc = e.event_descriptions[0].text if e.event_descriptions else ""
            events.append(CatalogEvent(
                o.time.timestamp, o.latitude, o.longitude, (o.depth or 0) / 1000.0,
                m.mag if m else float("nan"), m.magnitude_type if m else "",
                str(e.event_type or "earthquake"), key, desc))
    events.sort(key=lambda c: c.time)
    unique = []
    for c in events:
        if not any(abs(c.time - u.time) < 20 and haversine_km(c.lat, c.lon, u.lat, u.lon) < 100 for u in unique):
            unique.append(c)
    return unique
