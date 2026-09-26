"""Real data from public FDSN web services (needs ObsPy + internet).

* Raspberry Shake community network ("AM", data.raspberryshake.org)
  - RS1D/RS3D/RS4D/RS&BOOM: EHZ vertical geophone, HDF infrasound (RS&BOOM)
* EarthScope (IRIS) - global networks (II, IU, ...), temporary deployments
* GEOFON (GFZ) - GE network and partners
* USGS / EMSC event catalogues for cross-checking.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .config import DATA_CENTERS, INFRASOUND_CHANNELS, SEISMIC_CHANNEL_PREFERENCE
from .geo import haversine_km
from .models import CatalogEvent, Station, Waveform


def obspy_available() -> bool:
    try:
        import obspy  # noqa: F401
        return True
    except ImportError:
        return False


def _client(key, timeout=60):
    from obspy.clients.fdsn import Client
    return Client(key, timeout=timeout)


def discover_stations(center_keys, region, t_start, t_end, include_infrasound=True,
                      max_stations=40, log=print):
    """Find stations with a vertical seismic channel (and infrasound) in the region.

    Returns (stations, inventories) where inventories maps station id -> (client key, Inventory).
    """
    from obspy import UTCDateTime

    channels = ",".join(SEISMIC_CHANNEL_PREFERENCE + (INFRASOUND_CHANNELS if include_infrasound else []))
    found: dict[str, Station] = {}
    inventories = {}
    for key in center_keys:
        try:
            inv = _client(key).get_stations(
                minlatitude=region["min_lat"], maxlatitude=region["max_lat"],
                minlongitude=region["min_lon"], maxlongitude=region["max_lon"],
                channel=channels, starttime=UTCDateTime(t_start), endtime=UTCDateTime(t_end),
                level="response")
        except Exception as exc:  # no data / service down / no network
            log(f"{key}: station query failed ({exc.__class__.__name__}: {str(exc)[:120]})")
            continue
        n_before = len(found)
        for net in inv:
            for sta in net:
                sid = f"{net.code}.{sta.code}"
                if sid in found:
                    continue
                chans = {c.code: c for c in sta.channels}
                seis = next((c for c in SEISMIC_CHANNEL_PREFERENCE if c in chans), None)
                if seis is None:
                    continue
                infra = next((c for c in INFRASOUND_CHANNELS if c in chans), None)
                is_rs = key == "RASPISHAKE" or net.code == "AM"
                kind = ("Raspberry Shake & Boom" if infra else "Raspberry Shake") if is_rs else "Broadband"
                found[sid] = Station(
                    network=net.code, code=sta.code, lat=sta.latitude, lon=sta.longitude,
                    elevation=sta.elevation, location=chans[seis].location_code, channel=seis,
                    kind=kind, source=key, infrasound_channel=infra,
                    site=(sta.site.name or "") if sta.site else "")
                inventories[sid] = (key, inv)
        log(f"{key}: {len(found) - n_before} stations")
    stations = list(found.values())
    if len(stations) > max_stations:
        # keep the stations closest to the region centre, but spread out
        clat = (region["min_lat"] + region["max_lat"]) / 2
        clon = (region["min_lon"] + region["max_lon"]) / 2
        stations.sort(key=lambda s: float(haversine_km(clat, clon, s.lat, s.lon)))
        stations = stations[:max_stations]
    return stations, inventories


def _fetch_one(sta: Station, client, inv, t_start, t_end, target_fs=100.0):
    from obspy import UTCDateTime

    out = []
    chans = [(sta.channel, "VEL", "m/s")]
    if sta.infrasound_channel:
        chans.append((sta.infrasound_channel, "DEF", "Pa"))
    for cha, output, units in chans:
        st = client.get_waveforms(sta.network, sta.code, sta.location or "*", cha,
                                  UTCDateTime(t_start), UTCDateTime(t_end))
        if not len(st):
            continue
        st.merge(method=1, fill_value="interpolate")
        tr = st[0]
        fs = tr.stats.sampling_rate
        tr.detrend("linear")
        tr.remove_response(inventory=inv, output=output, water_level=60,
                           pre_filt=(0.2, 0.4, 0.40 * fs, 0.48 * fs))
        if fs > target_fs * 1.01:
            tr.resample(target_fs)
        out.append(Waveform(sta, cha, tr.stats.starttime.timestamp, tr.stats.sampling_rate,
                            tr.data.astype(np.float64), units))
    return out


def fetch_waveforms(stations, inventories, t_start, t_end, log=print, progress=None, workers=8):
    """Download and instrument-correct waveforms (m/s, Pa) in parallel."""
    waveforms = []
    done = 0
    clients = {}
    for key in {inventories[s.id][0] for s in stations}:
        try:
            clients[key] = _client(key, timeout=90)
        except Exception as exc:
            log(f"{key}: cannot connect ({exc.__class__.__name__})")
    stations = [s for s in stations if inventories[s.id][0] in clients]
    if not stations:
        return waveforms
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_one, s, clients[inventories[s.id][0]], inventories[s.id][1],
                          t_start, t_end): s for s in stations}
        for fut in as_completed(futs):
            s = futs[fut]
            done += 1
            try:
                w = fut.result()
                waveforms.extend(w)
                if not w:
                    log(f"{s.id}: no data")
            except Exception as exc:
                log(f"{s.id}: download failed ({exc.__class__.__name__}: {str(exc)[:100]})")
            if progress:
                progress(done / len(futs), f"Downloaded {done}/{len(futs)} stations")
    return waveforms


def fetch_catalog(region, t_start, t_end, log=print, local_min_mag=2.0, global_min_mag=5.0):
    """Earthquakes/explosions from USGS (+EMSC): regional small events + large global ones."""
    from obspy import UTCDateTime

    events = []
    queries = [
        ("USGS", dict(minlatitude=region["min_lat"] - 3, maxlatitude=region["max_lat"] + 3,
                      minlongitude=region["min_lon"] - 3, maxlongitude=region["max_lon"] + 3,
                      minmagnitude=local_min_mag)),
        ("USGS", dict(minmagnitude=global_min_mag)),
        ("EMSC", dict(minlatitude=region["min_lat"] - 3, maxlatitude=region["max_lat"] + 3,
                      minlongitude=region["min_lon"] - 3, maxlongitude=region["max_lon"] + 3,
                      minmagnitude=local_min_mag)),
    ]
    for key, q in queries:
        try:
            cat = _client(key).get_events(starttime=UTCDateTime(t_start - 3600),
                                          endtime=UTCDateTime(t_end), **q)
        except Exception as exc:
            if "No data" not in str(exc) and "204" not in str(exc):
                log(f"{key} catalogue: {exc.__class__.__name__}: {str(exc)[:100]}")
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
    # de-duplicate (same event from two agencies)
    events.sort(key=lambda c: c.time)
    unique = []
    for c in events:
        if not any(abs(c.time - u.time) < 20 and haversine_km(c.lat, c.lon, u.lat, u.lon) < 100 for u in unique):
            unique.append(c)
    return unique


def load_real_data(center_labels, region, t_start, t_end, include_infrasound=True,
                   max_stations=40, with_catalog=True, log=print, progress=None):
    """One call that does everything for the app."""
    keys = [DATA_CENTERS[c] for c in center_labels]
    stations, invs = discover_stations(keys, region, t_start, t_end, include_infrasound,
                                       max_stations, log)
    if not stations:
        return [], [], []
    waveforms = fetch_waveforms(stations, invs, t_start, t_end, log, progress)
    catalog = fetch_catalog(region, t_start, t_end, log) if with_catalog else []
    return stations, waveforms, catalog
