"""A stand-in for ObsPy's FDSN Client that serves simulated recordings.

Lets the test suite exercise the real-data path (station search, response
removal, resampling, chunked monitoring) without internet access.
"""

from __future__ import annotations

import numpy as np
from obspy import Stream, Trace, UTCDateTime
from obspy.core.inventory import Channel, Inventory, Network, Site
from obspy.core.inventory import Station as InvStation
from obspy.core.inventory.response import Response

GAIN_SEIS = 4.0e8     # counts per m/s  (Raspberry Shake EHZ is ~3.9e8)
GAIN_INFRA = 5.6e4    # counts per Pa   (Raspberry Boom HDF order of magnitude)


class FakeClient:
    def __init__(self, waveforms):
        self.waveforms = waveforms
        self.requests = []

    def _inventory(self, level, network="*", station="*"):
        nets = {}
        for w in self.waveforms:
            s = w.station
            if station not in ("*", s.code):
                continue
            infra = w.units == "Pa"
            resp = Response.from_paz([], [], GAIN_INFRA if infra else GAIN_SEIS,
                                     input_units="PA" if infra else "M/S", output_units="COUNTS")
            ch = Channel(w.channel, "00", s.lat, s.lon, s.elevation, 0.0, sample_rate=w.fs,
                         response=resp if level == "response" else None)
            net = nets.setdefault(s.network, Network(s.network, stations=[]))
            sta = next((x for x in net.stations if x.code == s.code), None)
            if sta is None:
                sta = InvStation(s.code, s.lat, s.lon, s.elevation, site=Site(s.site))
                net.stations.append(sta)
            sta.channels.append(ch)
        return Inventory(networks=list(nets.values()), source="fake")

    def get_stations(self, network="*", station="*", level="station", **kw):
        return self._inventory(level, network, station)

    def get_waveforms(self, network, station, location, channel, t1, t2):
        self.requests.append((station, channel, float(t1), float(t2)))
        out = Stream()
        for w in self.waveforms:
            if w.station.code == station and w.channel == channel:
                i1 = max(w.index(float(t1)), 0)
                i2 = min(w.index(float(t2)), len(w.data))
                if i2 <= i1:
                    continue
                gain = GAIN_INFRA if w.units == "Pa" else GAIN_SEIS
                data = np.round(w.data[i1:i2] * gain + 1000).astype(np.int32)   # counts with offset
                out.append(Trace(data, header=dict(network=w.station.network, station=station,
                                                   location="00", channel=channel, sampling_rate=w.fs,
                                                   starttime=UTCDateTime(w.starttime + i1 / w.fs))))
        return out
