"""GulfSeis - explosion & noise monitor for the Persian Gulf (real data only).

Run with:   python run_app.py        (or: streamlit run app.py)
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone
from datetime import time as dtime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from gulfseis import data_sources, monitor, physics
from gulfseis import region as reg
from gulfseis.config import DATA_CENTERS, DetectionParams, Settings, VelocityModel
from gulfseis.discrimination import PRIOR_BIAS
from gulfseis.geo import KM_PER_DEG, circle_polygon, haversine_km
from gulfseis.models import DetectedEvent
from gulfseis.places import PLACES, describe

st.set_page_config(page_title="GulfSeis - Explosion Monitor", page_icon="💥", layout="wide")

EVENT_COLORS = {"Likely explosion": "#eb6834", "Possible explosion": "#f0a078",
                "Likely earthquake": "#2a78d6", "Possible earthquake": "#86b6ef",
                "Distant earthquake": "#1baf7a", "Uncertain": "#8a8984"}
PHASE_COLORS = {"P": "#2a78d6", "S": "#eb6834", "Air": "#1baf7a"}
AREA_COLOR = "#2bd12b"
STATION_ON, STATION_OFF = "#3d3c38", "#b5b3ab"
TIMEZONES = {"UTC": 0.0, "Kuwait · Saudi · Qatar · Bahrain · Iraq (UTC+3)": 3.0,
             "Iran (UTC+3:30)": 3.5, "UAE · Oman (UTC+4)": 4.0}
EXPLOSIVE = ("Likely explosion", "Possible explosion")


# ---------------------------------------------------------------------------
# Time helpers (everything is stored in UTC, shown in the chosen time zone)
# ---------------------------------------------------------------------------
def tz_offset() -> float:
    return TIMEZONES.get(st.session_state.get("tz", "UTC"), 0.0)


def tz_label() -> str:
    off = tz_offset()
    if off == 0:
        return "UTC"
    h, m = int(off), int(round((off % 1) * 60))
    return f"UTC+{h}" + (f":{m:02d}" if m else "")


def tfmt(ts: float, with_date: bool = True) -> str:
    d = datetime.fromtimestamp(ts + tz_offset() * 3600, tz=timezone.utc)
    s = d.strftime("%Y-%m-%d %H:%M:%S" if with_date else "%H:%M:%S") + f".{d.microsecond // 100000}"
    return f"{s} {tz_label()}" if with_date else s


def local_to_utc(d: date, t: dtime) -> float:
    return datetime.combine(d, t, tzinfo=timezone.utc).timestamp() - tz_offset() * 3600


def local_dt(ts: float) -> datetime:
    """Naive datetime in the display time zone (for plot axes)."""
    return datetime.fromtimestamp(ts + tz_offset() * 3600, timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar():
    sb = st.sidebar
    sb.title("💥 GulfSeis")
    sb.caption("Explosions vs noise in the Persian Gulf area, from public seismometers and "
               "infrasound microphones (Raspberry Shake & Boom, EarthScope, GEOFON).")
    cfg = {}
    sb.header("1 · When")
    sb.selectbox("Time zone for input and display", list(TIMEZONES), key="tz")
    mode = sb.radio("Period", ["Recent hours", "Time range", "Check a known event"], key="mode",
                    help="'Check a known event' looks closely at one moment: enter when (and roughly "
                         "where) something happened and the app shows what every station recorded.")
    now = time.time()
    loc_now = datetime.fromtimestamp(now + tz_offset() * 3600, tz=timezone.utc)
    cfg["mode"] = mode
    if mode == "Recent hours":
        hours = sb.select_slider("Last …", [1, 2, 3, 6, 12, 24], value=3, format_func=lambda h: f"{h} h")
        cfg["t2"] = now - 180
        cfg["t1"] = cfg["t2"] - hours * 3600
        cfg["auto"] = sb.checkbox("Auto-refresh", False)
        if cfg["auto"]:
            cfg["auto_min"] = sb.slider("Refresh every (minutes)", 5, 60, 10)
    elif mode == "Time range":
        d = sb.date_input("Start date", loc_now.date())
        t = sb.time_input("Start time", dtime(max(loc_now.hour - 3, 0), 0), step=300)
        hours = sb.slider("Length (hours)", 0.5, 24.0, 3.0, 0.5)
        cfg["t1"] = local_to_utc(d, t)
        cfg["t2"] = min(cfg["t1"] + hours * 3600, now - 120)
    else:
        d = sb.date_input("Date", loc_now.date())
        t = sb.time_input("Approximate time", dtime(loc_now.hour, 0), step=60)
        names = [p[0] for p in PLACES]
        where = sb.selectbox("Roughly where?", ["Unknown"] + names + ["Custom coordinates"],
                             index=1 + names.index("Kuwait City"))
        if where == "Custom coordinates":
            lat = sb.number_input("Latitude", 20.0, 35.0, 29.37, 0.01)
            lon = sb.number_input("Longitude", 44.0, 62.0, 47.98, 0.01)
        elif where == "Unknown":
            lat = lon = None
        else:
            _, lat, lon = next(p for p in PLACES if p[0] == where)
        before = sb.slider("Minutes before", 2, 30, 10)
        after = sb.slider("Minutes after", 5, 60, 30,
                          help="Air-blast waves travel ~18 km per minute, so 30 min covers ~500 km.")
        t0 = local_to_utc(d, t)
        cfg["focus"] = (lat, lon, t0)
        cfg["t1"], cfg["t2"] = t0 - before * 60, min(t0 + after * 60, now - 60)

    sb.header("2 · Stations")
    if not data_sources.obspy_available():
        sb.error("ObsPy is required: `pip install obspy`")
    cfg["centers"] = sb.multiselect("Data centres", list(DATA_CENTERS), default=list(DATA_CENTERS))
    cfg["infrasound"] = sb.checkbox("Use infrasound (Raspberry Shake & Boom)", True)
    s = Settings()
    s.station_buffer_km = sb.slider("Also use stations up to … km outside the area", 0, 400, 150, 25)
    s.area_margin_km = sb.slider("Count events up to … km outside the outline", 0, 200, 50, 10,
                                 help="The outline is hand-drawn; events just beyond it (e.g. inland "
                                      "Kuwait, southern Iraq, the Iranian coast) are still reported.")
    cfg["max_stations"] = sb.slider("Max stations", 5, 150, 80)
    cfg["catalog"] = sb.checkbox("Cross-check with USGS/EMSC catalogues", True)

    sb.header("3 · Detection")
    p = DetectionParams()
    default_sens = "High (catch small events)" if mode == "Check a known event" else "Normal"
    sens = sb.select_slider("Sensitivity", ["Low (fewer false alarms)", "Normal", "High (catch small events)"],
                            value=default_sens, key=f"sens_{mode}")
    if sens.startswith("High"):
        p.trigger_on, p.infra_trigger_on, p.min_snr_small, p.small_min_stations = 3.5, 4.0, 4.0, 2
        p.stack_threshold, p.stack_mad_factor, p.strong_signal_snr = 1.5, 6.0, 10.0
    elif sens.startswith("Low"):
        p.trigger_on, p.infra_trigger_on, p.min_snr_small, p.small_min_stations = 5.0, 6.0, 8.0, 3
        p.stack_threshold, p.stack_mad_factor, p.strong_signal_snr = 3.0, 10.0, 25.0
    with sb.expander("Advanced trigger settings"):
        p.freqmin, p.freqmax = st.slider("Band-pass filter (Hz)", 0.5, 20.0, (p.freqmin, p.freqmax), 0.5)
        p.sta = st.slider("STA window (s)", 0.2, 5.0, p.sta, 0.1)
        p.lta = st.slider("LTA window (s)", 5.0, 120.0, p.lta, 5.0)
        p.trigger_on = st.slider("Seismic trigger ON (STA/LTA)", 2.0, 10.0, p.trigger_on, 0.25)
        p.infra_trigger_on = st.slider("Infrasound trigger ON (STA/LTA)", 2.0, 12.0, p.infra_trigger_on, 0.25)
        p.min_stations = st.slider("Stations for a fully located event", 4, 8, p.min_stations)
        p.small_min_stations = st.slider("Stations for a small detection", 2, 3, p.small_min_stations)
        p.min_snr_small = st.slider("Min signal/noise for 2-station detections", 3.0, 20.0, p.min_snr_small, 0.5)
        p.max_trigger_rate = st.slider("Max triggers per station per hour", 5, 200, int(p.max_trigger_rate), 5,
                                       help="Very noisy stations (traffic, machinery) keep only their "
                                            "strongest triggers.")
        p.stack_enabled = st.checkbox("Network stacking detector", p.stack_enabled,
                                      help="Adds up all stations along predicted travel times, so events "
                                           "too weak to trigger enough stations are still found.")
        p.stack_threshold = st.slider("Stacking: minimum brightness", 0.5, 6.0, p.stack_threshold, 0.25)
        p.stack_mad_factor = st.slider("Stacking: × background spread", 3.0, 15.0, p.stack_mad_factor, 0.5)
        p.strong_signal_snr = st.slider("List single-station signals stronger than (signal/noise)",
                                        5.0, 50.0, p.strong_signal_snr, 1.0)
    vm = VelocityModel()
    with sb.expander("Earth & air model"):
        vm.vp = st.number_input("Crust P speed Vp (km/s)", 5.0, 7.0, vm.vp, 0.05)
        vm.vs = st.number_input("Crust S speed Vs (km/s)", 2.8, 4.2, vm.vs, 0.05)
        vm.moho = st.number_input("Crust thickness (km)", 25.0, 60.0, vm.moho, 1.0)
        vm.vpn = st.number_input("Mantle P speed Pn (km/s)", 7.5, 8.5, vm.vpn, 0.05)
        vm.vsn = st.number_input("Mantle S speed Sn (km/s)", 4.0, 5.0, vm.vsn, 0.05)
        vm.air_temp_c = st.slider("Air temperature (°C)", -10, 50, int(vm.air_temp_c))
        st.caption(f"Speed of sound: {vm.c_air * 1000:.0f} m/s")
    s.velocity, s.detection = vm, p
    cfg["settings"] = s
    run = sb.button("▶ Run", type="primary", width="stretch")
    return cfg, run


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run_analysis(cfg):
    s: Settings = cfg["settings"]
    log = []
    st.session_state.log = log
    st.session_state.result = None
    if not data_sources.obspy_available():
        st.error("ObsPy is not installed. Run `pip install -r requirements.txt`.")
        return None
    if not cfg["centers"]:
        st.error("Choose at least one data centre.")
        return None
    if cfg["t2"] <= cfg["t1"]:
        st.error("The chosen time is in the future.")
        return None
    bar = st.progress(0.0, "Searching for stations…")
    fetcher = data_sources.DataFetcher(cfg["centers"], s, cfg["infrasound"], log=log.append)
    stations = fetcher.discover(cfg["t1"], cfg["t2"], cfg["max_stations"])
    if not stations:
        bar.empty()
        st.error("No stations found in or near the area (see the log below). Check the internet "
                 "connection, or allow stations farther outside the area.")
        return None
    catalog = []
    if cfg["catalog"]:
        bar.progress(0.03, "Reading USGS/EMSC catalogues…")
        catalog = data_sources.fetch_catalog(s, cfg["t1"], cfg["t2"], log=log.append)
    R = monitor.run(fetcher, cfg["t1"], cfg["t2"], s, catalog,
                    progress=lambda f, t: bar.progress(0.05 + 0.95 * min(f, 1.0), t),
                    focus=cfg.get("focus"))
    R.messages = log + R.messages
    bar.empty()
    if not any(x.ok for x in R.status.values()):
        st.error("Stations were found but none returned data for this period (see the Stations tab log).")
    st.session_state.result = R
    st.session_state.ran_at = time.time()
    return R


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def envelope(x, y, n=2500):
    """Min/max decimation so long traces plot quickly without losing peaks."""
    if len(y) <= 2 * n:
        return x, y
    k = len(y) // n
    m = k * n
    yy = y[:m].reshape(n, k)
    xx = x[:m].reshape(n, k)
    return np.repeat(xx[:, 0], 2), np.column_stack([yy.min(1), yy.max(1)]).ravel()


def metric(col, label, value, sub=None):
    col.metric(label, value)
    if sub:
        col.caption(sub)


def where_text(ev):
    if ev.kind == "distant":
        if ev.catalog_match and ev.catalog_match.description:
            return ev.catalog_match.description
        return f"{ev.distance_deg:.0f}° away (direction {ev.back_azimuth:.0f}°)"
    if ev.tier == "single" and ev.ring:
        return f"{ev.ring[1]:.0f}–{ev.ring[2]:.0f} km from {ev.ring[0].code} ({ev.ring[0].site or ev.ring[0].id})"
    return describe(ev.lat, ev.lon)


def event_title(ev):
    return f"{ev.icon} {ev.id} · {ev.label} · {where_text(ev)}"


def event_label(ev):
    return f"{ev.id} – {ev.label} – {tfmt(ev.origin_time)}"


def mag_text(ev):
    if ev.kind == "distant":
        return f"M{ev.catalog_match.magnitude:.1f}" if ev.catalog_match else "—"
    return f"ML {ev.ml:.1f}" if ev.ml is not None else "—"


def fmt_energy(j):
    for unit, f in (("TJ", 1e12), ("GJ", 1e9), ("MJ", 1e6), ("kJ", 1e3)):
        if j >= f:
            return f"{j / f:.1f} {unit}"
    return f"{j:.0f} J"


def fmt_tons(t):
    if t >= 1000:
        return f"{t / 1000:.1f} kt"
    if t >= 1:
        return f"{t:.1f} t"
    return f"{t * 1000:.0f} kg"


TIER_TEXT = {"network": "located by the seismic network", "small": "2–3 seismic stations (area only)",
             "stack": "network stacking (weak signals added up; area only)",
             "acoustic": "located from air-pressure (infrasound) arrivals",
             "single": "one station: ground wave + air wave", "distant": "distant earthquake"}


def reference_event(R):
    """The user's 'known event' as a pseudo-event, to draw predicted arrivals."""
    lat, lon, t0 = R.focus
    return DetectedEvent("REF", "local", t0, lat, lon, label="Your reference point")


# ---------------------------------------------------------------------------
# Header + event cards
# ---------------------------------------------------------------------------
def header(R):
    st.title("GulfSeis · Explosion & Noise Monitor")
    n_ok = sum(1 for s in R.status.values() if s.ok)
    st.caption(f"{tfmt(R.t_start)} → {tfmt(R.t_end, False)} · {n_ok} of {len(R.status)} stations "
               f"returned data · monitored area ≈ {reg.area_km2(R.settings.polygon):,.0f} km²")
    ins = R.in_region
    hours = max((R.t_end - R.t_start) / 3600, 0.01)
    c = st.columns(6)
    c[0].metric("Stations with data", f"{n_ok} / {len(R.status)}")
    c[1].metric("💥 Explosions", sum(e.label in EXPLOSIVE for e in ins),
                help="Likely + possible explosions inside the monitored area.")
    c[2].metric("🌍 Earthquakes", sum("earthquake" in e.label.lower() for e in ins))
    c[3].metric("❓ Uncertain", sum(e.label == "Uncertain" for e in ins))
    c[4].metric("🔇 Noise triggers", len(R.noise_picks),
                help=f"Single-station triggers that match no event: traffic, machinery, wind, "
                     f"people near the sensor ({len(R.noise_picks) / hours:.0f} per hour in total).")
    c[5].metric("🌐 Outside area", len(R.outside))
    if R.focus:
        focus_card(R)
    if not ins:
        st.info("No events inside the monitored area in this period. Every trigger was classified "
                "as local noise (see the *Timeline & noise* and *Stations & data* tabs).")
    for ev in ins:
        event_card(ev)
    strong = [x for x in R.single_signals if reg.in_area(x["station"].lat, x["station"].lon,
                                                           R.settings.station_buffer_km, R.settings.polygon)]
    if strong:
        with st.expander(f"⚠️ {len(strong)} clear signal(s) seen by only ONE station (unconfirmed)"):
            st.caption("Very strong signals, or a P wave followed by an S wave, that no other station confirms, "
                       "so they cannot be located. "
                       "They may be a nearby event that only this station recorded (sparse coverage), or "
                       "very local noise (vehicle, machinery, door). If a second, later arrival (possible S "
                       "wave) was seen, the S–P time gives a distance (dashed ring on the map).")
            st.dataframe(pd.DataFrame([{
                "Time": tfmt(x["time"]), "Station": x["station"].id, "Site": x["station"].site,
                "Signal/noise": round(x["snr"], 1), "Peak (µm/s)": round(x["peak_um_s"], 3),
                "Duration (s)": round(x["duration_s"], 1),
                "Second arrival": "yes (P+S pair)" if x.get("pair") else ("weak" if x["sp_s"] else "no"),
                "S–P (s)": None if x["sp_s"] is None else round(x["sp_s"], 1),
                "Possible distance (km)": None if x["dist_km"] is None else round(x["dist_km"])}
                for x in strong]), hide_index=True, width="stretch")
    if R.outside:
        with st.expander(f"{len(R.outside)} event(s) outside the monitored area"):
            for ev in R.outside:
                st.markdown(f"**{event_title(ev)}** – {tfmt(ev.origin_time)} – {mag_text(ev)}  \n{ev.summary}")


def event_card(ev):
    with st.container(border=True):
        a, b, c, d = st.columns([3.2, 1.5, 1.1, 1.7])
        a.markdown(f"**{event_title(ev)}**")
        a.caption(ev.summary)
        b.markdown(f"**Time**  \n{tfmt(ev.origin_time)}")
        c.markdown(f"**Size**  \n{mag_text(ev)}")
        loc = f"{ev.lat:.2f}°N {ev.lon:.2f}°E ± {ev.error_km:.0f} km" if ev.tier != "single" else "direction unknown"
        d.markdown(f"**Location**  \n{loc}  \n_{TIER_TEXT.get(ev.tier, '')}_")
        st.progress(ev.p_explosion, f"Explosion probability {ev.p_explosion * 100:.0f}% "
                                    "(0% = earthquake-like, 100% = explosion-like)")


def focus_card(R):
    lat, lon, t0 = R.focus
    with st.container(border=True):
        st.markdown(f"**🔎 Your event: {tfmt(t0)}" + (f", near {describe(lat, lon)}**" if lat is not None else "**"))
        near = [e for e in R.events if abs(e.origin_time - t0) <= 900
                and (lat is None or haversine_km(lat, lon, e.lat, e.lon) <= 150 + e.error_km)]
        singles = [x for x in R.single_signals if abs(x["time"] - t0) <= 900
                   and (lat is None or haversine_km(lat, lon, x["station"].lat, x["station"].lon) <= 300)]
        if near:
            for e in near:
                st.success(f"Found: **{event_title(e)}** at {tfmt(e.origin_time, False)} "
                           f"({e.origin_time - t0:+.0f} s from your time).")
        elif singles:
            for x in singles:
                dist = f", possible distance {x['dist_km']:.0f} km (S–P {x['sp_s']:.1f} s)" if x["dist_km"] else ""
                st.info(f"Only one station recorded a clear signal: **{x['station'].id}** "
                        f"({x['station'].site}) at {tfmt(x['time'], False)}, signal/noise {x['snr']:.0f}{dist}. "
                        "No other station confirmed it, so it cannot be located or classified with confidence.")
        else:
            st.warning("Nothing was detected close to this time and place. The table shows what each "
                       "station recorded around the moment the waves should have arrived (±3 min, since "
                       "your time is approximate). Signal/noise below ~2 means the station did not see it. "
                       "Check the time zone, try a wider window, and look at the Waveforms tab.")
        if lat is not None and R.traces is not None:
            st.dataframe(focus_table(R), hide_index=True, width="stretch")


def focus_table(R):
    lat, lon, t0 = R.focus
    p, vm = R.settings.detection, R.settings.velocity
    rows = []
    items = sorted(R.traces.items(), key=lambda kv: float(haversine_km(lat, lon, kv[1].station.lat, kv[1].station.lon)))
    for sid, stt in items:
        sta = stt.station
        d = float(haversine_km(lat, lon, sta.lat, sta.lon))
        row = {"Station": sid, "Site": sta.site, "Distance (km)": round(d)}
        if stt.seismic is not None:
            w = stt.seismic
            tp = t0 + float(physics.p_time(d, 0, vm))
            f = stt.filtered[max(w.index(tp - 180), 0):max(w.index(tp + 180), 0)]
            n = stt.filtered[max(w.index(tp - 480), 0):max(w.index(tp - 200), 0)]
            if len(f) and len(n) > 10:
                row["Ground: signal/noise"] = round(float(np.max(np.abs(f)) / (3 * np.std(n) + 1e-30)), 1)
                row["Ground: peak (µm/s)"] = round(float(np.max(np.abs(f))) * 1e6, 3)
        if stt.infrasound is not None:
            iw = stt.infrasound
            t_lo, t_hi = t0 + d / p.celerity_max - 180, t0 + d / p.celerity_min + 180
            f = stt.infra_filtered[max(iw.index(t_lo), 0):max(iw.index(t_hi), 0)]
            n = stt.infra_filtered[max(iw.index(t_lo - 300), 0):max(iw.index(t_lo), 0)]
            if len(f) and len(n) > 10:
                row["Air: signal/noise"] = round(float(np.max(np.abs(f)) / (3 * np.std(n) + 1e-30)), 1)
                row["Air: peak (Pa)"] = round(float(np.max(np.abs(f))), 3)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
def map_tab(R):
    s = R.settings
    wide = st.radio("View", ["Monitored area", "Wide (includes distant events)"], horizontal=True,
                    key="map_view").startswith("Wide")
    show_out = st.checkbox("Show events outside the area (faded)", value=True, key="map_out")
    fig = go.Figure()
    poly = s.polygon + [s.polygon[0]]
    fig.add_trace(go.Scattergeo(lat=[p[0] for p in poly], lon=[p[1] for p in poly], mode="lines",
                                line=dict(width=3, color=AREA_COLOR), name="Monitored area", hoverinfo="skip"))
    events = [e for e in R.events if e.in_region or show_out]
    for ev in events:
        color = EVENT_COLORS.get(ev.label, "#8a8984")
        if ev.feasible_lat and ev.tier in ("small", "acoustic", "single", "stack"):
            fig.add_trace(go.Scattergeo(lat=ev.feasible_lat, lon=ev.feasible_lon, mode="markers",
                                        marker=dict(size=4, color=color, opacity=0.25), hoverinfo="skip",
                                        showlegend=False))
        if ev.ring is not None:
            sta, d1, d2 = ev.ring
            for r_km in (d1, d2):
                la, lo = circle_polygon(sta.lat, sta.lon, r_km, 120)
                fig.add_trace(go.Scattergeo(lat=la, lon=lo, mode="lines", hoverinfo="skip", showlegend=False,
                                            line=dict(width=1.5, color=color, dash="dash")))
        elif ev.kind == "local":
            el, eo = circle_polygon(ev.lat, ev.lon, max(ev.error_km, 1.0))
            fig.add_trace(go.Scattergeo(lat=el, lon=eo, mode="lines", hoverinfo="skip", showlegend=False,
                                        line=dict(width=1.5, color=color, dash="dot")))
            lats, lons = [], []
            for q in ev.picks + ev.air_picks:
                lats += [ev.lat, q.station.lat, None]
                lons += [ev.lon, q.station.lon, None]
            fig.add_trace(go.Scattergeo(lat=lats, lon=lons, mode="lines", hoverinfo="skip", showlegend=False,
                                        line=dict(width=1, color=color), opacity=0.35))

    used = {q.station.id for e in R.events for q in e.picks + e.air_picks}
    groups = {}
    for stat in R.status.values():
        sym = "diamond" if stat.station.infrasound_channel else "triangle-up"
        groups.setdefault((sym, stat.ok), []).append(stat)
    names = {("triangle-up", True): "Seismometer", ("diamond", True): "With infrasound microphone",
             ("triangle-up", False): "No data", ("diamond", False): "No data (infrasound)"}
    for (sym, ok), stats in groups.items():
        fig.add_trace(go.Scattergeo(
            lat=[x.station.lat for x in stats], lon=[x.station.lon for x in stats], mode="markers+text",
            text=[x.station.code for x in stats], textposition="top center",
            textfont=dict(size=9, color="#52514e"),
            marker=dict(size=10, symbol=sym if ok else sym + "-open",
                        line=dict(width=1, color="white" if ok else STATION_OFF),
                        color=[STATION_ON if x.station.id in used else ("#6f6d66" if ok else STATION_OFF)
                               for x in stats]),
            name=names[(sym, ok)],
            hovertext=[f"<b>{x.station.id}</b> {x.station.site}<br>{x.station.kind} · {x.station.source}"
                       f"<br>{'data OK' if x.ok else 'no data: ' + x.error}"
                       + (f"<br>{x.dist_to_area_km:.0f} km outside the area" if x.dist_to_area_km > 0 else "")
                       for x in stats], hoverinfo="text"))

    if R.catalog:
        cat = [c for c in R.catalog if wide or reg.distance_km(c.lat, c.lon, s.polygon) < 300]
        if cat:
            fig.add_trace(go.Scattergeo(
                lat=[c.lat for c in cat], lon=[c.lon for c in cat], mode="markers",
                marker=dict(size=[6 + 3 * max(c.magnitude, 0) for c in cat], symbol="circle-open",
                            color="#52514e", line=dict(width=2)), name="Catalogue event (USGS/EMSC)",
                hovertext=[f"<b>{c.source}</b> {c.description}<br>M{c.magnitude:.1f} {c.event_type}"
                           f"<br>{tfmt(c.time)}" for c in cat], hoverinfo="text"))

    for label, color in EVENT_COLORS.items():
        evs = [e for e in events if e.label == label and e.tier != "single"]
        if not evs:
            continue
        fig.add_trace(go.Scattergeo(
            lat=[e.lat for e in evs], lon=[e.lon for e in evs], mode="markers", name=label,
            marker=dict(size=[12 + 5 * (e.ml if e.ml is not None else 2) for e in evs], color=color,
                        opacity=[1.0 if e.in_region else 0.4 for e in evs],
                        symbol="star" if label in EXPLOSIVE else "circle", line=dict(width=2, color="white")),
            hovertext=[f"<b>{e.id} {e.label}</b><br>{tfmt(e.origin_time)}<br>{mag_text(e)}"
                       f"<br>{TIER_TEXT.get(e.tier, '')}" for e in evs], hoverinfo="text"))

    if R.single_signals:
        xs = R.single_signals
        fig.add_trace(go.Scattergeo(
            lat=[x["station"].lat for x in xs], lon=[x["station"].lon for x in xs], mode="markers",
            name="Strong signal at 1 station", marker=dict(size=22, symbol="circle-open", color="#eda100",
                                                          line=dict(width=3)),
            hovertext=[f"{x['station'].id}: signal/noise {x['snr']:.0f} at {tfmt(x['time'])}"
                       + (f"<br>possible distance {x['dist_km']:.0f} km" if x["dist_km"] else "") for x in xs],
            hoverinfo="text"))
        for x in xs:
            if x["dist_km"]:
                la, lo = circle_polygon(x["station"].lat, x["station"].lon, x["dist_km"], 90)
                fig.add_trace(go.Scattergeo(lat=la, lon=lo, mode="lines", hoverinfo="skip", showlegend=False,
                                            line=dict(width=1.5, color="#eda100", dash="dash")))
    if R.focus and R.focus[0] is not None:
        fig.add_trace(go.Scattergeo(lat=[R.focus[0]], lon=[R.focus[1]], mode="markers", name="Your reference point",
                                    marker=dict(size=16, symbol="x", color="#e34948")))

    b = reg.bbox(s.polygon, 1.0)
    fig.update_geos(
        projection_type="natural earth" if wide else "mercator",
        lataxis_range=None if wide else [b["min_lat"], b["max_lat"]],
        lonaxis_range=None if wide else [b["min_lon"], b["max_lon"]],
        center=dict(lat=(b["min_lat"] + b["max_lat"]) / 2 + (8 if wide else 0),
                    lon=(b["min_lon"] + b["max_lon"]) / 2 + (8 if wide else 0)),
        projection_scale=2.2 if wide else 1, showland=True, landcolor="#ebe6d6", showocean=True,
        oceancolor="#b9d3ea", showcountries=True, countrycolor="#9a978c", showcoastlines=True,
        coastlinecolor="#6f6d66", showlakes=False, resolution=110 if wide else 50)
    fig.update_layout(height=720, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    # basemap served locally (static/topojson) so the map also works offline
    st.plotly_chart(fig, width="stretch", config={"topojsonURL": "app/static/topojson/"})
    st.caption("Green outline = monitored area (events up to "
               f"{s.area_margin_km:g} km beyond it also count). ▲ seismometer · ◆ with infrasound microphone · "
               "open symbol = no data returned. ★ explosion · ● earthquake · faded = outside the area. Dotted "
               "ring = 95% location uncertainty; shaded dots = possible source area when only a few stations "
               "detected it; dashed rings = distance from a single station (direction unknown). Orange circle = "
               "strong signal that only one station recorded.")


# ---------------------------------------------------------------------------
# Timeline (noise vs events)
# ---------------------------------------------------------------------------
def timeline_tab(R):
    if not R.overview:
        st.info("No seismic data to show.")
        return
    ids = sorted(R.overview, key=lambda i: -(R.status[i].station.lat if i in R.status else 0))
    t_min = min(v[0] for v in R.overview.values())
    dt = next(iter(R.overview.values()))[1]
    n = int((R.t_end - t_min) / dt) + 2
    Z = np.full((len(ids), n), np.nan)
    for k, sid in enumerate(ids):
        t0, _, arr = R.overview[sid]
        i0 = int(round((t0 - t_min) / dt))
        m = min(len(arr), n - i0)
        if m > 0:
            Z[k, i0:i0 + m] = np.log10(np.maximum(arr[:m] * 1e9, 1e-3))
    times = [local_dt(t_min + i * dt) for i in range(n)]
    finite = Z[np.isfinite(Z)]
    zmin, zmax = (np.percentile(finite, [2, 99.5]) if finite.size else (0, 1))
    fig = go.Figure(go.Heatmap(z=Z, x=times, y=ids, colorscale="Blues", colorbar=dict(title="log₁₀ nm/s"),
                               zmin=float(zmin), zmax=float(zmax),
                               hoverongaps=False, hovertemplate="%{y}<br>%{x}<br>10^%{z:.1f} nm/s<extra></extra>"))
    noise = [q for q in R.noise_picks if q.station.id in ids]
    if noise:
        fig.add_trace(go.Scatter(x=[local_dt(q.time) for q in noise], y=[q.station.id for q in noise],
                                 mode="markers", name="Noise trigger",
                                 marker=dict(symbol="line-ns", size=10, line=dict(width=1.5, color="#8a8984"))))
    for e in R.events:
        fig.add_vline(x=local_dt(e.origin_time), line=dict(color=EVENT_COLORS.get(e.label, "#8a8984"), width=2, dash="dash"),
                      annotation_text=f"{e.icon}{e.id}", annotation_position="top")
    fig.update_layout(height=max(360, 22 * len(ids) + 120), margin=dict(l=10, r=10, t=40, b=30),
                      xaxis_title=f"Time ({tz_label()})", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")
    st.caption("Ground-motion level at every station in 10-second steps (darker = more shaking). "
               "Grey ticks = noise triggers (one station only). Dashed lines = detected events. "
               "Noise shows up at one station at a time (traffic, machinery, people); a real event "
               "appears at several stations within seconds to minutes.")
    if R.stack:
        fig = go.Figure()
        for k, (t0, rate, b, thr) in enumerate(R.stack):
            x = [local_dt(t0 + i / rate) for i in range(0, len(b), 4)]
            fig.add_trace(go.Scattergl(x=x, y=b[::4], mode="lines", line=dict(width=1, color="#2a78d6"),
                                       name="Network brightness" if k == 0 else None, showlegend=k == 0))
            fig.add_trace(go.Scatter(x=[x[0], x[-1]] if x else [], y=[thr, thr], mode="lines",
                                     line=dict(color="#e34948", dash="dash", width=1),
                                     name="Detection threshold" if k == 0 else None, showlegend=k == 0))
        for e in R.events:
            if e.tier == "stack":
                fig.add_vline(x=local_dt(e.origin_time), line=dict(color=EVENT_COLORS.get(e.label), width=2, dash="dot"))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=30), title="Network stacking brightness B(t)",
                          xaxis_title=f"Time ({tz_label()})", legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width="stretch")
        st.caption("All stations' STA/LTA added up along the travel times from the best trial source "
                   "(see Formulas ⑧). A peak above the red line with support from ≥ 3 stations (or 2 stations "
                   "with both P and S) is an event, even if no station triggered by itself.")
    hours = max((R.t_end - R.t_start) / 3600, 0.01)
    rows = [{"Station": sid, "Noise triggers": sum(1 for q in R.noise_picks if q.station.id == sid)} for sid in ids]
    for r in rows:
        r["Per hour"] = round(r["Noise triggers"] / hours, 1)
    st.dataframe(pd.DataFrame(rows).sort_values("Noise triggers", ascending=False), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Waveforms
# ---------------------------------------------------------------------------
def waveform_tab(R):
    options = []
    if R.focus and R.focus[0] is not None and R.traces is not None:
        options.append(("Your reference point", reference_event(R)))
    options += [(event_label(e), e) for e in R.events if R.traces_for(e)]
    if not options:
        st.info("No waveforms to show. For long periods, waveforms are kept around detected events only.")
        return
    label = st.selectbox("Show", [o[0] for o in options], key="wf_event")
    ev = dict(options)[label]
    traces = R.traces_for(None if ev.id == "REF" else ev)
    if ev.id == "REF":
        st.caption("Dotted lines show when P, S and air waves from your point would arrive if the event "
                   "happened exactly at your time. If the time is approximate, look for arrivals that "
                   "follow the same *shape* shifted left or right.")
    record_section(R, ev, traces)
    st.subheader("One station in detail")
    ids = sorted(traces)
    used = [q.station.id for q in ev.picks + ev.air_picks]
    default = used[0] if used else min(ids, key=lambda i: float(haversine_km(ev.lat, ev.lon, traces[i].station.lat,
                                                                              traces[i].station.lon)))
    sid = st.selectbox("Station", ids, index=ids.index(default), key=f"wf_station_{ev.id}",
                       format_func=lambda i: f"{i} – {traces[i].station.site or traces[i].station.kind}")
    station_detail(R, traces[sid])


def _dist(ev, sta):
    if ev.ring is not None and ev.ring[0].id == sta.id:
        return 0.5 * (ev.ring[1] + ev.ring[2])
    return float(haversine_km(ev.lat, ev.lon, sta.lat, sta.lon))


def _section(rows, ev, t_lo, t_hi, spacing, color, which="seismic"):
    out = []
    for d, t in rows:
        w = t.seismic if which == "seismic" else t.infrasound
        data = t.filtered if which == "seismic" else t.infra_filtered
        if w is None or data is None:
            continue
        tt = w.times() - ev.origin_time
        m = (tt >= t_lo) & (tt <= t_hi)
        if not m.any():
            continue
        y = data[m]
        y = np.clip(y / (np.max(np.abs(y)) + 1e-30), -1, 1) * spacing * 0.6
        x, yy = envelope(tt[m], y, 2000)
        out.append(go.Scattergl(x=x, y=yy + d, mode="lines", line=dict(width=0.7, color=color),
                                showlegend=False, hoverinfo="skip"))
    return out


def _pick_markers(ev, phases, fig):
    for phase in phases:
        qs = [q for q in ev.picks + ev.air_picks if q.phase.startswith(phase)]
        if qs:
            fig.add_trace(go.Scatter(
                x=[q.time - ev.origin_time for q in qs], y=[_dist(ev, q.station) for q in qs],
                mode="markers", name=f"{phase} pick",
                marker=dict(symbol="line-ns", size=18, line=dict(width=3, color=PHASE_COLORS[phase])),
                hovertext=[f"{q.station.id} {phase} {tfmt(q.time, False)}" for q in qs], hoverinfo="text"))


def _labels(fig, rows):
    for d, t in rows:
        fig.add_annotation(x=0, xref="paper", y=d, text=t.station.code, showarrow=False, xanchor="right",
                           font=dict(size=9, color="#52514e"))


def record_section(R, ev, traces):
    vm, p = R.settings.velocity, R.settings.detection
    rows = sorted(((_dist(ev, t.station), t) for t in traces.values()), key=lambda r: r[0])
    seis_rows = [r for r in rows if r[1].seismic is not None]
    if seis_rows:
        dmax = seis_rows[-1][0]
        t_room = max(t.seismic.endtime for _, t in seis_rows) - ev.origin_time
        default = int(min(max(float(physics.s_time(min(dmax, 800), ev.depth, vm)) + 60, 60), max(t_room, 31)))
        tmax = st.slider("Seconds after origin", 30, int(max(t_room, 31)), max(default, 30), 10,
                         key=f"rs_tmax_{ev.id}")
        spacing = max(dmax / max(len(seis_rows), 1), 8)
        fig = go.Figure(_section(seis_rows, ev, -30, tmax, spacing, "#3d3c38"))
        _labels(fig, seis_rows)
        dd = np.linspace(0, dmax * 1.05, 200)
        for name, tt in (("P", physics.p_time(dd, ev.depth, vm)), ("S", physics.s_time(dd, ev.depth, vm))):
            fig.add_trace(go.Scatter(x=tt, y=dd, mode="lines", name=f"Predicted {name}",
                                     line=dict(color=PHASE_COLORS[name], width=2, dash="dot")))
        _pick_markers(ev, ("P", "S"), fig)
        fig.update_layout(height=max(420, 28 * len(seis_rows)), margin=dict(l=60, r=10, t=30, b=40),
                          xaxis=dict(title="Seconds after origin time", range=[-30, tmax]),
                          yaxis=dict(title="Distance (km)", range=[-spacing, dmax + spacing]),
                          legend=dict(orientation="h", y=1.03, x=0))
        st.plotly_chart(fig, width="stretch")
        st.caption("**Record section**: every seismogram drawn at its distance from the source "
                   f"({p.freqmin:g}–{p.freqmax:g} Hz, each scaled to its own maximum). Real arrivals line "
                   "up along the dotted P and S curves; noise does not.")
    air_rows = [r for r in rows if r[1].infrasound is not None and r[0] <= 2 * p.max_air_range_km]
    if air_rows:
        st.markdown("**Air-pressure section** – infrasound microphones")
        dmax_a = air_rows[-1][0]
        t_end = max(t.infrasound.endtime for _, t in air_rows) - ev.origin_time
        t_hi = max(min(dmax_a / p.celerity_min + 60, t_end), 30)
        sp = max(dmax_a / max(len(air_rows), 1), 8)
        fig = go.Figure(_section(air_rows, ev, -30, t_hi, sp, PHASE_COLORS["Air"], "infra"))
        _labels(fig, air_rows)
        dd = np.linspace(0, dmax_a * 1.1, 100)
        fig.add_trace(go.Scatter(x=np.concatenate([dd / p.celerity_max, (dd / p.celerity_min)[::-1]]),
                                 y=np.concatenate([dd, dd[::-1]]), fill="toself", mode="none",
                                 fillcolor="rgba(27,175,122,0.12)",
                                 name=f"Expected air arrival ({p.celerity_min * 1000:.0f}–{p.celerity_max * 1000:.0f} m/s)"))
        _pick_markers(ev, ("Air",), fig)
        fig.update_layout(height=max(300, 45 * len(air_rows)), margin=dict(l=60, r=10, t=30, b=40),
                          xaxis=dict(title="Seconds after origin time", range=[-30, t_hi]),
                          yaxis=dict(title="Distance (km)", range=[-sp, dmax_a + sp]),
                          legend=dict(orientation="h", y=1.06, x=0))
        st.plotly_chart(fig, width="stretch")
        st.caption("An explosion's air wave travels ~20× slower than seismic waves. A pulse inside the "
                   "shaded band is strong evidence of a surface explosion.")


def station_detail(R, t):
    p = R.settings.detection
    rows = []
    if t.seismic is not None:
        rows.append(("seismic", t.seismic, t.filtered * 1e6, t.cft, f"Ground velocity (µm/s), {p.freqmin:g}–{p.freqmax:g} Hz"))
    if t.infrasound is not None:
        rows.append(("infra", t.infrasound, t.infra_filtered, t.infra_cft, "Air pressure (Pa)"))
    titles = [x for r in rows for x in (r[4], "STA/LTA")]
    n = 2 * len(rows)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=titles)
    k = 1
    for kind, w, data, cft, _ in rows:
        times = np.array([local_dt(x) for x in (w.starttime, w.endtime)])
        tx = pd.date_range(times[0], periods=len(data), freq=pd.Timedelta(seconds=1 / w.fs))
        x, y = envelope(np.arange(len(data)), data, 4000)
        color = "#3d3c38" if kind == "seismic" else PHASE_COLORS["Air"]
        fig.add_trace(go.Scattergl(x=tx[x.astype(int)], y=y, mode="lines", line=dict(width=0.8, color=color),
                                   showlegend=False), row=k, col=1)
        x, y = envelope(np.arange(len(cft)), cft, 4000)
        fig.add_trace(go.Scattergl(x=tx[x.astype(int)], y=np.maximum(y, 0.01), mode="lines",
                                   line=dict(width=1, color=PHASE_COLORS["P"]), showlegend=False), row=k + 1, col=1)
        on = p.trigger_on if kind == "seismic" else p.infra_trigger_on
        fig.add_hline(y=on, line=dict(color="#e34948", dash="dash", width=1), row=k + 1, col=1)
        fig.update_yaxes(type="log", row=k + 1, col=1)
        for q in (R.picks if kind == "seismic" else R.infra_picks):
            if q.station.id == t.station.id and w.starttime <= q.time <= w.endtime:
                ph = "Air" if q.phase.startswith("Air") else q.phase
                for row in (k, k + 1):
                    fig.add_vline(x=local_dt(q.time), row=row, col=1,
                                  line=dict(color=PHASE_COLORS.get(ph, "#8a8984"), width=1.2,
                                            dash="solid" if q.event_id else "dot"))
        k += 2
    fig.update_layout(height=190 * n + 60, margin=dict(l=10, r=10, t=40, b=30))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{t.station.id} ({t.station.kind}), times in {tz_label()}. Red dashed = trigger level. "
               "Vertical lines: solid = part of an event (blue P, orange S, green air wave), dotted grey = noise.")


# ---------------------------------------------------------------------------
# Event analysis
# ---------------------------------------------------------------------------
def analysis_tab(R):
    evs = R.in_region + R.outside
    if not evs:
        st.info("No events to analyse.")
        return
    labels = [event_label(e) for e in evs]
    ev = evs[labels.index(st.selectbox("Event", labels, key="an_event"))]
    st.subheader(event_title(ev))
    st.caption(f"Detection type: {TIER_TEXT.get(ev.tier, '')}. "
               + ("Inside the monitored area." if ev.in_region else "**Outside the monitored area.**"))
    if ev.kind == "distant":
        a, b, c, d = st.columns(4)
        a.metric("Direction (back-azimuth)", f"{ev.back_azimuth:.0f}°")
        b.metric("Apparent speed", f"{ev.app_velocity:.1f} km/s")
        metric(c, "Distance", f"{ev.distance_deg:.1f}°", f"{ev.distance_deg * KM_PER_DEG:,.0f} km")
        d.metric("Size", mag_text(ev))
        st.markdown(ev.summary)
        return
    for n in ev.notes:
        st.warning(n)
    c1, c2 = st.columns([1, 1.6])
    with c1:
        g = go.Figure(go.Indicator(
            mode="gauge+number", value=ev.p_explosion * 100, number=dict(suffix="%"),
            title=dict(text="Explosion probability"),
            gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#3d3c38", thickness=0.25),
                       steps=[dict(range=[0, 30], color="#b7d3f6"), dict(range=[30, 70], color="#e6e4dd"),
                              dict(range=[70, 100], color="#f6c3ac")])))
        g.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(g, width="stretch")
        st.markdown(f"**Verdict: {ev.label}.** Below 30% → earthquake-like, above 70% → explosion-like. "
                    "'Possible' = seen by few stations; treat with care.")
    with c2:
        names = ["Starting point (prior)"] + [e.name for e in ev.evidence]
        vals = [PRIOR_BIAS] + [e.contribution for e in ev.evidence]
        meas = ["Natural earthquakes are more common than explosions"] + [e.measurement for e in ev.evidence]
        fig = go.Figure()
        for sign, name, color in ((1, "Points to explosion", EVENT_COLORS["Likely explosion"]),
                                  (-1, "Points to earthquake", EVENT_COLORS["Likely earthquake"])):
            idx = [i for i, v in enumerate(vals) if (v > 0 if sign > 0 else v <= 0)]
            fig.add_trace(go.Bar(y=[names[i] for i in idx], x=[vals[i] for i in idx], orientation="h", name=name,
                                 marker=dict(color=color, cornerradius=4), hovertext=[meas[i] for i in idx],
                                 hoverinfo="text+x"))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=30), barmode="relative",
                          title="What pushed the verdict (weight × score)",
                          xaxis=dict(title="← earthquake-like      explosion-like →", zeroline=True,
                                     zerolinecolor="#8a8984"),
                          yaxis=dict(autorange="reversed", categoryorder="array", categoryarray=names),
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame([{
        "Clue": e.name, "Measured": e.measurement, "Score (−2…+2)": round(e.score, 2), "Weight": e.weight,
        "Contribution": round(e.contribution, 2), "Why it matters": e.explanation} for e in ev.evidence]),
        hide_index=True, width="stretch")

    st.subheader("Where and when")
    a, b, c, d = st.columns(4)
    metric(a, "Origin time", tfmt(ev.origin_time, False), tz_label())
    if ev.tier == "single":
        metric(b, "Distance from station", f"{ev.ring[1]:.0f}–{ev.ring[2]:.0f} km", f"from {ev.ring[0].id}")
    else:
        metric(b, "Epicentre", f"{ev.lat:.2f}°N, {ev.lon:.2f}°E", f"± {ev.error_km:.0f} km (95%)")
    metric(c, "Depth", f"{ev.depth:.0f} km",
           f"95%: {ev.depth_min:.0f}–{ev.depth_max:.0f} km" + ("" if ev.depth_constrained else " (unresolved)"))
    metric(d, "Stations", f"{len({q.station.id for q in ev.picks + ev.air_picks})}", f"RMS misfit {ev.rms:.2f} s")
    st.caption(f"Location in words: **{where_text(ev)}**.")
    travel_time_plot(R, ev)

    st.subheader("How big")
    if ev.ml is not None:
        e_j = physics.seismic_energy_joules(ev.ml)
        a, b, c, d = st.columns(4)
        metric(a, "Local magnitude", f"ML {ev.ml:.1f}", f"{len(ev.ml_stations)} station(s)")
        metric(b, "Seismic energy", fmt_energy(e_j), f"= {fmt_tons(e_j / physics.JOULES_PER_TON_TNT)} of TNT")
        metric(c, "Yield if buried explosion", fmt_tons(float(physics.explosion_yield_tons(ev.ml))),
               "fully coupled to rock")
        metric(d, "Yield if surface blast",
               fmt_tons(float(physics.explosion_yield_tons(ev.ml, physics.SURFACE_BLAST_COUPLING))),
               f"assumes {physics.SURFACE_BLAST_COUPLING:.0%} ground coupling")
        st.caption("Yield estimates are order-of-magnitude only (factor 3–10 uncertainty).")
        st.dataframe(pd.DataFrame([{
            "Station": m["station"], "Site": m["site"], "Distance (km)": round(m["dist_km"]),
            "Wood-Anderson amplitude (nm)": round(m["amp_nm"], 1), "ML": round(m["ml"], 2)}
            for m in sorted(ev.ml_stations, key=lambda m: m["dist_km"])]), hide_index=True, width="stretch")
    else:
        st.info("Magnitude not measured (no clear ground-wave amplitude).")
    if ev.air_picks:
        st.subheader("Air-wave arrivals")
        rows = []
        for q in ev.air_picks:
            d = _dist(ev, q.station)
            tt = q.time - ev.origin_time
            rows.append({"Station": q.station.id,
                         "Sensor": "infrasound" if q.channel.startswith(("HD", "BD")) else "seismometer (air-coupled)",
                         "Distance (km)": round(d), "Arrival": tfmt(q.time, False), "Travel time (s)": round(tt, 1),
                         "Celerity (m/s)": round(d / tt * 1000) if tt > 0 else None,
                         "Peak (Pa)": round(q.amplitude, 3) if q.amplitude else None})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Celerity = distance ÷ travel time; sound in air arrives at ~260–360 m/s.")


def travel_time_plot(R, ev):
    vm, pd_ = R.settings.velocity, R.settings.detection
    rows = [(q, _dist(ev, q.station)) for q in ev.picks + ev.air_picks
            if q.phase in ("P", "S") or q.phase.startswith("Air")]
    if not rows:
        return
    air = [r for r in rows if r[0].phase.startswith("Air")]
    seis = [r for r in rows if not r[0].phase.startswith("Air")]
    panels = ([("Seismic waves", ("P", "S"), seis)] if seis else []) + ([("Air wave", ("Air",), air)] if air else [])
    fig = make_subplots(rows=1, cols=len(panels), horizontal_spacing=0.1, subplot_titles=[x[0] for x in panels])
    for col, (_, phases, sel) in enumerate(panels, start=1):
        dd = np.linspace(0, max(d for _, d in sel) * 1.1 + 1, 200)
        if phases == ("Air",):
            fig.add_trace(go.Scatter(x=np.concatenate([dd, dd[::-1]]),
                                     y=np.concatenate([dd / pd_.celerity_max, (dd / pd_.celerity_min)[::-1]]),
                                     fill="toself", mode="none", fillcolor="rgba(27,175,122,0.12)",
                                     name="Plausible air arrivals"), row=1, col=col)
        for ph in phases:
            tt = {"P": physics.p_time(dd, ev.depth, vm), "S": physics.s_time(dd, ev.depth, vm),
                  "Air": physics.air_time(dd, vm.c_air)}[ph]
            fig.add_trace(go.Scatter(x=dd, y=tt, mode="lines", name=f"Predicted {ph}",
                                     line=dict(color=PHASE_COLORS[ph], width=2)), row=1, col=col)
            qs = [(q, d) for q, d in sel if q.phase.startswith(ph)]
            fig.add_trace(go.Scatter(x=[d for _, d in qs], y=[q.time - ev.origin_time for q, _ in qs],
                                     mode="markers", name=f"Observed {ph}",
                                     marker=dict(size=9, color=PHASE_COLORS[ph], line=dict(width=2, color="white")),
                                     hovertext=[f"{q.station.id}: {q.time - ev.origin_time:.1f} s" for q, _ in qs],
                                     hoverinfo="text"), row=1, col=col)
        fig.update_xaxes(title_text="Distance (km)", row=1, col=col)
        fig.update_yaxes(title_text="Travel time (s)", row=1, col=col)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=40), legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, width="stretch")
    st.caption("Dots = measured arrival times, lines = the Earth model's prediction for this source.")


# ---------------------------------------------------------------------------
# Stations & data
# ---------------------------------------------------------------------------
def stations_tab(R):
    hours = max((R.t_end - R.t_start) / 3600, 0.01)
    rows = [{"Station": sid, "Site": s.station.site, "Type": s.station.kind, "Source": s.station.source,
             "Outside area (km)": round(s.dist_to_area_km), "Data": "✅" if s.ok else "❌",
             "Data coverage": f"{s.availability * 100:.0f}%" if s.ok else "", "Channels": ", ".join(s.channels),
             "Triggers / hour": round(R.triggers_total.get(sid, 0) / hours, 1), "Problem": s.error}
            for sid, s in sorted(R.status.items(), key=lambda kv: kv[1].dist_to_area_km)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("❌ = the station returned no data for this period (offline, not sharing data, or the server "
               "refused the request).")
    st.subheader("All triggers")
    allp = sorted(R.picks + R.infra_picks, key=lambda q: q.time)
    st.dataframe(pd.DataFrame([{
        "Time": tfmt(q.time), "Station": q.station.id,
        "Sensor": "air" if q.channel.startswith(("HD", "BD")) else "ground",
        "Event": q.event_id or "—", "Role": q.phase if q.event_id else "noise", "Signal/noise": round(q.snr, 1),
        "Peak STA/LTA": round(q.peak_ratio, 1), "First motion": {1: "up", -1: "down", 0: "?"}[q.polarity]}
        for q in allp]), hide_index=True, width="stretch", height=400)
    if R.events:
        df = pd.DataFrame([{
            "id": e.id, "type": e.label, "in_area": e.in_region, "detection": e.tier,
            "origin_utc": datetime.fromtimestamp(e.origin_time, timezone.utc).isoformat(),
            "lat": round(e.lat, 3), "lon": round(e.lon, 3), "error_km": round(e.error_km, 1),
            "depth_km": e.depth, "ml": None if e.ml is None else round(e.ml, 2),
            "p_explosion": round(e.p_explosion, 3), "summary": e.summary} for e in R.events])
        st.download_button("⬇ Download events (CSV)", df.to_csv(index=False), "gulfseis_events.csv", "text/csv")
    if R.messages:
        with st.expander("Log"):
            st.code("\n".join(R.messages))


# ---------------------------------------------------------------------------
# Formulas & guide
# ---------------------------------------------------------------------------
def formulas_tab(R):
    vm, p = R.settings.velocity, R.settings.detection
    ev = next((e for e in R.in_region if e.kind == "local"), None)
    st.markdown("Every number in this app comes from these formulas"
                + (f"; worked examples use **{ev.id}** ({ev.label.lower()})." if ev else "."))
    with st.expander("① Is it inside the monitored area? (ray casting)"):
        st.latex(r"\text{inside} \iff \text{a ray from the point crosses the outline an odd number of times}")
        st.markdown(f"The green outline has {len(R.settings.polygon)} corners and covers about "
                    f"{reg.area_km2(R.settings.polygon):,.0f} km². Stations up to "
                    f"{R.settings.station_buffer_km:g} km outside it are also used, because waves travel "
                    "beyond the coast.")
    with st.expander("② Detecting a signal: STA/LTA trigger", expanded=True):
        st.latex(r"\mathrm{STA}(t)=\frac{1}{N_s}\sum_{k=t-N_s+1}^{t}x_k^2 \qquad "
                 r"\mathrm{LTA}(t)=\frac{1}{N_l}\sum_{k=t-N_s-N_l+1}^{t-N_s}x_k^2 \qquad R(t)=\frac{\mathrm{STA}}{\mathrm{LTA}}")
        st.markdown(f"Short-term average ({p.sta:g} s) vs long-term average ({p.lta:g} s) of the "
                    f"{p.freqmin:g}–{p.freqmax:g} Hz filtered signal. A trigger starts at R > {p.trigger_on:g} "
                    f"(infrasound: {p.infra_trigger_on:g}). The onset is refined with the AIC picker: "
                    r"$\mathrm{AIC}(k)=k\log\mathrm{var}(x_{1..k})+(N-k-1)\log\mathrm{var}(x_{k+1..N})$.")
    with st.expander("③ Noise or event? (coincidence between stations)", expanded=True):
        st.latex(r"|t_i - t_j| \le \frac{d_{ij}}{V_p} + \delta")
        st.markdown("A wave from one source reaches two stations at most (distance ÷ wave speed) apart. "
                    "Triggers that no other station confirms within that time — and that have no matching "
                    "air wave — are **noise**: traffic, machinery, wind or people near that one sensor. "
                    f"With only 2 stations, chance coincidences happen, so both must be strong "
                    f"(signal/noise ≥ {p.min_snr_small:g}) and the possible source area small "
                    f"(≤ {p.max_small_error_km:g} km) and inside the outline.")
    with st.expander("④ Travel times and location"):
        st.latex(r"t_{Pg}=\frac{\sqrt{d^2+h^2}}{V_p},\quad t_{Pn}=\frac{d}{V_{pn}}+(2H-h)\sqrt{\tfrac{1}{V_p^2}-\tfrac{1}{V_{pn}^2}},\quad "
                 r"\chi^2=\sum_i\Big(\frac{t_i-t_0-T_i}{\sigma_i}\Big)^2")
        st.markdown(f"Crust {vm.moho:g} km, Vp {vm.vp:g}, Vs {vm.vs:g}, Pn {vm.vpn:g} km/s. The source is found by "
                    "trying every grid point; the 95% region is where χ² ≤ χ²_min + 5.99. With 2–3 stations the "
                    "source is only known to lie in an **area** (all grid points inside the outline that fit).")
        st.latex(r"d \approx \Delta t_{S-P}\cdot\frac{V_pV_s}{V_p-V_s}\approx "
                 + f"{vm.vp * vm.vs / (vm.vp - vm.vs):.1f}" + r"\,\Delta t_{S-P}\ \text{km}")
    with st.expander("⑤ One station, two waves: ground + air (seismo-acoustic ranging)", expanded=True):
        st.latex(r"t_{air}-t_{ground}=\frac{d}{c}-\frac{d}{V_p}\ \Rightarrow\ d=\frac{t_{air}-t_{ground}}{1/c-1/V_p}")
        st.markdown(f"An explosion sends a fast wave through the ground and a slow one through the air "
                    f"(c ≈ {p.celerity_min * 1000:.0f}–{p.celerity_max * 1000:.0f} m/s). Every minute between them "
                    f"≈ {60 / (1 / 0.31 - 1 / vm.vp):.0f} km of distance, so a Raspberry Shake & Boom can range an "
                    "explosion on its own (but cannot tell the direction).")
        st.latex(r"c = 331.3\sqrt{1+T/273.15}\ \text{m/s}")
    with st.expander("⑧ Network stacking: finding events too weak for single-station triggers", expanded=True):
        st.latex(r"c_i(t)=\min\big(\max(R_i(t)-q_i,\,0),\,5\big)\qquad "
                 r"s_i(\mathbf{x},t)=c_i\big(t+T_{P,i}(\mathbf{x})\big)+c_i\big(t+T_{S,i}(\mathbf{x})\big)")
        st.latex(r"B(\mathbf{x},t)=\sum_i s_i(\mathbf{x},t)-\max_i s_i(\mathbf{x},t)\qquad "
                 r"\text{event if }\ \max_{\mathbf{x}}B>\max\big(B_0,\ \mathrm{median}+k\cdot1.4826\,\mathrm{MAD}\big)")
        st.markdown(f"R_i is station i's STA/LTA and q_i its own 90th percentile, so busy stations don't "
                    f"dominate. For every trial source **x** (a {p.stack_step_deg:g}° grid) and origin time t, the "
                    "stations' values at the predicted P and S arrival times are added. A real event lines up "
                    "across stations; noise at one station does not, and subtracting the largest term means "
                    f"one station alone can never make an event. B₀ = {p.stack_threshold:g}, k = "
                    f"{p.stack_mad_factor:g}; support from ≥ 3 stations, or 2 stations that both show P and S, "
                    "is required. Waves of events already found are masked out first.")
    with st.expander("⑥ Magnitude, energy and yield"):
        st.latex(r"M_L=\log_{10}A+1.11\log_{10}R+0.00189R-2.09,\qquad \log_{10}E=1.5M+4.8,\qquad "
                 r"Y_{kt}=10^{(M-4.45)/0.75}/\varepsilon")
        st.markdown("A = Wood-Anderson amplitude (nm), R = distance (km). ε ≈ 0.05 for a surface blast "
                    "(calibrated on Beirut 2020: ~0.5–1 kt TNT, ML ≈ 3.3).")
        if ev and ev.ml_stations:
            m = ev.ml_stations[0]
            st.info(f"{m['station']}: A = {m['amp_nm']:.1f} nm, R = {m['hypo_km']:.0f} km → ML = {m['ml']:.2f}")
    with st.expander("⑦ Explosion or earthquake: combining clues", expanded=True):
        st.latex(r"P(\text{explosion})=\frac{1}{1+e^{-z}},\quad z=b+\sum_i w_i s_i")
        st.markdown(f"b = {PRIOR_BIAS}. Clues: P/S amplitude ratio (explosions: strong P), depth (explosions "
                    "are at the surface), first motion (explosions push every station UP), air wave "
                    "(only surface/air explosions), catalogue match, time of day.")
        if ev:
            z = PRIOR_BIAS + sum(e.contribution for e in ev.evidence)
            st.info(f"{ev.id}: z = {z:.2f} → P = {ev.p_explosion * 100:.0f}% → {ev.label}.")


def guide_tab():
    st.markdown("""
### What the app does
It downloads **real recordings** from public stations in and around the green outline, then sorts every
signal into one of these:

| Result | Meaning |
|---|---|
| 💥 **Likely explosion** | Several stations, and the clues (surface source, strong P wave, ground pushed up everywhere, air-pressure wave) point to a blast. |
| 💥 **Possible explosion** | Explosion-like, but seen by only 1–3 stations or only as an air-pressure wave. Worth a closer look. |
| 🌍 **Earthquake** | Deeper source, strong S wave, mixed first motions, no air wave. |
| ❓ **Uncertain** | A real event, but the clues disagree or are missing. |
| 🔇 **Noise** | A trigger at one station that nothing else confirms: traffic, machinery, wind, doors, footsteps. |
| 🌐 **Outside** | A real event whose source is outside the green outline (e.g. a Zagros or distant earthquake). |

### How to check an explosion you know about
1. Choose **Check a known event**, pick your time zone, the date, the approximate time and the place.
2. Press **Run**. The top card says whether it was found.
3. If not, the table shows each station's **signal/noise** where the waves should have arrived, and the
   *Waveforms* tab draws every recording with the predicted P, S and air-wave arrival times.

### Why an event can be missed
- **No station close enough**: small blasts are only recorded within tens of km. Check the Stations tab:
  are there stations near the place, and did they return data (✅)?
- **Stations offline or not sharing**: Raspberry Shakes are run by volunteers and are sometimes off.
- **Wrong time zone or window**: pick the right time zone; air waves arrive minutes after the blast.
- **Noisy site**: city stations are busy in daytime; use *High* sensitivity for a known event.

### Honest limitations
Locations from 2–3 stations are areas, not points; single-station ranges have no direction. Air-only
detections can also be thunder, sonic booms or other loud sounds. Confirm important events with official
agencies (Kuwait National Seismic Network, IRSC, NCM, USGS, EMSC).
""")


# ---------------------------------------------------------------------------
def main():
    cfg, run = sidebar()
    if run or st.session_state.get("pending_refresh"):
        st.session_state.pending_refresh = False
        run_analysis(cfg)
    R = st.session_state.get("result")
    if R is None:
        st.title("GulfSeis · Explosion & Noise Monitor")
        st.markdown("Choose a period in the sidebar and press **▶ Run**. The app downloads real recordings "
                    "from Raspberry Shake, EarthScope and GEOFON stations in and around the Persian Gulf and "
                    "sorts every signal into **explosion**, **earthquake** or **noise**.")
        if st.session_state.get("log"):
            with st.expander("Log", expanded=True):
                st.code("\n".join(st.session_state.log))
        return
    header(R)
    tabs = st.tabs(["🗺️ Map", "🕒 Timeline & noise", "📈 Waveforms", "🔍 Event analysis",
                    "📡 Stations & data", "📐 Formulas", "❓ How it works"])
    with tabs[0]:
        map_tab(R)
    with tabs[1]:
        timeline_tab(R)
    with tabs[2]:
        waveform_tab(R)
    with tabs[3]:
        analysis_tab(R)
    with tabs[4]:
        stations_tab(R)
    with tabs[5]:
        formulas_tab(R)
    with tabs[6]:
        guide_tab()
    if cfg["mode"] == "Recent hours" and cfg.get("auto"):
        wait = cfg["auto_min"] * 60 - (time.time() - st.session_state.get("ran_at", 0))
        st.sidebar.caption(f"Next refresh in ~{max(wait, 0) / 60:.0f} min")
        time.sleep(max(wait, 1))
        st.session_state.pending_refresh = True
        st.rerun()


main()
