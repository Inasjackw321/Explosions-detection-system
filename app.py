"""GulfSeis - Seismic & Explosion Monitor for the Persian Gulf.

Run with:   python run_app.py        (or: streamlit run app.py)
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from gulfseis import data_sources, physics, synthetic
from gulfseis.config import DATA_CENTERS, DEFAULT_REGION, DetectionParams, Settings, VelocityModel
from gulfseis.discrimination import PRIOR_BIAS
from gulfseis.geo import KM_PER_DEG, circle_polygon, destination, haversine_km
from gulfseis.models import fmt_time
from gulfseis.pipeline import analyze
from gulfseis.places import describe

st.set_page_config(page_title="GulfSeis - Explosion & Earthquake Monitor", page_icon="💥",
                   layout="wide")

# Colours (validated categorical slots 1-3 + neutral grey for "uncertain")
EVENT_COLORS = {"Likely explosion": "#eb6834", "Likely earthquake": "#2a78d6",
                "Distant earthquake": "#1baf7a", "Uncertain": "#8a8984"}
PHASE_COLORS = {"P": "#2a78d6", "S": "#eb6834", "Air": "#1baf7a"}
STATION_ON, STATION_OFF = "#3d3c38", "#b5b3ab"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar():
    sb = st.sidebar
    sb.title("💥 GulfSeis")
    sb.caption("Detect explosions and earthquakes around the Persian Gulf with public "
               "seismographs, including Raspberry Shake citizen stations.")

    sb.header("1 · Data")
    mode = sb.radio("Data source", ["Demo – simulated Gulf network", "Real data – internet (FDSN)"],
                    help="Demo mode needs no internet: it simulates 19 stations around the Gulf "
                         "recording an explosion, an earthquake and a distant earthquake.")
    cfg = {"mode": "demo" if mode.startswith("Demo") else "real"}

    if cfg["mode"] == "demo":
        scen = sb.selectbox("Scenario", ["Explosion + earthquake + distant earthquake",
                                         "Build your own event", "Quiet period (noise only)"])
        cfg["scenario"] = scen
        if scen == "Build your own event":
            c = sb.container(border=True)
            kind = c.selectbox("Event type", ["explosion", "earthquake"])
            lat = c.number_input("Latitude", 22.0, 33.0, 26.9, 0.1)
            lon = c.number_input("Longitude", 45.0, 60.0, 55.9, 0.1)
            depth = c.slider("Depth (km)", 0.0, 35.0, 0.0 if kind == "explosion" else 12.0, 1.0)
            mag = c.slider("Magnitude (ML)", 1.5, 5.5, 3.0, 0.1)
            cfg["custom"] = synthetic.ScenarioEvent(kind, 120.0, lat, lon, depth, mag,
                                                    f"Your {kind}", strike=45.0)
        cfg["noise"] = sb.slider("Background noise level", 0.5, 5.0, 1.0, 0.25,
                                 help="1 = typical; higher = noisier sites (traffic, wind).")
        cfg["seed"] = int(sb.number_input("Random seed", 0, 9999, 7))
    else:
        if not data_sources.obspy_available():
            sb.error("Real data needs ObsPy: `pip install obspy`")
        cfg["centers"] = sb.multiselect("Data centres", list(DATA_CENTERS), default=list(DATA_CENTERS))
        latest = sb.checkbox("Most recent data", True,
                             help="Ends 5 minutes ago (data needs a few minutes to reach the servers).")
        if latest:
            end = datetime.now(timezone.utc) - timedelta(minutes=5)
        else:
            d = sb.date_input("End date (UTC)", datetime.now(timezone.utc).date())
            t = sb.time_input("End time (UTC)", datetime.now(timezone.utc).time().replace(microsecond=0))
            end = datetime.combine(d, t, tzinfo=timezone.utc)
        cfg["end"] = end.timestamp()
        cfg["minutes"] = sb.slider("Time window (minutes)", 10, 120, 30, 5,
                                   help="Air-blast waves travel slowly (~18 km per minute), "
                                        "so use 30+ minutes to catch them at distant stations.")
        cfg["max_stations"] = sb.slider("Max stations", 5, 80, 40)
        cfg["infrasound"] = sb.checkbox("Use infrasound (Raspberry Shake & Boom)", True)
        cfg["catalog"] = sb.checkbox("Cross-check with USGS/EMSC catalogues", True)
        cfg["auto"] = sb.checkbox("Auto-refresh", False, help="Re-run the analysis on a timer.")
        if cfg["auto"]:
            cfg["auto_min"] = sb.slider("Refresh every (minutes)", 2, 30, 5)

    sb.header("2 · Detection")
    p = DetectionParams()
    with sb.expander("Trigger settings"):
        p.freqmin, p.freqmax = st.slider("Band-pass filter (Hz)", 0.5, 20.0, (p.freqmin, p.freqmax), 0.5)
        p.sta = st.slider("STA window (s)", 0.2, 5.0, p.sta, 0.1)
        p.lta = st.slider("LTA window (s)", 5.0, 120.0, p.lta, 5.0)
        p.trigger_on = st.slider("Trigger ON when STA/LTA >", 2.0, 10.0, p.trigger_on, 0.25)
        p.trigger_off = st.slider("Trigger OFF when STA/LTA <", 0.5, 3.0, p.trigger_off, 0.1)
        p.min_stations = st.slider("Stations needed for an event", 3, 8, p.min_stations)
    vm = VelocityModel()
    with sb.expander("Earth & air model"):
        vm.vp = st.number_input("Crust P speed Vp (km/s)", 5.0, 7.0, vm.vp, 0.05)
        vm.vs = st.number_input("Crust S speed Vs (km/s)", 2.8, 4.2, vm.vs, 0.05)
        vm.moho = st.number_input("Crust thickness (km)", 25.0, 60.0, vm.moho, 1.0)
        vm.vpn = st.number_input("Mantle P speed Pn (km/s)", 7.5, 8.5, vm.vpn, 0.05)
        vm.vsn = st.number_input("Mantle S speed Sn (km/s)", 4.0, 5.0, vm.vsn, 0.05)
        vm.air_temp_c = st.slider("Air temperature (°C)", -10, 50, int(vm.air_temp_c))
        st.caption(f"Speed of sound: {vm.c_air * 1000:.0f} m/s")
    region = dict(DEFAULT_REGION)
    with sb.expander("Region"):
        region["min_lat"], region["max_lat"] = st.slider("Latitude", 15.0, 40.0,
                                                         (region["min_lat"], region["max_lat"]), 0.5)
        region["min_lon"], region["max_lon"] = st.slider("Longitude", 38.0, 68.0,
                                                         (region["min_lon"], region["max_lon"]), 0.5)
    cfg["settings"] = Settings(region=region, velocity=vm, detection=p)
    run = sb.button("▶ Run analysis", type="primary", width="stretch")
    return cfg, run


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run_analysis(cfg):
    s: Settings = cfg["settings"]
    log = []
    bar = st.progress(0.0, "Starting…")

    def progress(frac, text):
        bar.progress(min(max(frac, 0.0), 1.0), text)

    if cfg["mode"] == "demo":
        events = {"Explosion + earthquake + distant earthquake": None,
                  "Quiet period (noise only)": [],
                  "Build your own event": [cfg.get("custom")]}[cfg["scenario"]]
        progress(0.05, "Simulating seismograms…")
        stations, wfs, catalog, truth = synthetic.generate(events, noise_level=cfg["noise"],
                                                           seed=cfg["seed"], vm=s.velocity)
        source = "Demo: simulated stations (network code XX)"
    else:
        truth = None
        if not cfg["centers"]:
            st.error("Pick at least one data centre.")
            bar.empty()
            return None
        t_end = cfg["end"]
        t_start = t_end - cfg["minutes"] * 60
        progress(0.02, "Finding stations…")
        stations, wfs, catalog = data_sources.load_real_data(
            cfg["centers"], s.region, t_start, t_end, cfg["infrasound"], cfg["max_stations"],
            cfg["catalog"], log=log.append, progress=lambda f, t: progress(0.05 + 0.6 * f, t))
        source = "Real data: " + ", ".join(cfg["centers"])
        if not [w for w in wfs if not w.is_infrasound]:
            bar.empty()
            st.session_state.result = None
            st.session_state.log = log
            st.error("No seismic data could be downloaded. The download log is shown below.")
            return None
    res = analyze(wfs, s, catalog, progress=lambda f, t: progress(0.65 + 0.35 * f, t))
    res.messages = log
    bar.empty()
    st.session_state.result = res
    st.session_state.truth = truth
    st.session_state.source = source
    st.session_state.ran_at = time.time()
    return res


# ---------------------------------------------------------------------------
# Helpers
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
    """A metric with a neutral grey sub-line (no up/down arrow)."""
    col.metric(label, value)
    if sub:
        col.caption(sub)


def event_title(ev):
    if ev.kind == "distant":
        where = ev.catalog_match.description if ev.catalog_match and ev.catalog_match.description else \
            f"{ev.distance_deg:.0f}° away, back-azimuth {ev.back_azimuth:.0f}°"
    else:
        where = describe(ev.lat, ev.lon)
    return f"{ev.icon} {ev.id} · {ev.label} · {where}"


def event_label(ev):
    return f"{ev.id} – {ev.label} – {fmt_time(ev.origin_time, False)}"


def mag_text(ev):
    if ev.kind == "distant":
        if ev.catalog_match:
            return f"M{ev.catalog_match.magnitude:.1f} ({ev.catalog_match.mag_type or 'catalogue'})"
        return "—"
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


# ---------------------------------------------------------------------------
# Header, metrics and event cards
# ---------------------------------------------------------------------------
def header(res):
    st.title("GulfSeis · Explosion & Earthquake Monitor")
    st.caption(f"{st.session_state.get('source', '')} · {fmt_time(res.t_start)} → "
               f"{fmt_time(res.t_end, False)} · "
               f"{len(res.traces)} seismic stations, "
               f"{sum(t.infrasound is not None for t in res.traces.values())} with infrasound")
    evs = res.events
    c = st.columns(6)
    c[0].metric("Stations", len(res.traces))
    c[1].metric("Triggers", len(res.picks), help="Every time a station's STA/LTA crossed the ON level.")
    c[2].metric("Events", len(evs), help="Triggers seen by several stations that fit one source.")
    c[3].metric("💥 Explosions", sum(e.label == "Likely explosion" for e in evs))
    c[4].metric("🌍 Earthquakes", sum(e.label == "Likely earthquake" for e in evs))
    c[5].metric("🌐 Distant / ❓", f"{sum(e.kind == 'distant' for e in evs)} / "
                                   f"{sum(e.label == 'Uncertain' for e in evs)}")

    if not evs:
        st.info("No events detected in this time window. Individual station triggers (noise) are "
                "listed in the *Stations & triggers* tab.")
    for ev in evs:
        with st.container(border=True):
            a, b, c2, d = st.columns([3.2, 1.4, 1.3, 1.6])
            a.markdown(f"**{event_title(ev)}**")
            a.caption(ev.summary)
            b.markdown(f"**Time (UTC)**  \n{fmt_time(ev.origin_time)}")
            if ev.kind == "local":
                c2.markdown(f"**Size**  \n{mag_text(ev)}")
                d.markdown(f"**Location**  \n{ev.lat:.2f}°N {ev.lon:.2f}°E ± {ev.error_km:.0f} km, "
                           f"depth {ev.depth:.0f} km")
            else:
                c2.markdown(f"**Size**  \n{mag_text(ev)}")
                d.markdown(f"**Location**  \n{ev.lat:.1f}°N {ev.lon:.1f}°E  \n"
                           f"{ev.distance_deg * KM_PER_DEG:,.0f} km away")
            if ev.kind == "local":
                st.progress(ev.p_explosion,
                            f"Explosion probability {ev.p_explosion * 100:.0f}%  "
                            f"(0% = certainly earthquake, 100% = certainly explosion)")


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
def map_tab(res):
    s = res.settings
    r = s.region
    view = st.radio("View", ["Gulf region", "Wide (includes distant events)"], horizontal=True, key="map_view")
    fig = go.Figure()
    used = {q.station.id for e in res.events for q in e.picks}

    # rays from local events to the stations that recorded them
    for ev in res.events:
        if ev.kind != "local":
            continue
        lats, lons = [], []
        for sid in ev.stations:
            sta = res.traces[sid].seismic.station
            lats += [ev.lat, sta.lat, None]
            lons += [ev.lon, sta.lon, None]
        fig.add_trace(go.Scattergeo(lat=lats, lon=lons, mode="lines", hoverinfo="skip",
                                    line=dict(width=1, color=EVENT_COLORS[ev.label]), opacity=0.35,
                                    showlegend=False))
        el, eo = circle_polygon(ev.lat, ev.lon, max(ev.error_km, 1.0))
        fig.add_trace(go.Scattergeo(lat=el, lon=eo, mode="lines", hoverinfo="skip",
                                    line=dict(width=1.5, color=EVENT_COLORS[ev.label], dash="dot"),
                                    showlegend=False))

    # stations
    for label, sel, symbol in (("Seismometer", lambda t: t.infrasound is None, "triangle-up"),
                               ("Seismometer + infrasound", lambda t: t.infrasound is not None, "diamond")):
        ts = [t for t in res.traces.values() if sel(t)]
        if not ts:
            continue
        stas = [t.seismic.station for t in ts]
        fig.add_trace(go.Scattergeo(
            lat=[x.lat for x in stas], lon=[x.lon for x in stas], mode="markers+text",
            text=[x.code for x in stas], textposition="top center",
            textfont=dict(size=9, color="#52514e"),
            marker=dict(size=10, symbol=symbol, line=dict(width=1, color="white"),
                        color=[STATION_ON if x.id in used else STATION_OFF for x in stas]),
            name=label,
            hovertext=[f"<b>{x.id}</b> {x.site}<br>{x.kind} · {x.channel}"
                       f"{' + ' + x.infrasound_channel if x.infrasound_channel else ''}"
                       f"<br>{x.source}<br>{'recorded an event' if x.id in used else 'no event'}"
                       for x in stas],
            hoverinfo="text"))

    # catalogue events
    if res.catalog:
        fig.add_trace(go.Scattergeo(
            lat=[c.lat for c in res.catalog], lon=[c.lon for c in res.catalog], mode="markers",
            marker=dict(size=[6 + 3 * max(c.magnitude, 0) for c in res.catalog], symbol="circle-open",
                        color="#52514e", line=dict(width=2)),
            name="Catalogue event (USGS/EMSC)",
            hovertext=[f"<b>{c.source}</b> {c.description}<br>M{c.magnitude:.1f} {c.event_type}"
                       f"<br>{fmt_time(c.time)}<br>depth {c.depth:.0f} km" for c in res.catalog],
            hoverinfo="text"))

    # detected events, one legend entry per class
    for label, color in EVENT_COLORS.items():
        evs = [e for e in res.events if e.label == label]
        if not evs:
            continue
        sizes = [10 + 5 * ((e.ml if e.ml is not None else (e.catalog_match.magnitude if e.catalog_match else 3)))
                 for e in evs]
        fig.add_trace(go.Scattergeo(
            lat=[e.lat for e in evs], lon=[e.lon for e in evs], mode="markers",
            marker=dict(size=sizes, color=color, symbol="star" if label == "Likely explosion" else "circle",
                        line=dict(width=2, color="white")),
            name=label,
            hovertext=[f"<b>{e.id} {e.label}</b><br>{fmt_time(e.origin_time)}<br>{mag_text(e)}"
                       f"<br>{e.lat:.2f}°N {e.lon:.2f}°E, depth {e.depth:.0f} km"
                       + (f"<br>explosion probability {e.p_explosion * 100:.0f}%" if e.kind == "local" else "")
                       for e in evs],
            hoverinfo="text"))

    # direction arrows towards distant events (drawn from the network's edge)
    lat0 = np.mean([t.seismic.station.lat for t in res.traces.values()])
    lon0 = np.mean([t.seismic.station.lon for t in res.traces.values()])
    for ev in res.events:
        if ev.kind == "distant":
            pts = [destination(lat0, lon0, ev.back_azimuth, k) for k in np.linspace(450, 850, 20)]
            fig.add_trace(go.Scattergeo(
                lat=[a for a, _ in pts], lon=[b for _, b in pts], mode="lines+markers",
                marker=dict(size=[0] * 19 + [14], symbol="triangle-up", angleref="previous",
                            color=EVENT_COLORS[ev.label]),
                line=dict(width=3, color=EVENT_COLORS[ev.label]), showlegend=False,
                hovertext=f"{ev.id}: distant earthquake {ev.distance_deg * KM_PER_DEG:,.0f} km away "
                          f"(direction {ev.back_azimuth:.0f}°)", hoverinfo="text"))

    wide = view.startswith("Wide")
    fig.update_geos(
        projection_type="mercator" if not wide else "natural earth",
        lataxis_range=None if wide else [r["min_lat"] - 0.5, r["max_lat"] + 0.5],
        lonaxis_range=None if wide else [r["min_lon"] - 0.5, r["max_lon"] + 0.5],
        center=dict(lat=(r["min_lat"] + r["max_lat"]) / 2 + (8 if wide else 0),
                    lon=(r["min_lon"] + r["max_lon"]) / 2 + (8 if wide else 0)),
        projection_scale=2.2 if wide else 1,
        showland=True, landcolor="#ebe6d6", showocean=True, oceancolor="#b9d3ea",
        showcountries=True, countrycolor="#9a978c", showcoastlines=True, coastlinecolor="#6f6d66",
        showlakes=False, resolution=110 if wide else 50)
    fig.update_layout(height=720, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    # basemap served locally (static/topojson) so the map also works offline
    st.plotly_chart(fig, width="stretch", config={"topojsonURL": "app/static/topojson/"})
    st.caption("▲ seismometer · ◆ seismometer + infrasound microphone (Raspberry Shake & Boom) · "
               "dark = recorded an event · ★ explosion · ● earthquake · dotted ring = 95% location "
               "uncertainty · thin lines = stations used to locate the event · green arrow = direction of a "
               "distant earthquake (switch to the wide view to see it).")


# ---------------------------------------------------------------------------
# Waveforms
# ---------------------------------------------------------------------------
def waveform_tab(res):
    s = res.settings
    vm = s.velocity
    locals_ = [e for e in res.events if e.kind == "local"]
    choices = ["Whole time window"] + [event_label(e) for e in res.events]
    pick = st.selectbox("Show", choices, index=1 if locals_ else 0, key="wf_event")
    ev = next((e for e in res.events if event_label(e) == pick), None)

    if ev is None or ev.kind == "distant":
        overview_plot(res, ev)
    else:
        record_section(res, ev)

    st.subheader("One station in detail")
    ids = sorted(res.traces)
    default = ev.stations[0] if ev is not None and ev.stations else ids[0]
    sid = st.selectbox("Station", ids, index=ids.index(default), key=f"wf_station_{default}",
                       format_func=lambda i: f"{i} – {res.traces[i].seismic.station.site or res.traces[i].seismic.station.kind}")
    station_detail(res, sid, ev)


def overview_plot(res, ev):
    traces = sorted(res.traces.values(), key=lambda t: -t.seismic.station.lat)
    fig = go.Figure()
    for k, t in enumerate(traces):
        w = t.seismic
        y = t.filtered / (np.max(np.abs(t.filtered)) + 1e-30) * 0.45
        x, yy = envelope(w.times() - res.t_start, y)
        fig.add_trace(go.Scattergl(x=x, y=yy - k, mode="lines", line=dict(width=0.7, color="#3d3c38"),
                                   showlegend=False, hoverinfo="skip"))
    names = [f"{t.seismic.station.code}" for t in traces]
    for e in res.events:
        fig.add_vline(x=e.origin_time - res.t_start, line=dict(color=EVENT_COLORS[e.label], width=1.5, dash="dash"),
                      annotation_text=f"{e.icon} {e.id}", annotation_position="top")
    for phase in ("P", "S", "Air"):
        qs = [(q, names.index(q.station.code)) for e in res.events for q in e.picks + e.air_picks
              if q.phase == phase and q.station.code in names and not q.channel.startswith("HD")]
        if qs:
            fig.add_trace(go.Scatter(x=[q.time - res.t_start for q, _ in qs], y=[-k for _, k in qs],
                                     mode="markers", name=f"{phase} pick",
                                     marker=dict(symbol="line-ns", size=16, line=dict(width=2.5, color=PHASE_COLORS[phase]))))
    fig.update_layout(height=max(400, 32 * len(traces)), margin=dict(l=10, r=10, t=30, b=40),
                      xaxis_title=f"Seconds after {fmt_time(res.t_start)}",
                      yaxis=dict(tickvals=[-k for k in range(len(traces))], ticktext=names, showgrid=False),
                      legend=dict(orientation="h", y=1.02, x=0))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Every station, north (top) to south, band-pass {res.settings.detection.freqmin:g}–"
               f"{res.settings.detection.freqmax:g} Hz, each trace scaled to its own maximum. "
               "Dashed lines = event origin times.")


def _section(traces, ev, t_lo, t_hi, spacing, color, which="seismic"):
    """Seismogram traces drawn at their epicentral distance (list of (dist, x, y))."""
    out = []
    for d, t in traces:
        w = t.seismic if which == "seismic" else t.infrasound
        data = t.filtered if which == "seismic" else t.infra_filtered
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
        qs = [q for q in ev.picks + ev.air_picks if q.phase == phase]
        if qs:
            fig.add_trace(go.Scatter(
                x=[q.time - ev.origin_time for q in qs],
                y=[float(haversine_km(ev.lat, ev.lon, q.station.lat, q.station.lon)) for q in qs],
                mode="markers", name=f"{phase} pick",
                marker=dict(symbol="line-ns", size=18, line=dict(width=3, color=PHASE_COLORS[phase])),
                hovertext=[f"{q.station.id} {phase} {fmt_time(q.time, False)}" for q in qs],
                hoverinfo="text"))


def record_section(res, ev):
    vm = res.settings.velocity
    p = res.settings.detection
    rows = sorted(((float(haversine_km(ev.lat, ev.lon, t.seismic.station.lat, t.seismic.station.lon)), t)
                   for t in res.traces.values()), key=lambda r: r[0])
    dmax = rows[-1][0]
    t_room = res.t_end - ev.origin_time
    default = int(min(max(float(physics.s_time(dmax, ev.depth, vm)) + 60, 60), t_room))
    tmax = st.slider("Seconds after origin", 30, int(max(t_room, 31)), default, 10, key=f"rs_tmax_{ev.id}")
    spacing = max(dmax / max(len(rows), 1), 12)

    fig = go.Figure(_section(rows, ev, -10, tmax, spacing, "#3d3c38"))
    for d, t in rows:
        fig.add_annotation(x=-10, y=d, text=t.seismic.station.code, showarrow=False, xanchor="right",
                           font=dict(size=9, color="#52514e"))
    dd = np.linspace(0, dmax * 1.05, 200)
    for name, tt in (("P", physics.p_time(dd, ev.depth, vm)), ("S", physics.s_time(dd, ev.depth, vm))):
        fig.add_trace(go.Scatter(x=tt, y=dd, mode="lines", name=f"Predicted {name}",
                                 line=dict(color=PHASE_COLORS[name], width=2, dash="dot")))
    _pick_markers(ev, ("P", "S"), fig)
    fig.update_layout(height=max(480, 30 * len(rows)), margin=dict(l=50, r=10, t=30, b=40),
                      xaxis=dict(title="Seconds after origin time", range=[-12, tmax]),
                      yaxis=dict(title="Distance from event (km)", range=[-spacing, dmax + spacing]),
                      legend=dict(orientation="h", y=1.02, x=0))
    st.plotly_chart(fig, width="stretch")
    st.caption("**Record section**: each seismogram is drawn at its distance from the event "
               f"(band-pass {p.freqmin:g}–{p.freqmax:g} Hz, each scaled to its own maximum). "
               "Arrivals line up along slanted lines whose slope is the wave speed: P (fast, ~6–8 km/s) "
               "then S (~3.5 km/s). The dotted lines are the Earth model's predictions.")

    # air-blast section for stations with infrasound microphones
    air_rows = [(d, t) for d, t in rows if t.infrasound is not None and d <= p.max_air_range_km]
    if not air_rows:
        return
    st.markdown("**Air-blast section** – infrasound microphones (air pressure)")
    dmax_a = air_rows[-1][0]
    t_hi = min(dmax_a / p.celerity_min + 40, t_room)
    sp = max(dmax_a / max(len(air_rows), 1), 15)
    fig = go.Figure(_section(air_rows, ev, -10, t_hi, sp, PHASE_COLORS["Air"], "infra"))
    for d, t in air_rows:
        fig.add_annotation(x=-10, y=d, text=t.seismic.station.code, showarrow=False, xanchor="right",
                           font=dict(size=9, color="#52514e"))
    dd = np.linspace(0, dmax_a * 1.1, 100)
    fig.add_trace(go.Scatter(x=np.concatenate([dd / p.celerity_max, (dd / p.celerity_min)[::-1]]),
                             y=np.concatenate([dd, dd[::-1]]), fill="toself", mode="none",
                             fillcolor="rgba(27,175,122,0.12)",
                             name=f"Expected air arrival ({p.celerity_min * 1000:.0f}–{p.celerity_max * 1000:.0f} m/s)"))
    fig.add_trace(go.Scatter(x=dd / vm.c_air, y=dd, mode="lines", name=f"Speed of sound ({vm.c_air * 1000:.0f} m/s)",
                             line=dict(color=PHASE_COLORS["Air"], dash="dot", width=2)))
    _pick_markers(ev, ("Air",), fig)
    fig.update_layout(height=max(320, 60 * len(air_rows)), margin=dict(l=50, r=10, t=30, b=40),
                      xaxis=dict(title="Seconds after origin time", range=[-12, t_hi]),
                      yaxis=dict(title="Distance (km)", range=[-sp, dmax_a + sp]),
                      legend=dict(orientation="h", y=1.05, x=0))
    st.plotly_chart(fig, width="stretch")
    st.caption("Sound travels ~20× slower than seismic waves, so an explosion's air wave reaches a "
               "microphone 200 km away about 10 minutes later. A pulse inside the shaded band at "
               "several stations is strong evidence of a surface or air explosion.")


def station_detail(res, sid, ev):
    t = res.traces[sid]
    p = res.settings.detection
    w = t.seismic
    n_rows = 3 if t.infrasound is not None else 2
    titles = [f"Ground velocity (µm/s), {p.freqmin:g}–{p.freqmax:g} Hz", "STA/LTA ratio"]
    if n_rows == 3:
        titles.append("Air pressure (Pa) – infrasound")
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=titles)
    times = pd.to_datetime(w.times(), unit="s", utc=True)
    x, y = envelope(np.arange(len(w.data)), t.filtered * 1e6, 4000)
    fig.add_trace(go.Scattergl(x=times[x.astype(int)], y=y, mode="lines", line=dict(width=0.8, color="#3d3c38"),
                               name="velocity", showlegend=False), row=1, col=1)
    x, y = envelope(np.arange(len(t.cft)), t.cft, 4000)
    fig.add_trace(go.Scattergl(x=times[x.astype(int)], y=y, mode="lines", line=dict(width=1, color=PHASE_COLORS["P"]),
                               name="STA/LTA", showlegend=False), row=2, col=1)
    fig.add_hline(y=p.trigger_on, line=dict(color="#e34948", dash="dash", width=1), row=2, col=1,
                  annotation_text="ON", annotation_position="right")
    fig.add_hline(y=p.trigger_off, line=dict(color="#8a8984", dash="dash", width=1), row=2, col=1,
                  annotation_text="OFF", annotation_position="right")
    for q in res.picks:
        if q.station.id == sid:
            fig.add_vline(x=datetime.fromtimestamp(q.time, timezone.utc),
                          line=dict(color=PHASE_COLORS.get(q.phase, "#8a8984"), width=1.2,
                                    dash="solid" if q.event_id else "dot"))
    if n_rows == 3:
        iw = t.infrasound
        itimes = pd.to_datetime(iw.times(), unit="s", utc=True)
        x, y = envelope(np.arange(len(iw.data)), t.infra_filtered, 4000)
        fig.add_trace(go.Scattergl(x=itimes[x.astype(int)], y=y, mode="lines",
                                   line=dict(width=0.8, color=PHASE_COLORS["Air"]), showlegend=False), row=3, col=1)
        if ev is not None:
            for q in ev.air_picks:
                if q.station.id == sid:
                    fig.add_vline(x=datetime.fromtimestamp(q.time, timezone.utc),
                                  line=dict(color=PHASE_COLORS["Air"], width=2))
    fig.update_yaxes(type="log", row=2, col=1)
    fig.update_layout(height=220 * n_rows + 60, margin=dict(l=10, r=40, t=40, b=30))
    st.plotly_chart(fig, width="stretch")
    picks = [q for q in res.picks if q.station.id == sid]
    st.caption(f"{len(picks)} trigger(s) at {sid}. Vertical lines: solid = part of an event "
               "(blue P, orange S, green air wave), dotted = local noise. A trigger starts when the "
               "STA/LTA ratio rises above the red ON line.")


# ---------------------------------------------------------------------------
# Event analysis
# ---------------------------------------------------------------------------
def analysis_tab(res):
    if not res.events:
        st.info("No events to analyse.")
        return
    labels = [event_label(e) for e in res.events]
    ev = res.events[labels.index(st.selectbox("Event", labels, key="an_event"))]
    st.subheader(event_title(ev))
    if ev.kind == "distant":
        distant_analysis(res, ev)
        return
    vm = res.settings.velocity

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
        st.markdown(f"**Verdict: {ev.label}.** Below 30% → earthquake, above 70% → explosion, "
                    "in between → not enough evidence.")
        for n in ev.notes:
            st.warning(n)
    with c2:
        ev_list = ev.evidence
        names = ["Starting point (prior)"] + [e.name for e in ev_list]
        vals = [PRIOR_BIAS] + [e.contribution for e in ev_list]
        meas = ["Natural earthquakes are more common than explosions"] + [e.measurement for e in ev_list]
        fig = go.Figure()
        for sign, name, color in ((1, "Points to explosion", EVENT_COLORS["Likely explosion"]),
                                  (-1, "Points to earthquake", EVENT_COLORS["Likely earthquake"])):
            idx = [i for i, v in enumerate(vals) if (v > 0 if sign > 0 else v <= 0)]
            fig.add_trace(go.Bar(y=[names[i] for i in idx], x=[vals[i] for i in idx], orientation="h",
                                 name=name, marker=dict(color=color, cornerradius=4),
                                 hovertext=[meas[i] for i in idx], hoverinfo="text+x"))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=30), barmode="relative",
                          title="What pushed the verdict (weight × score)",
                          xaxis=dict(title="← earthquake-like      explosion-like →", zeroline=True,
                                     zerolinecolor="#8a8984"),
                          yaxis=dict(autorange="reversed", categoryorder="array", categoryarray=names),
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")

    st.dataframe(pd.DataFrame([{
        "Clue": e.name, "Measured": e.measurement, "Score (−2…+2)": round(e.score, 2),
        "Weight": e.weight, "Contribution": round(e.contribution, 2), "Why it matters": e.explanation}
        for e in ev.evidence]), hide_index=True, width="stretch")

    st.subheader("Where and when")
    a, b, c, d = st.columns(4)
    a.metric("Origin time (UTC)", fmt_time(ev.origin_time, False))
    metric(b, "Epicentre", f"{ev.lat:.2f}°N, {ev.lon:.2f}°E", f"± {ev.error_km:.0f} km (95%)")
    metric(c, "Depth", f"{ev.depth:.0f} km",
           f"95%: {ev.depth_min:.0f}–{ev.depth_max:.0f} km" + ("" if ev.depth_constrained else " (unresolved)"))
    metric(d, "Fit (RMS residual)", f"{ev.rms:.2f} s", f"{ev.n_stations} stations")
    st.caption(f"Location in words: **{describe(ev.lat, ev.lon)}**.")
    travel_time_plot(res, ev)

    st.subheader("How big")
    if ev.ml is not None:
        e_j = physics.seismic_energy_joules(ev.ml)
        y_c = float(physics.explosion_yield_tons(ev.ml))
        y_s = float(physics.explosion_yield_tons(ev.ml, physics.SURFACE_BLAST_COUPLING))
        a, b, c, d = st.columns(4)
        metric(a, "Local magnitude", f"ML {ev.ml:.1f}",
                 f"{len(ev.ml_stations)} stations, spread ±{np.std([m['ml'] for m in ev.ml_stations]):.1f}")
        metric(b, "Seismic energy", fmt_energy(e_j), f"= {fmt_tons(e_j / physics.JOULES_PER_TON_TNT)} of TNT")
        metric(c, "Yield if buried explosion", fmt_tons(y_c), "fully coupled to rock")
        metric(d, "Yield if surface blast", fmt_tons(y_s),
                 f"assumes {physics.SURFACE_BLAST_COUPLING:.0%} ground coupling")
        st.caption("Yield estimates are order-of-magnitude only (factor 3–10 uncertainty): "
                   "they depend strongly on how well the blast couples to the ground. "
                   "For earthquakes the yield numbers are only a size comparison.")
        st.dataframe(pd.DataFrame([{
            "Station": m["station"], "Site": m["site"], "Distance (km)": round(m["dist_km"]),
            "Wood-Anderson amplitude (nm)": round(m["amp_nm"], 1), "ML": round(m["ml"], 2)}
            for m in sorted(ev.ml_stations, key=lambda m: m["dist_km"])]),
            hide_index=True, width="stretch")
    else:
        st.info("Magnitude could not be measured (signals too weak or too short).")

    if ev.air_picks:
        st.subheader("Air-blast arrivals")
        rows = []
        for q in ev.air_picks:
            d = float(haversine_km(ev.lat, ev.lon, q.station.lat, q.station.lon))
            rows.append({"Station": q.station.id, "Sensor": "infrasound" if q.channel.startswith(("HD", "BD"))
                         else "seismometer (air-coupled)", "Distance (km)": round(d),
                         "Arrival (UTC)": fmt_time(q.time, False),
                         "Travel time (s)": round(q.time - ev.origin_time, 1),
                         "Celerity (m/s)": round(d / (q.time - ev.origin_time) * 1000)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Celerity = distance ÷ travel time. Sound in air travels ~300–345 m/s, so matching "
                   "celerities prove the energy went through the atmosphere – a hallmark of a surface "
                   "or air explosion.")


def travel_time_plot(res, ev):
    vm = res.settings.velocity
    rows = [(q, float(haversine_km(ev.lat, ev.lon, q.station.lat, q.station.lon)))
            for q in ev.picks + ev.air_picks if q.phase in ("P", "S", "Air")]
    if not rows:
        return
    air = [r for r in rows if r[0].phase == "Air"]
    panels = [("P", "S")] + ([("Air",)] if air else [])
    fig = make_subplots(rows=1, cols=len(panels), horizontal_spacing=0.1,
                        subplot_titles=["Seismic waves (through the ground)"] +
                        (["Air wave (through the atmosphere)"] if air else []))
    for col, phases in enumerate(panels, start=1):
        sel = [(q, d) for q, d in rows if q.phase in phases]
        dd = np.linspace(0, max(d for _, d in sel) * 1.1, 200)
        if phases == ("Air",):
            pd_ = res.settings.detection
            fig.add_trace(go.Scatter(x=np.concatenate([dd, dd[::-1]]),
                                     y=np.concatenate([dd / pd_.celerity_max, (dd / pd_.celerity_min)[::-1]]),
                                     fill="toself", mode="none", fillcolor="rgba(27,175,122,0.12)",
                                     name="Plausible air arrivals (wind, temperature)"), row=1, col=col)
        for ph in phases:
            tt = {"P": physics.p_time(dd, ev.depth, vm), "S": physics.s_time(dd, ev.depth, vm),
                  "Air": physics.air_time(dd, vm.c_air)}[ph]
            fig.add_trace(go.Scatter(x=dd, y=tt, mode="lines", name=f"Predicted {ph}",
                                     line=dict(color=PHASE_COLORS[ph], width=2)), row=1, col=col)
            qs = [(q, d) for q, d in sel if q.phase == ph]
            fig.add_trace(go.Scatter(
                x=[d for _, d in qs], y=[q.time - ev.origin_time for q, _ in qs], mode="markers",
                name=f"Observed {ph}",
                marker=dict(size=9, color=PHASE_COLORS[ph], line=dict(width=2, color="white")),
                hovertext=[f"{q.station.id}: {q.time - ev.origin_time:.1f} s"
                           + (f", residual {q.residual:+.2f} s" if q.residual is not None and ph != "Air" else "")
                           for q, _ in qs], hoverinfo="text"), row=1, col=col)
        fig.update_xaxes(title_text="Distance from epicentre (km)", row=1, col=col)
        fig.update_yaxes(title_text="Travel time (s)", row=1, col=col)
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=40),
                      legend=dict(orientation="h", y=-0.28))
    st.plotly_chart(fig, width="stretch")
    st.caption("Dots = measured arrival times, lines = what the Earth model predicts for this "
               "location. Dots on the lines mean the location explains the data.")


def distant_analysis(res, ev):
    a, b, c, d = st.columns(4)
    a.metric("Direction (back-azimuth)", f"{ev.back_azimuth:.0f}°")
    b.metric("Apparent speed across network", f"{ev.app_velocity:.1f} km/s")
    metric(c, "Distance", f"{ev.distance_deg:.1f}°", f"{ev.distance_deg * KM_PER_DEG:,.0f} km")
    d.metric("Size", mag_text(ev))
    st.markdown(ev.summary)
    if ev.catalog_match:
        cm = ev.catalog_match
        st.success(f"Matched catalogue event: **{cm.description or 'earthquake'}**, M{cm.magnitude:.1f} "
                   f"{cm.mag_type}, depth {cm.depth:.0f} km, {fmt_time(cm.time)} ({cm.source}).")
    else:
        st.info("No catalogue match (turn on the catalogue cross-check in real-data mode).")
    lat0 = np.mean([q.station.lat for q in ev.picks])
    lon0 = np.mean([q.station.lon for q in ev.picks])
    rows = []
    for q in ev.picks:
        dist = float(haversine_km(ev.lat, ev.lon, q.station.lat, q.station.lon)) / KM_PER_DEG
        pred = ev.origin_time + float(physics.tele_p_time(dist, ev.depth))
        rows.append({"Station": q.station.id, "Distance (°)": round(dist, 2),
                     "Observed P (UTC)": fmt_time(q.time, False),
                     "Predicted P (UTC)": fmt_time(pred, False), "Residual (s)": round(q.time - pred, 2)})
    st.dataframe(pd.DataFrame(rows).sort_values("Distance (°)"), hide_index=True, width="stretch")
    st.caption("Waves from far away arrive at the network almost as a flat wavefront. Their fast "
               "apparent speed (>8 km/s) shows they came up steeply from deep in the mantle, i.e. "
               "from a distant source, and the arrival order across the stations gives the direction.")


# ---------------------------------------------------------------------------
# Formulas
# ---------------------------------------------------------------------------
def formulas_tab(res):
    vm = res.settings.velocity
    p = res.settings.detection
    ev = next((e for e in res.events if e.kind == "local"), None)
    st.markdown("Every number in this app comes from the formulas below. Where possible a **worked "
                "example** plugs in the values measured for "
                + (f"**{ev.id}** ({ev.label.lower()})." if ev else "a detected event."))

    with st.expander("① Detecting a signal: STA/LTA trigger", expanded=True):
        st.latex(r"\mathrm{STA}(t)=\frac{1}{N_s}\sum_{k=t-N_s+1}^{t}x_k^2 \qquad "
                 r"\mathrm{LTA}(t)=\frac{1}{N_l}\sum_{k=t-N_s-N_l+1}^{t-N_s}x_k^2 \qquad "
                 r"R(t)=\frac{\mathrm{STA}(t)}{\mathrm{LTA}(t)}")
        st.markdown(f"The **short-term average** (last {p.sta:g} s) of the filtered signal's energy is "
                    f"compared with the **long-term average** (the {p.lta:g} s before it). Background "
                    f"noise gives R ≈ 1; a sudden arrival makes R jump. A trigger starts when "
                    f"R > **{p.trigger_on:g}** and ends when R < **{p.trigger_off:g}**. The signal is "
                    f"first band-pass filtered to {p.freqmin:g}–{p.freqmax:g} Hz (Butterworth, "
                    r"$|H(f)|^2 = 1/(1+(f/f_c)^{2n})$ per corner) to remove ocean microseisms and hum.")
    with st.expander("② Timing the arrival precisely: AIC picker"):
        st.latex(r"\mathrm{AIC}(k)=k\,\log\!\big(\mathrm{var}(x_{1..k})\big)+(N-k-1)\,\log\!\big(\mathrm{var}(x_{k+1..N})\big)")
        st.markdown("Around each trigger the waveform is split into 'noise before' and 'signal after' "
                    "at every possible sample k. The onset is where the split is best, i.e. where "
                    "AIC is smallest (Maeda, 1985). Accuracy ≈ a few hundredths of a second.")
    with st.expander("③ Distances on the Earth: haversine"):
        st.latex(r"d = 2R\,\arcsin\sqrt{\sin^2\frac{\Delta\varphi}{2}+\cos\varphi_1\cos\varphi_2\sin^2\frac{\Delta\lambda}{2}},\qquad R=6371\ \mathrm{km}")
    with st.expander("④ Travel times: how long waves take", expanded=True):
        st.markdown(f"Earth model: crust {vm.moho:g} km thick with Vp = {vm.vp:g} km/s, Vs = {vm.vs:g} km/s, "
                    f"over mantle with Pn = {vm.vpn:g} km/s, Sn = {vm.vsn:g} km/s.")
        st.latex(r"t_{Pg}=\frac{\sqrt{d^2+h^2}}{V_p}\qquad "
                 r"t_{Pn}=\frac{d}{V_{pn}}+(2H-h)\sqrt{\frac{1}{V_p^2}-\frac{1}{V_{pn}^2}}\qquad "
                 r"t_P=\min(t_{Pg},t_{Pn})")
        st.markdown("The direct wave (Pg) travels through the crust; beyond ~150–200 km the head wave "
                    "(Pn) that races along the top of the faster mantle overtakes it. S waves use the "
                    "same formulas with Vs and Vsn.")
        st.latex(r"d \approx \Delta t_{S-P}\cdot\frac{V_pV_s}{V_p-V_s}")
        k = vm.vp * vm.vs / (vm.vp - vm.vs)
        st.markdown(f"**One-station distance rule:** every second between the P and S arrivals ≈ "
                    f"**{k:.1f} km** of distance.")
        if ev:
            sp = [(q.station, q.time) for q in ev.picks if q.phase == "S"]
            pp = {q.station.id: q.time for q in ev.picks if q.phase == "P"}
            both = [(s_, ts, pp[s_.id]) for s_, ts in sp if s_.id in pp]
            if both:
                s_, ts, tp = min(both, key=lambda x: x[1] - x[2])
                dt = ts - tp
                true_d = float(haversine_km(ev.lat, ev.lon, s_.lat, s_.lon))
                st.info(f"Worked example ({s_.id}): S − P = {dt:.1f} s → d ≈ {dt:.1f} × {k:.2f} = "
                        f"**{physics.sp_distance_km(dt, vm.vp, vm.vs):.0f} km** "
                        f"(network location gives {true_d:.0f} km).")
    with st.expander("⑤ Locating the source: grid search"):
        st.latex(r"\chi^2(\varphi,\lambda,h)=\sum_i\left(\frac{t_i^{obs}-t_0-T_i(\varphi,\lambda,h)}{\sigma_i}\right)^2,"
                 r"\qquad t_0=\frac{\sum_i w_i\,(t_i^{obs}-T_i)}{\sum_i w_i},\ w_i=\sigma_i^{-2}")
        st.markdown(f"Every point on a 0.2° grid (then 0.02°) and every depth 0–40 km is tried; the "
                    f"origin time t₀ follows directly. Pick uncertainty σ = {p.pick_sigma:g} s for P, "
                    f"{2 * p.pick_sigma:g} s for S. The 95% confidence region is where "
                    r"$\chi^2 \le \chi^2_{min} + 5.99$ (2 unknowns), and the depth range where "
                    r"$\chi^2 \le \chi^2_{min}+3.84$. First, a *phase-free association* decides which "
                    "triggers belong to the event and whether each is a P or an S wave: "
                    r"$r_i=\min(|t_i-t_0-T_{P,i}|,\ |t_i-t_0-T_{S,i}|)$.")
        if ev:
            st.info(f"{ev.id}: {len(ev.picks)} arrivals at {ev.n_stations} stations, RMS residual "
                    f"{ev.rms:.2f} s, 95% epicentre uncertainty ±{ev.error_km:.0f} km, depth "
                    f"{ev.depth_min:.0f}–{ev.depth_max:.0f} km.")
    with st.expander("⑥ Distant earthquakes: global travel times"):
        st.latex(r"t_i = t_0 + T_P(\Delta_i)\quad(\text{iasp91 table}),\qquad "
                 r"v_{app}=\frac{1}{|\vec p|},\ \ \vec p=\left(\frac{\partial t}{\partial x},\frac{\partial t}{\partial y}\right)")
        st.markdown("A distant quake's wavefront crosses the Gulf almost flat, at an *apparent* speed "
                    "of 8–25 km/s (faster = farther, because the ray comes up more steeply). "
                    "Fitting the arrival times with the global travel-time table over a world-wide "
                    "grid gives direction and distance; back-azimuth = direction of −p.")
    with st.expander("⑦ Magnitude: how big (Richter-type local magnitude)", expanded=True):
        st.latex(r"M_L=\log_{10}A+1.11\log_{10}R+0.00189\,R-2.09")
        st.latex(r"H_{WA}(s)=\frac{s^2}{(s-p_1)(s-p_2)},\quad p_{1,2}=-6.283\pm4.712i")
        st.markdown("A = largest amplitude (nm) of the record a **Wood-Anderson** seismometer "
                    "(the instrument Richter used) would have written, simulated from the ground "
                    "velocity; R = distance to the source in km (IASPEI standard, Hutton & Boore 1987). "
                    "The event magnitude is the median over stations. Each +1 in magnitude = 10× the "
                    "shaking amplitude and ~32× the energy.")
        if ev and ev.ml_stations:
            m = min(ev.ml_stations, key=lambda m: abs(m["ml"] - ev.ml))
            st.info(f"Worked example ({m['station']}): A = {m['amp_nm']:.1f} nm, R = {m['hypo_km']:.0f} km → "
                    f"ML = log₁₀({m['amp_nm']:.1f}) + 1.11·log₁₀({m['hypo_km']:.0f}) + 0.00189·{m['hypo_km']:.0f} − 2.09 "
                    f"= {np.log10(m['amp_nm']):.2f} + {1.11 * np.log10(m['hypo_km']):.2f} + "
                    f"{0.00189 * m['hypo_km']:.2f} − 2.09 = **{m['ml']:.2f}**")
    with st.expander("⑧ Energy and explosive yield"):
        st.latex(r"\log_{10}E_s = 1.5\,M + 4.8\quad(\mathrm{J}),\qquad 1\ \text{t TNT}=4.184\times10^{9}\ \mathrm{J}")
        st.latex(r"M = 4.45 + 0.75\,\log_{10}Y_{kt}\ \Rightarrow\ Y_{kt}=10^{(M-4.45)/0.75},"
                 r"\qquad Y_{surface}\approx \frac{Y}{\varepsilon},\ \varepsilon\approx" +
                 f"{physics.SURFACE_BLAST_COUPLING:g}")
        st.markdown("The first line is the Gutenberg–Richter energy of the seismic waves. The second "
                    "is the standard magnitude–yield relation for a fully coupled (underground) "
                    "explosion; a blast on the surface or in the air puts only a few percent (ε) of "
                    "its energy into the ground. Calibration: the 2020 Beirut port explosion "
                    "(~0.5–1 kt TNT) registered ML ≈ 3.3, which this relation reproduces with ε ≈ 0.05.")
        if ev and ev.ml is not None:
            st.info(f"{ev.id}: ML {ev.ml:.2f} → E = 10^(1.5·{ev.ml:.2f}+4.8) = "
                    f"{fmt_energy(physics.seismic_energy_joules(ev.ml))}; fully-coupled yield "
                    f"{fmt_tons(float(physics.explosion_yield_tons(ev.ml)))}, surface blast ≈ "
                    f"{fmt_tons(float(physics.explosion_yield_tons(ev.ml, physics.SURFACE_BLAST_COUPLING)))}.")
    with st.expander("⑨ The air-blast (infrasound) wave"):
        st.latex(r"c = 331.3\sqrt{1+\frac{T}{273.15}}\ \mathrm{m/s},\qquad t_{air}=t_0+\frac{d}{c_{eff}},"
                 r"\quad c_{eff}\in[" + f"{p.celerity_min * 1000:.0f},{p.celerity_max * 1000:.0f}" + r"]\ \mathrm{m/s}")
        st.markdown(f"At {vm.air_temp_c:g} °C sound travels at {vm.c_air * 1000:.0f} m/s. Winds and "
                    "temperature layers make the effective speed along the path ('celerity') vary, so "
                    f"the app searches each infrasound microphone within {p.max_air_range_km:g} km "
                    "for a pressure pulse between the fastest and slowest plausible arrival.")
    with st.expander("⑩ Explosion or earthquake? Combining the clues", expanded=True):
        st.latex(r"P(\text{explosion})=\frac{1}{1+e^{-z}},\qquad z=b+\sum_i w_i\,s_i")
        st.markdown(f"Each clue gives a score s from −2 (earthquake-like) to +2 (explosion-like), "
                    f"multiplied by its weight w; b = {PRIOR_BIAS} because natural earthquakes are more "
                    "common. Main clues:")
        st.latex(r"\text{P/S ratio: } \log_{10}\frac{A_P}{A_S}\ \text{(4–12 Hz)}\quad\begin{cases}"
                 r"\gtrsim 0 & \text{explosion (pushes outward: strong P)}\\"
                 r"\approx -0.5 & \text{earthquake (shear slip: strong S)}\end{cases}")
        st.markdown("- **Depth**: explosions are at the surface; anything deeper than ~6 km is natural.\n"
                    "- **First motion**: an explosion pushes the ground UP at every station; a fault "
                    "gives a mix of up and down.\n"
                    "- **Air wave**: only surface/air explosions send a pressure wave through the air.\n"
                    "- **Catalogue**: agencies (USGS/EMSC) sometimes already classified the event.\n"
                    "- **Time of day**: blasting is usually in working hours (weak clue).")
        if ev:
            z = PRIOR_BIAS + sum(e.contribution for e in ev.evidence)
            terms = " ".join(f"{'+' if e.contribution >= 0 else '−'} {abs(e.contribution):.2f}" for e in ev.evidence)
            st.info(f"{ev.id}: z = {PRIOR_BIAS} {terms} = {z:.2f} → P = 1/(1+e^({-z:.2f})) = "
                    f"**{ev.p_explosion * 100:.0f}%** → {ev.label}.")


# ---------------------------------------------------------------------------
# Stations & triggers, guide
# ---------------------------------------------------------------------------
def stations_tab(res):
    rows = []
    for sid, t in sorted(res.traces.items()):
        s = t.seismic.station
        n = [q for q in res.picks if q.station.id == sid]
        rows.append({"Station": sid, "Site": s.site, "Type": s.kind,
                     "Channels": s.channel + (f" + {s.infrasound_channel}" if t.infrasound is not None else ""),
                     "Lat": round(s.lat, 3), "Lon": round(s.lon, 3), "Sample rate (Hz)": t.seismic.fs,
                     "Triggers": len(n), "In events": sum(q.event_id is not None for q in n),
                     "Noise (nm/s rms)": round(float(np.median(np.abs(t.filtered))) * 1.4826e9, 1),
                     "Source": s.source})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.subheader("All triggers")
    st.dataframe(pd.DataFrame([{
        "Time (UTC)": fmt_time(q.time), "Station": q.station.id, "Event": q.event_id or "— (noise)",
        "Phase": q.phase if q.event_id else "unassociated", "SNR": round(q.snr, 1),
        "Peak STA/LTA": round(q.peak_ratio, 1), "First motion": {1: "up", -1: "down", 0: "?"}[q.polarity],
        "Residual (s)": None if q.residual is None else round(q.residual, 2)} for q in res.picks]),
        hide_index=True, width="stretch")
    if res.events:
        df = pd.DataFrame([{
            "id": e.id, "type": e.label, "origin_utc": fmt_time(e.origin_time), "lat": round(e.lat, 3),
            "lon": round(e.lon, 3), "depth_km": e.depth, "error_km": round(e.error_km, 1),
            "ml": None if e.ml is None else round(e.ml, 2), "p_explosion": round(e.p_explosion, 3),
            "stations": e.n_stations, "summary": e.summary} for e in res.events])
        st.download_button("⬇ Download events (CSV)", df.to_csv(index=False), "gulfseis_events.csv",
                           "text/csv")
    if res.messages:
        with st.expander("Download log"):
            st.code("\n".join(res.messages))
    truth = st.session_state.get("truth")
    if truth:
        with st.expander("Demo answer key (what was actually simulated)"):
            t0 = synthetic.default_start()
            st.dataframe(pd.DataFrame([{"Type": e.kind, "Name": e.name, "Origin (UTC)": fmt_time(t0 + e.t),
                                        "Lat": e.lat, "Lon": e.lon, "Depth (km)": e.depth,
                                        "Magnitude": e.magnitude} for e in truth]),
                         hide_index=True, width="stretch")


def guide_tab():
    st.markdown("""
### How it works – in five steps
1. **Listen.** Every seismometer records how fast the ground moves (micrometres per second).
   Raspberry Shake & Boom stations also have an infrasound microphone that records air pressure.
2. **Detect.** Each station watches for a sudden jump in shaking (**STA/LTA trigger**).
   One station alone can be fooled by a lorry or a door slamming.
3. **Associate & locate.** When several stations trigger in an order that fits waves spreading from
   one point, that's an **event**. Its position, depth and time are found by trying every point on a
   map grid and keeping the one whose predicted arrival times match best.
4. **Measure.** The size of the waves gives the **magnitude** (Richter-style, ML) and the energy.
5. **Classify.** Physical clues decide **explosion or earthquake**:

| Clue | Explosion 💥 | Earthquake 🌍 |
|---|---|---|
| Depth | at the surface (0–2 km) | usually 5–20 km deep |
| P vs S waves | strong P, weak S | weak P, strong S |
| First movement of the ground | UP at every station | up at some, down at others |
| Sound in the air | a pressure wave arrives minutes later | no delayed air wave |
| Time | often daytime (quarries, construction) | any time |

Events far away (Afghanistan, Turkey, Indonesia …) arrive at all stations almost at once with a
very fast apparent speed and are labelled **🌐 Distant earthquake**.

### Reading the tabs
- **Map** – stations (▲ / ◆), events (★ explosion, ● earthquake), and the 95% uncertainty ring.
- **Waveforms** – the *record section* shows the seismograms ordered by distance: straight lines of
  arrivals are waves travelling outward. The station view shows the trigger at work.
- **Event analysis** – the verdict, which clues pushed it, the location quality and the size.
- **Formulas** – every formula used, with this event's numbers plugged in.

### Honest limitations
- Few public stations exist in parts of the region; small events (below ~ML 2.5) seen by fewer than
  4 stations cannot be located reliably and are left as unassociated triggers.
- The velocity model is a simple average crust; locations are typically good to 5–20 km inside the
  network and worse outside it.
- Explosion/earthquake discrimination is probabilistic. Treat "Likely" as a strong hint, not proof,
  and check official agencies (USGS, EMSC, IRSC, national centres) for confirmation.
- Raspberry Shake data servers sometimes limit how much data can be requested at once; if a
  download fails, reduce the number of stations or the time window.
""")


# ---------------------------------------------------------------------------
def main():
    cfg, run = sidebar()
    if run or st.session_state.get("pending_refresh") or "result" not in st.session_state:
        st.session_state.pending_refresh = False
        run_analysis(cfg)
    res = st.session_state.get("result")
    if res is None:
        st.title("GulfSeis · Explosion & Earthquake Monitor")
        st.info("Choose a data source in the sidebar and press **▶ Run analysis**.")
        if st.session_state.get("log"):
            st.code("\n".join(st.session_state.log))
        return
    header(res)
    tabs = st.tabs(["🗺️ Map", "📈 Waveforms", "🔍 Event analysis", "📐 Formulas",
                    "📡 Stations & triggers", "❓ How it works"])
    with tabs[0]:
        map_tab(res)
    with tabs[1]:
        waveform_tab(res)
    with tabs[2]:
        analysis_tab(res)
    with tabs[3]:
        formulas_tab(res)
    with tabs[4]:
        stations_tab(res)
    with tabs[5]:
        guide_tab()

    if cfg["mode"] == "real" and cfg.get("auto"):
        wait = cfg["auto_min"] * 60 - (time.time() - st.session_state.get("ran_at", 0))
        st.sidebar.caption(f"Next refresh in ~{max(wait, 0) / 60:.0f} min")
        time.sleep(max(wait, 1))
        st.session_state.pending_refresh = True
        st.rerun()


main()
