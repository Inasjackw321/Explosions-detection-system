# GulfSeis – Explosion & Earthquake Monitor for the Persian Gulf

A Python app that uses public seismographs around the Persian (Arabian) Gulf, including
**Raspberry Shake** citizen seismometers and **Raspberry Shake & Boom** infrasound sensors, to
**detect, locate, size and classify** seismic events:

- 💥 **explosions**: surface blasts, industrial accidents, quarry blasts
- 🌍 **local and regional earthquakes**: Zagros, Makran, Oman mountains …
- 🌐 **distant earthquakes**: Afghanistan, Turkey, Indonesia … (direction, distance, catalogue match)

Every step uses standard seismological formulas, and the app shows each one with the
measured numbers plugged in.

## Open the app

```bash
pip install -r requirements.txt
python run_app.py
```

The app opens in your web browser at http://localhost:8501. You can also start it by
double-clicking `start_windows.bat` on Windows or running `./start_mac_linux.sh` on macOS/Linux.
On first run, `run_app.py` installs the requirements if Streamlit is missing.

There are two data modes (sidebar):

| Mode | Needs | What it does |
|---|---|---|
| **Demo** | nothing (works offline) | Simulates 19 stations around the Gulf (network code `XX`) recording an explosion, a Zagros earthquake and a distant Hindu Kush earthquake. You can also build your own test event. |
| **Real data** | internet + ObsPy | Finds stations in the region on the Raspberry Shake (`AM`), EarthScope/IRIS and GEOFON FDSN servers, downloads and instrument-corrects the data, and cross-checks events against USGS/EMSC catalogues. Optional auto-refresh. |

Command line (no GUI): `python -m gulfseis --demo` or `python -m gulfseis --real 30`.

## What you see

- **Event cards**: verdict, explosion probability, time, magnitude and location in words
  (for example "near Asaluyeh").
- **Map**: stations (▲ seismometer, ◆ + infrasound), events (★ explosion, ● earthquake),
  95 % location-uncertainty ring, and the direction arrow to distant earthquakes.
- **Waveforms**: a record section (seismograms ordered by distance, with predicted P/S curves),
  an air-blast section for infrasound, and a per-station view of the STA/LTA trigger at work.
- **Event analysis**: an explosion-probability gauge, a bar chart of the clues behind the verdict,
  a travel-time check, magnitude per station, energy, TNT-equivalent yield, and air-blast celerities.
- **Formulas**: every formula, with worked examples from the detected events.
- **Stations & triggers**: station table, every trigger, a CSV download of the events, and a
  demo answer key.

## How it works

1. **Detection**: Butterworth band-pass filter, then an STA/LTA energy-ratio trigger.
   The onset is refined with the AIC picker.
2. **Association**: phase-free grid association decides which triggers at which stations
   belong to one source, and whether each is a P or an S wave.
3. **Location**: grid search over latitude, longitude and depth in a one-layer crust over a mantle
   model (Pg/Pn, Sg/Sn). It gives 95 % χ² confidence regions for the epicentre and the depth.
4. **Distant events**: a world-wide grid search with iasp91 P travel times gives the direction,
   distance and apparent velocity.
5. **Magnitude**: simulated Wood-Anderson amplitude and the IASPEI (Hutton & Boore) M_L formula.
   Energy comes from log E = 1.5 M + 4.8. Yield comes from M = 4.45 + 0.75 log Y, with a
   surface-coupling factor calibrated on the Beirut 2020 explosion.
6. **Discrimination**: evidence scores are combined with the logistic formula
   P = 1/(1+e^−z). The clues are the P/S amplitude ratio (4–12 Hz), depth,
   first-motion polarity, an air-blast arrival at the speed of sound (infrasound or air-coupled),
   a catalogue match, and the time of day.

## Project layout

```
app.py                  Streamlit user interface
run_app.py              launcher (python run_app.py)
gulfseis/
  config.py             region, Earth model, detection settings, data centres
  physics.py            travel times, magnitude, energy, yield (all formulas)
  detection.py          filtering, STA/LTA, AIC picker, first motion
  location.py           association, local grid search, distant-event locator
  discrimination.py     explosion vs earthquake evidence and scoring
  pipeline.py           detect → associate → locate → size → classify
  data_sources.py       FDSN (Raspberry Shake, EarthScope, GEOFON), USGS/EMSC catalogues
  synthetic.py          realistic simulated data for the demo
  places.py             reference towns for "23 km SW of Bushehr"-style descriptions
static/topojson/        bundled map outlines (MIT, from plotly/sane-topojson) so the map works offline
tests/                  pytest suite (python -m pytest)
```

## Limitations

- Public station coverage in the region is uneven. Events seen by fewer than 4 stations
  (usually below about ML 2.5) are not located; they stay listed as unassociated triggers.
- The Earth model is a simple average crust. Expect location errors of about 5–20 km inside
  the network, and larger errors outside it.
- The discrimination is probabilistic and the yield estimates are order-of-magnitude.
  Confirm important events with official agencies (USGS, EMSC, national seismological centres).
- Raspberry Shake servers may limit request sizes. If downloads fail, reduce the station count
  or the time window.
