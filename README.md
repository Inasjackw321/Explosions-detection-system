# GulfSeis – Explosion & Noise Monitor for the Persian Gulf

A Python app that downloads **real recordings** from public seismometers and infrasound microphones
in and around the Persian (Arabian) Gulf. It then sorts every signal inside the monitored area into
one of these:

- 💥 **Likely / possible explosion**
- 🌍 **Earthquake**
- ❓ **Uncertain**
- 🔇 **Noise**: a trigger at one station that nothing confirms, such as traffic, machinery, wind or people.

Events whose source is outside the area, such as inland Zagros or distant earthquakes, are listed
separately.

The **monitored area** is the green outline: from Kuwait along the head of the Gulf, down the Iranian
coast to the Strait of Hormuz and the Gulf of Oman (Sohar), then back along the UAE, Qatar, Bahrain and
Saudi coasts. It is defined in `gulfseis/region.py`.

## Open the app

```bash
pip install -r requirements.txt
python run_app.py
```

The app opens in your browser at http://localhost:8501. You can also double-click
`start_windows.bat` on Windows or run `./start_mac_linux.sh` on macOS/Linux. It needs internet access
to the data centres.

### Three ways to look

| Mode | Use it for |
|---|---|
| **Recent hours** | the last 1–24 h, with optional auto-refresh |
| **Time range** | any past period of up to 24 h |
| **Check a known event** | you know roughly *when* and *where* something happened, for example an explosion near Kuwait. The app shows whether it was detected. If it wasn't, it shows each station's signal/noise at the moment the waves should have arrived, and plots every recording with the predicted P, S and air-wave arrival times. |

You can enter and display times in UTC, Kuwait/Saudi/Qatar/Bahrain/Iraq time (UTC+3), Iran time
(UTC+3:30) or UAE/Oman time (UTC+4).

Command line: `python -m gulfseis --hours 3`, or
`python -m gulfseis --at "2026-09-26 10:25" --utc-offset 3`.

## Data sources

- **Raspberry Shake** citizen network (`AM`): RS1D/3D/4D seismometers (EHZ) and Raspberry Shake &
  Boom / Raspberry Boom infrasound (HDF).
- **EarthScope / IRIS** and **GEOFON**: permanent and temporary networks.
- **USGS / EMSC** catalogues, used to cross-check events.

The app searches for stations inside the outline and up to 150 km outside it (adjustable). It removes
each instrument's response, so seismic data are in m/s and infrasound in Pa, and resamples everything
to 50 Hz. Long periods are processed in 1-hour chunks with overlap, so slow air waves are not lost.

## How signals are classified

1. **Trigger**: a band-pass filter, then STA/LTA on every seismometer and every infrasound microphone.
   Very noisy stations keep only their strongest triggers.
2. **Coincidence**: a real event reaches different stations within (distance ÷ wave speed) of each
   other. Triggers that nothing else confirms are **noise**.
3. **Detection, depending on how many stations saw it**:
   - 4 or more stations: full location, including depth, with a 95 % uncertainty region.
   - 2–3 stations: the possible **source area** inside the outline, shown shaded on the map.
     Two-station detections must be strong and close together.
   - Infrasound: the air wave arriving at microphones is located at the speed of sound.
   - One Raspberry Shake & Boom: the delay between the ground wave and the air wave gives the distance,
     d = Δt / (1/c − 1/Vp). It cannot give the direction.
4. **Explosion or earthquake**: the evidence is combined with P = 1/(1+e^−z). It comes from the air
   wave, the P/S amplitude ratio, depth, first motion, a catalogue match and the time of day. Events
   seen by few stations are labelled "Possible".
5. **Size**: local magnitude M_L (IASPEI), energy, and a TNT-equivalent yield.

The **Formulas** tab shows every formula with the detected events' numbers.

## Why a known explosion might be missed

- No public station close enough, or those stations were offline. Check the **Stations & data** tab
  (✅/❌ and data coverage).
- The time zone or time window is wrong. Air waves arrive minutes after the ground wave.
- The site is very noisy. Use **High** sensitivity in *Check a known event* mode.

## Project layout

```
app.py                  Streamlit interface          run_app.py   launcher
gulfseis/
  region.py             the monitored polygon (point-in-polygon, distance, grid)
  data_sources.py       FDSN station search, download, response removal, catalogues
  monitor.py            chunked processing of long periods
  pipeline.py           detect → associate → locate → classify (network, small, air, single)
  detection.py          filters, STA/LTA, AIC picker, first motion
  location.py           grid-search and distant-event locators
  discrimination.py     explosion vs earthquake evidence
  physics.py            travel times, magnitude, energy, yield
static/topojson/        bundled map outlines (MIT, plotly/sane-topojson)
tests/                  pytest suite
```

`tests/sim.py` and `tests/fake_fdsn.py` produce simulated recordings. **Only the tests use them**, to
check the processing and the whole download path without internet. The app itself only ever uses
real data.

## Limitations

- Public station coverage around the Gulf is uneven. Small blasts are only seen by nearby stations.
- Locations from 2–3 stations are areas, and single-station detections give only a distance.
- An air-only detection can also be thunder, a sonic boom or another loud sound.
- Yield estimates are order-of-magnitude. Confirm important events with official agencies (KNSN, IRSC,
  NCM, USGS, EMSC).
