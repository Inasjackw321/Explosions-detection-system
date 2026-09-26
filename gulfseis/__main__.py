"""Command line:  python -m gulfseis --hours 3
                 python -m gulfseis --at "2026-09-26 10:25" --utc-offset 3

Downloads real data for the monitored Gulf area and prints the detections.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from . import data_sources, monitor
from .config import DATA_CENTERS, Settings
from .models import fmt_time


def main():
    ap = argparse.ArgumentParser(description="GulfSeis explosion & noise detector (real data)")
    ap.add_argument("--hours", type=float, default=3.0, help="analyse the last N hours (default 3)")
    ap.add_argument("--at", help="check a known time instead, 'YYYY-MM-DD HH:MM' (local time, see --utc-offset)")
    ap.add_argument("--utc-offset", type=float, default=0.0, help="hours ahead of UTC for --at (Kuwait: 3)")
    args = ap.parse_args()
    s = Settings()
    if args.at:
        t0 = datetime.strptime(args.at, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp()
        t0 -= args.utc_offset * 3600
        t1, t2 = t0 - 600, min(t0 + 1800, time.time() - 60)
        s.detection.trigger_on, s.detection.min_snr_small = 3.5, 4.0
    else:
        t2 = time.time() - 180
        t1 = t2 - args.hours * 3600
    f = data_sources.DataFetcher(list(DATA_CENTERS), s, log=print)
    if not f.discover(t1, t2):
        print("No stations found.")
        return
    cat = data_sources.fetch_catalog(s, t1, t2, log=print)
    r = monitor.run(f, t1, t2, s, cat, progress=lambda fr, t: print(f"\r{fr * 100:5.1f}% {t[:60]:60s}", end=""))
    print(f"\n{sum(x.ok for x in r.status.values())}/{len(r.status)} stations with data, "
          f"{len(r.in_region)} events in the area, {len(r.outside)} outside, {len(r.noise_picks)} noise triggers")
    for e in r.events:
        size = f"ML {e.ml:.1f}" if e.ml is not None else ""
        print(f"{e.icon} {e.id} {fmt_time(e.origin_time)} {e.lat:7.2f} {e.lon:7.2f} {size:7s} "
              f"{e.label:20s} {'in area' if e.in_region else 'OUTSIDE'} ({e.tier})")
        print(f"     {e.summary}")


if __name__ == "__main__":
    main()
