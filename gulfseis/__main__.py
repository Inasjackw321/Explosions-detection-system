"""Command-line mode:  python -m gulfseis [--demo | --real MINUTES]

Prints detected events as text - handy for scripts, cron jobs or servers.
"""

from __future__ import annotations

import argparse
import time

from . import data_sources, synthetic
from .config import DATA_CENTERS, Settings
from .models import fmt_time
from .pipeline import analyze
from .places import describe


def main():
    ap = argparse.ArgumentParser(description="GulfSeis explosion & earthquake detector")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--demo", action="store_true", help="simulated data (default)")
    g.add_argument("--real", type=int, metavar="MINUTES", help="analyse the last MINUTES of real data")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    s = Settings()
    if args.real:
        t_end = time.time() - 300
        _, wfs, cat = data_sources.load_real_data(list(DATA_CENTERS), s.region, t_end - 60 * args.real,
                                                  t_end, log=print)
    else:
        _, wfs, cat, _ = synthetic.generate(seed=args.seed)
    res = analyze(wfs, s, cat)
    print(f"{len(res.traces)} stations, {len(res.picks)} triggers, {len(res.events)} events")
    for e in res.events:
        size = f"ML {e.ml:.1f}" if e.ml is not None else ""
        where = describe(e.lat, e.lon) if e.kind == "local" else f"{e.distance_deg:.0f} deg away"
        print(f"{e.icon} {e.id} {fmt_time(e.origin_time)}  {e.lat:7.2f} {e.lon:7.2f}  "
              f"depth {e.depth:4.0f} km  {size:7s} {e.label:20s} {where}")
        print(f"     {e.summary}")


if __name__ == "__main__":
    main()
