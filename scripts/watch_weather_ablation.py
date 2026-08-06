#!/usr/bin/env python3
"""Compare the rebuttal ablation arms from the terminal (no browser needed).

Usage:  PYTHONPATH=. python scripts/watch_weather_ablation.py [--tail N]
"""
from __future__ import annotations
import argparse, glob, os, statistics as st
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TB = "artifacts/weather_ablation_rebuttal/tb"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail", type=int, default=100, help="average over the last N iters")
    ap.add_argument("--tag", default="Metrics/success_rate")
    a = ap.parse_args()

    runs = sorted(d for d in glob.glob(f"{TB}/*") if os.path.isdir(d))
    if not runs:
        raise SystemExit(f"no runs under {TB} -- run scripts/link_weather_ablation_tb.sh")

    print(f"{'run':<16} {'iters':>7} {'last':>8} {f'mean[-{a.tail}]':>14} {'sd':>7}")
    print("-" * 56)
    for d in runs:
        ev = sorted(glob.glob(f"{d}/events.out.tfevents*"))
        if not ev:
            print(f"{os.path.basename(d):<16} {'(starting)':>7}")
            continue
        ea = EventAccumulator(ev[-1], size_guidance={"scalars": 0})
        ea.Reload()
        if a.tag not in ea.Tags()["scalars"]:
            print(f"{os.path.basename(d):<16} {'(no data)':>7}")
            continue
        s = ea.Scalars(a.tag)
        v = [x.value for x in s][-a.tail:]
        print(f"{os.path.basename(d):<16} {s[-1].step:>7} {s[-1].value:>8.4f} "
              f"{st.mean(v):>14.4f} {st.pstdev(v):>7.4f}")
    print(f"\ntag={a.tag}   target=1500 iters")


if __name__ == "__main__":
    main()
