#!/usr/bin/env python3
"""Aggregate the multi-seed KNN-backbone rebuttal eval into per-condition stats.

Reads every run under logs/rsl_rl/knn_backbone_rebuttal_eval/ whose run name
matches ``rebut_<termination>_<od>_seed<N>`` and reports, per protocol cell,
the across-seed mean +/- population std of each metric.

Two levels of aggregation are reported because they answer different questions:

  agent-weighted : sum(success_count) / sum(spawned_count) over the 64 worlds of
                   a seed.  This is the headline "success rate" and matches how
                   scene_factory_policy_eval_summary.json computes it.
  world-mean     : unweighted mean of the per-world rates.  Less sensitive to a
                   few crowded scenes dominating the total.

Usage:  python scripts/aggregate_knn_backbone_eval.py [--csv out.csv]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

EVAL_ROOT = Path("logs/rsl_rl/knn_backbone_rebuttal_eval")
RUN_RE = re.compile(r"rebut_(terminating|invincible)_(spawns|randomod)_seed(\d+)$")

# --weather switches to the weather ablation (scripts/eval_knn_backbone_weather.sh).
WEATHER_ROOT = Path("logs/rsl_rl/knn_backbone_weather_eval")
WEATHER_RE = re.compile(r"wx_(dry|w04|w08)_(cond|blind)_seed(\d+)$")
# mu actually applied to the ground, verified from the env read-back print.
WEATHER_MU = {"dry": 1.000, "w04": 0.904, "w08": 0.804}

# (jsonl key, count key or None for an already-normalised rate, higher_is_better)
RATE_METRICS = [
    ("success", "success_count", True),
    ("collision", "collision_count", False),
    ("crash", "crash_count", False),
    ("lane_forbidden", "lane_forbidden_count", False),
    ("box_entry", "box_entry_count", False),
]
# per-world rates / means that have no underlying count to re-derive
MEAN_METRICS = [
    ("near_miss_rate", False),
    ("high_drac_rate", False),
    ("speed_violation_rate", False),
    ("obstacle_near_miss_rate", False),
    ("mean_max_drac", False),
    ("mean_min_ttc_s", True),
    ("mean_final_distance_to_goal", False),
]


def load_runs(root: Path = EVAL_ROOT, pattern: re.Pattern = RUN_RE) -> dict[tuple[str, str], list[dict]]:
    cells: dict[tuple[str, str], list[dict]] = {}
    if not root.is_dir():
        raise SystemExit(f"no eval root: {root}")
    for run_dir in sorted(root.iterdir()):
        m = pattern.search(run_dir.name)
        if not m:
            continue  # skips the probe run and anything hand-made
        # Only aggregate FINISHED runs.  A run still in flight has a partially
        # written worlds.jsonl, which would otherwise silently contribute a
        # short, biased sample to the cell.
        outcome_path = run_dir / "outcome.json"
        if not outcome_path.exists():
            print(f"[skip] still running (no outcome.json): {run_dir.name}")
            continue
        try:
            status = json.loads(outcome_path.read_text()).get("status")
        except json.JSONDecodeError:
            print(f"[skip] unreadable outcome.json: {run_dir.name}")
            continue
        if status != "success":
            print(f"[skip] status={status}: {run_dir.name}")
            continue
        worlds_path = run_dir / "scene_factory_policy_eval_worlds.jsonl"
        if not worlds_path.exists():
            print(f"[warn] no worlds.jsonl, skipping: {run_dir.name}")
            continue
        worlds = [json.loads(line) for line in worlds_path.read_text().splitlines() if line.strip()]
        if not worlds:
            print(f"[warn] empty worlds.jsonl, skipping: {run_dir.name}")
            continue
        term, od, seed = m.group(1), m.group(2), int(m.group(3))
        cells.setdefault((term, od), []).append(
            {"seed": seed, "worlds": worlds, "run": run_dir.name}
        )
    return cells


def _live(values) -> list[float]:
    """Drop the env's -1.0 "not applicable" sentinel and any None.

    Plain Waymo scenes carry no workzone boxes or cones, so the env reports
    box_entry / speed_violation / cone_near_miss as exactly -1.0.  Averaging
    those as if they were measurements produces negative "rates".
    """
    out = []
    for v in values:
        if v is None:
            continue
        f = float(v)
        if f <= -1.0:
            continue
        out.append(f)
    return out


def seed_stats(worlds: list[dict]) -> dict[str, float]:
    spawned = sum(_live([w.get("spawned_count") for w in worlds]))
    out: dict[str, float] = {"n_worlds": float(len(worlds)), "spawned": spawned}
    for name, count_key, _ in RATE_METRICS:
        counts = _live([w.get(count_key) for w in worlds])
        # If every world reported the sentinel, the metric does not apply here.
        out[f"{name}_agentw"] = (sum(counts) / spawned * 100.0) if (counts and spawned) else float("nan")
        per_world = _live([w.get(f"{name}_rate") for w in worlds])
        out[f"{name}_worldmean"] = statistics.fmean(per_world) * 100.0 if per_world else float("nan")
    for key, _ in MEAN_METRICS:
        vals = _live([w.get(key) for w in worlds])
        out[key] = statistics.fmean(vals) if vals else float("nan")
    return out


def fmt(vals: list[float], pct: bool) -> str:
    vals = [v for v in vals if v == v]  # drop NaN
    if not vals:
        return "n/a"
    mu = statistics.fmean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    unit = "%" if pct else ""
    return f"{mu:.1f}{unit} +/- {sd:.1f}" if pct else f"{mu:.3f} +/- {sd:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="")
    ap.add_argument("--weather", action="store_true",
                    help="aggregate the weather ablation instead of the 2x2 protocol cross")
    args = ap.parse_args()

    root, pattern = (WEATHER_ROOT, WEATHER_RE) if args.weather else (EVAL_ROOT, RUN_RE)
    cells = load_runs(root, pattern)
    if not cells:
        raise SystemExit(f"no matching runs under {root}")

    if args.weather:
        # Print the conditioned-vs-blind delta per weather cell up front: that
        # difference IS the weather-conditioning claim, and it has to be read
        # against the across-seed std, not in isolation.
        print("\n### weather-conditioning contrast (success %, agent-weighted) ###")
        print(f"{'weather':>9} {'mu':>7} {'conditioned':>18} {'blinded':>18} {'delta':>8}")
        for wx in ["dry", "w04", "w08"]:
            got = {}
            for obs in ["cond", "blind"]:
                runs = cells.get((wx, obs), [])
                if not runs:
                    continue
                vals = [seed_stats(r["worlds"])["success_agentw"] for r in runs]
                got[obs] = (statistics.fmean(vals),
                            statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                            len(vals))
            if "cond" in got and "blind" in got:
                c, b = got["cond"], got["blind"]
                print(f"{wx:>9} {WEATHER_MU[wx]:7.3f} "
                      f"{c[0]:8.1f} +/- {c[1]:4.1f} (n{c[2]}) "
                      f"{b[0]:8.1f} +/- {b[1]:4.1f} (n{b[2]}) "
                      f"{c[0]-b[0]:+8.1f}")
            else:
                have = ",".join(sorted(got)) or "none"
                print(f"{wx:>9} {WEATHER_MU[wx]:7.3f}   incomplete (have: {have})")
        print("  NOTE: a delta smaller than the +/- values is not a weather effect.")

    rows = []
    for key in sorted(cells):
        term, od = key
        runs = sorted(cells[key], key=lambda r: r["seed"])
        stats = [seed_stats(r["worlds"]) for r in runs]
        print(f"\n=== {term} / {od}  ({len(runs)} seeds: "
              f"{','.join(str(r['seed']) for r in runs)}) ===")
        print(f"  worlds/seed {stats[0]['n_worlds']:.0f}   "
              f"spawned/seed {fmt([s['spawned'] for s in stats], False)}")
        for name, _, hib in RATE_METRICS:
            aw = fmt([s[f"{name}_agentw"] for s in stats], True)
            wm = fmt([s[f"{name}_worldmean"] for s in stats], True)
            arrow = "^" if hib else "v"
            print(f"  {name:16s} {arrow}  agent-weighted {aw:20s} world-mean {wm}")
        for key2, hib in MEAN_METRICS:
            arrow = "^" if hib else "v"
            print(f"  {key2:16s} {arrow}  {fmt([s[key2] for s in stats], False)}")
        def _mu_sd(vals: list[float]) -> tuple[float, float]:
            vals = [v for v in vals if v == v]  # NaN-only columns (n/a metrics)
            if not vals:
                return float("nan"), float("nan")
            return statistics.fmean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)

        row = {"termination": term, "od": od, "n_seeds": len(runs)}
        for name, _, _ in RATE_METRICS:
            row[f"{name}_pct_mean"], row[f"{name}_pct_std"] = _mu_sd(
                [s[f"{name}_agentw"] for s in stats]
            )
        for key2, _ in MEAN_METRICS:
            row[f"{key2}_mean"], row[f"{key2}_std"] = _mu_sd([s[key2] for s in stats])
        rows.append(row)

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[csv] {args.csv}")


if __name__ == "__main__":
    main()
