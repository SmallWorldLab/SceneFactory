#!/usr/bin/env python3
"""Aggregate the Table 4 re-run: paper table + extended metrics + minable CSVs.

Reads logs/rsl_rl/table4_eval/<stamp>_t4_<policy>_<surface>_seed<N>/ and emits

  1. the paper's Table 4 layout (SR / CR / mean-max-DRAC), mean +/- sd over seeds
  2. an extended safety/behaviour table over every metric the eval logs
  3. --csv       cell-level summary, one row per (surface, policy, metric set)
  4. --world-csv LONG format, one row per (policy, surface, seed, world) with all
                 raw fields -- this is the one to mine

Seeds vary map sampling (a random 64 of the 199 held-out scenes), so the spread
across seeds is unseen-map variance, which is the uncertainty the AC asked for.

Rates are AGENT-WEIGHTED across worlds (sum counts / sum spawned), not means of
per-world rates -- a world with 1 spawned agent must not carry the same weight as
one with 16. Per-world means (DRAC, TTC, distance) are averaged per world since
they have no underlying count.

Fields carrying -1.0 are "not applicable" sentinels for this scenario type
(box entry, speed violation, cone near-miss are workzone-only). They are DROPPED,
never averaged -- averaging a -1 sentinel silently poisons the cell.

Usage:
  PYTHONPATH=. python scripts/aggregate_table4_eval.py \
      [--csv artifacts/table4.csv] [--world-csv artifacts/table4_worlds.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path("logs/rsl_rl/table4_eval")
RUN_RE = re.compile(r"t4_(blind|aware)_(dry|wet0p5mm|wet5mm|wet12mm)_seed(\d+)$")
NA = -1.0  # sentinel used by the env for metrics that do not apply here

SURFACES = [
    ("dry", "Dry (AC 0.0mm, mu=1.174)"),
    ("wet0p5mm", "Moderate wet (AC 0.5mm, mu=0.944)"),
    ("wet5mm", "Heavy wet (AC 5mm, mu=0.506)"),
    ("wet12mm", "Extreme wet (AC 12mm, mu=0.408)"),
]
POLICIES = [("aware", "wet-exposed (v7-like)"), ("blind", "dry-only (v8-like)")]

# agent-weighted rates: (label, count field, higher_is_better)
COUNT_METRICS = [
    ("SR %", "success_count", True),
    ("CR %", "collision_count", False),
    ("crash %", "crash_count", False),
    ("  tilt %", "crash_bad_tilt_count", False),
    ("  toofar %", "crash_too_far_count", False),
    ("  toolow %", "crash_too_low_count", False),
    ("laneviol %", "lane_forbidden_count", False),
]
# per-world means: (label, field, higher_is_better)
MEAN_METRICS = [
    ("maxDRAC", "mean_max_drac", False),
    ("minTTC s", "mean_min_ttc_s", True),
    ("finalDist m", "mean_final_distance_to_goal", False),
    ("nearmiss", "near_miss_rate", False),
    ("hiDRAC", "high_drac_rate", False),
    ("ep_steps", "episode_length_steps", False),
]


def run_stats(worlds: list[dict]) -> dict[str, float]:
    spawned = sum(w.get("spawned_count", 0) for w in worlds)
    if spawned <= 0:
        return {}
    out: dict[str, float] = {"spawned": spawned, "worlds": len(worlds)}
    for label, field, _ in COUNT_METRICS:
        vals = [w.get(field, NA) for w in worlds]
        if all(v == NA for v in vals):
            continue
        out[label] = 100.0 * sum(v for v in vals if v != NA) / spawned
    for label, field, _ in MEAN_METRICS:
        vals = [w[field] for w in worlds
                if w.get(field) is not None and w.get(field) != NA and w[field] == w[field]]
        if vals:
            out[label] = st.mean(vals)
    return out


def fmt(vals: list[float]) -> str:
    if not vals:
        return "-"
    sd = st.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{st.mean(vals):.1f}±{sd:.1f}" if abs(st.mean(vals)) >= 10 else f"{st.mean(vals):.2f}±{sd:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="")
    ap.add_argument("--world-csv", default="")
    a = ap.parse_args()

    if not ROOT.is_dir():
        raise SystemExit(f"no eval root: {ROOT} -- run bash run_paper_table4_eval.sh first")

    by_seed: dict[tuple[str, str, int], tuple[str, dict]] = {}
    world_rows: list[dict] = []
    pending: list[str] = []
    for d in sorted(ROOT.iterdir()):
        m = RUN_RE.search(d.name)
        if not m:
            continue  # quarantined (*_INVALID_*, *_PREFIX_*) or unrelated
        oc = d / "outcome.json"
        if not oc.exists():
            pending.append(d.name); continue
        try:
            if json.loads(oc.read_text()).get("status") != "success":
                pending.append(d.name + " (failed)"); continue
        except json.JSONDecodeError:
            pending.append(d.name + " (bad outcome.json)"); continue
        wp = d / "scene_factory_policy_eval_worlds.jsonl"
        if not wp.exists():
            pending.append(d.name + " (no worlds.jsonl)"); continue
        worlds = [json.loads(l) for l in wp.read_text().splitlines() if l.strip()]
        s = run_stats(worlds)
        if not s:
            continue
        policy, surface, seed = m.group(1), m.group(2), int(m.group(3))
        # Re-running a cell leaves both dirs on disk; keep the newest per seed.
        key = (surface, policy, seed)
        if key not in by_seed or d.name > by_seed[key][0]:
            by_seed[key] = (d.name, s)
            world_rows = [r for r in world_rows
                          if not (r["policy"] == policy and r["surface"] == surface and r["seed"] == seed)]
            for w in worlds:
                world_rows.append({"policy": policy, "surface": surface, "seed": seed, **w})

    if pending:
        print(f"[info] {len(pending)} run(s) not yet collected: {', '.join(pending[:4])}"
              f"{' ...' if len(pending) > 4 else ''}")
    if not by_seed:
        raise SystemExit("no finished runs found")

    cells: dict[tuple[str, str], list[dict]] = {}
    for (surface, policy, _s), (_n, stats) in by_seed.items():
        cells.setdefault((surface, policy), []).append(stats)

    have = {(s, p) for (s, p) in cells}
    print(f"\n[coverage] {len(have)}/8 (surface x policy) cells collected, "
          f"{sum(len(v) for v in cells.values())}/40 seed-runs")

    # ---- 1. paper Table 4 layout -----------------------------------------
    print("\n=== TABLE 4 (paper layout) ===")
    hdr = f"{'Surface':<34} {'Policy':<24} {'SR %':>12} {'CR %':>12} {'maxDRAC':>12}"
    print(hdr); print("-" * len(hdr))
    for skey, slabel in SURFACES:
        first = True
        for pkey, plabel in POLICIES:
            runs = cells.get((skey, pkey))
            if not runs:
                continue
            g = lambda k: [r[k] for r in runs if k in r]
            print(f"{(slabel if first else ''):<34} {plabel:<24} "
                  f"{fmt(g('SR %')):>12} {fmt(g('CR %')):>12} {fmt(g('maxDRAC')):>12}   n={len(runs)}")
            first = False

    # ---- 2. extended metrics ---------------------------------------------
    print("\n=== EXTENDED METRICS (mean±sd over seeds) ===")
    labels = [l for l, _, _ in COUNT_METRICS] + [l for l, _, _ in MEAN_METRICS]
    labels = [l for l in labels if any(l in r for runs in cells.values() for r in runs)]
    print(f"{'cell':<40}" + "".join(f"{l:>13}" for l in labels))
    print("-" * (40 + 13 * len(labels)))
    for skey, slabel in SURFACES:
        for pkey, plabel in POLICIES:
            runs = cells.get((skey, pkey))
            if not runs:
                continue
            name = f"{skey}/{pkey}"
            row = f"{name:<40}"
            for l in labels:
                v = [r[l] for r in runs if l in r]
                row += f"{(fmt(v) if v else '-'):>13}"
            print(row)

    # ---- 3. aware - blind deltas -----------------------------------------
    print("\n=== DELTA: aware - blind  (the conditioning effect) ===")
    print(f"{'Surface':<34}" + "".join(f"{l:>13}" for l in labels))
    for skey, slabel in SURFACES:
        aw, bl = cells.get((skey, "aware")), cells.get((skey, "blind"))
        if not aw or not bl:
            continue
        row = f"{slabel:<34}"
        for l in labels:
            va = [r[l] for r in aw if l in r]; vb = [r[l] for r in bl if l in r]
            if not va or not vb:
                row += f"{'-':>13}"; continue
            d = st.mean(va) - st.mean(vb)
            noise = max(st.pstdev(va) if len(va) > 1 else 0.0, st.pstdev(vb) if len(vb) > 1 else 0.0)
            row += f"{f'{d:+.2f}' + ('' if abs(d) > 2 * noise else '~'):>13}"
        print(row)
    print("  '~' = difference is inside 2x the across-seed sd, i.e. NOT resolvable")

    # ---- 4. CSVs ----------------------------------------------------------
    if a.csv:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["surface", "policy", "n_seeds"]
                       + [f"{l}_mean" for l in labels] + [f"{l}_sd" for l in labels])
            for skey, _ in SURFACES:
                for pkey, _ in POLICIES:
                    runs = cells.get((skey, pkey))
                    if not runs:
                        continue
                    means, sds = [], []
                    for l in labels:
                        v = [r[l] for r in runs if l in r]
                        means.append(f"{st.mean(v):.4f}" if v else "")
                        sds.append(f"{(st.pstdev(v) if len(v) > 1 else 0.0):.4f}" if v else "")
                    w.writerow([skey, pkey, len(runs)] + means + sds)
        print(f"\n[write] {a.csv}  (cell-level summary)")

    if a.world_csv:
        Path(a.world_csv).parent.mkdir(parents=True, exist_ok=True)
        keys, seen = [], set()
        for r in world_rows:
            for k in r:
                if k not in seen:
                    seen.add(k); keys.append(k)
        with open(a.world_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader(); w.writerows(world_rows)
        print(f"[write] {a.world_csv}  ({len(world_rows)} rows, one per world -- "
              f"per-scene mining, joinable on scene_json_name)")


if __name__ == "__main__":
    main()
