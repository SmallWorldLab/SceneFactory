#!/usr/bin/env python3
"""Aggregate the cross-dynamics transfer evaluation into a Table-3-style report.

Reads logs/rsl_rl/transfer_eval/<stamp>_tx_<policy>2<backend>_seed<N>/ and emits

  1. a 2x2 matrix (policy x evaluation backend) of SR / collision / crash
  2. transfer deltas measured against each policy's OWN native cell, which is the
     correct reference -- a transfer drop must be read relative to what that
     policy achieves on its home backend, not relative to the other policy
  3. the goal-reaching vs safety split that reviewer bm7r asked for (Q5/B4):
     success rate and collision rate reported side by side, since high SR after
     transfer can coexist with unsafe behaviour
  4. --csv / --world-csv for mining

Rates are agent-weighted (sum counts / sum spawned). Fields equal to -1.0 are
"not applicable" sentinels and are dropped, never averaged.

Usage:
  PYTHONPATH=. python scripts/aggregate_transfer_eval.py \
      [--csv artifacts/transfer.csv] [--world-csv artifacts/transfer_worlds.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path("logs/rsl_rl/transfer_eval")
RUN_RE = re.compile(r"tx_(physx|bicycle)2(physx|bicycle)_seed(\d+)$")
NA = -1.0
BACKENDS = ["physx", "bicycle"]

COUNTS = [("SR %", "success_count"), ("CR %", "collision_count"),
          ("crash %", "crash_count"), ("laneviol %", "lane_forbidden_count")]
MEANS = [("maxDRAC", "mean_max_drac"), ("minTTC s", "mean_min_ttc_s"),
         ("finalDist m", "mean_final_distance_to_goal"), ("ep_steps", "episode_length_steps")]


def _checkpoint_of(d: Path) -> str:
    """The policy a run actually loaded, from its recorded config.

    Line-scanned rather than YAML-parsed: resolved_config.yaml carries the full
    RNG state and is expensive to load, and one key is all we need.
    """
    p = d / "params" / "resolved_config.yaml"
    if not p.exists():
        return ""
    for line in p.read_text(errors="replace").splitlines():
        s = line.strip()
        if s.startswith("checkpoint_path:"):
            return s.split(":", 1)[1].strip().strip('"\'')
    return ""


def run_stats(worlds: list[dict]) -> dict[str, float]:
    spawned = sum(w.get("spawned_count", 0) for w in worlds)
    if spawned <= 0:
        return {}
    out: dict[str, float] = {}
    for label, f in COUNTS:
        v = [w.get(f, NA) for w in worlds]
        if all(x == NA for x in v):
            continue
        out[label] = 100.0 * sum(x for x in v if x != NA) / spawned
    for label, f in MEANS:
        v = [w[f] for w in worlds if w.get(f) is not None and w.get(f) != NA and w[f] == w[f]]
        if v:
            out[label] = st.mean(v)
    return out


def ms(v: list[float]) -> str:
    if not v:
        return "-"
    sd = st.pstdev(v) if len(v) > 1 else 0.0
    return f"{st.mean(v):.1f}±{sd:.1f}" if abs(st.mean(v)) >= 10 else f"{st.mean(v):.2f}±{sd:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="")
    ap.add_argument("--world-csv", default="")
    a = ap.parse_args()

    if not ROOT.is_dir():
        raise SystemExit(f"no eval root: {ROOT} -- run bash run_transfer_eval.sh first")

    by_seed: dict[tuple[str, str, int], tuple[str, dict]] = {}
    ckpt_of: dict[tuple[str, str, int], str] = {}
    conflicts: list[str] = []
    world_rows: list[dict] = []
    pending: list[str] = []
    unmatched: list[str] = []
    for d in sorted(ROOT.iterdir()):
        m = RUN_RE.search(d.name)
        if not m:
            # A directory that looks like a transfer run but does not parse is
            # reported, never dropped. RUN_RE is anchored at the seed, so any
            # suffix (_INVINCTRAIN, V45, ...) lands here -- and a silently
            # skipped arm is indistinguishable from an arm that was never run.
            if d.is_dir() and "tx_" in d.name:
                unmatched.append(d.name)
            continue
        oc = d / "outcome.json"
        if not oc.exists():
            pending.append(d.name); continue
        try:
            if json.loads(oc.read_text()).get("status") != "success":
                pending.append(d.name + " (failed)"); continue
        except json.JSONDecodeError:
            pending.append(d.name + " (bad outcome)"); continue
        wp = d / "scene_factory_policy_eval_worlds.jsonl"
        if not wp.exists():
            pending.append(d.name + " (no worlds)"); continue
        worlds = [json.loads(l) for l in wp.read_text().splitlines() if l.strip()]
        s = run_stats(worlds)
        if not s:
            continue
        policy, backend, seed = m.group(1), m.group(2), int(m.group(3))
        key = (policy, backend, seed)
        ck = _checkpoint_of(d)
        # Two runs of the same cell against DIFFERENT policies are not a rerun,
        # they are different experiments, and taking the later name silently
        # picks one. Surface it instead.
        if key in ckpt_of and ck and ckpt_of[key] and ck != ckpt_of[key]:
            conflicts.append(f"{policy}2{backend} seed{seed}: {ckpt_of[key]}  vs  {ck}")
        if key not in by_seed or d.name > by_seed[key][0]:
            by_seed[key] = (d.name, s)
            ckpt_of[key] = ck
            world_rows = [r for r in world_rows if not (
                r["policy"] == policy and r["backend"] == backend and r["seed"] == seed)]
            for w in worlds:
                world_rows.append({"policy": policy, "backend": backend, "seed": seed, **w})

    if pending:
        print(f"[info] {len(pending)} run(s) not collected: {', '.join(pending[:4])}"
              f"{' ...' if len(pending) > 4 else ''}")
    if unmatched:
        print(f"[warn] {len(unmatched)} transfer-looking dir(s) did NOT parse as "
              f"tx_<policy>2<backend>_seed<N> and were EXCLUDED:")
        for n in unmatched:
            print(f"         {n}")
    if conflicts:
        print("[warn] same cell evaluated against different checkpoints; kept the "
              "later directory name:")
        for c in conflicts:
            print(f"         {c}")
    if not by_seed:
        raise SystemExit("no finished runs found")

    cells: dict[tuple[str, str], list[dict]] = {}
    seedmap: dict[tuple[str, str], dict[int, dict]] = {}
    for (p, b, seed), (_n, s) in by_seed.items():
        cells.setdefault((p, b), []).append(s)
        seedmap.setdefault((p, b), {})[seed] = s

    print(f"\n[coverage] {len(cells)}/4 cells, {sum(len(v) for v in cells.values())}/"
          f"{4 * max((len(v) for v in cells.values()), default=0)} seed-runs")

    labels = [l for l, _ in COUNTS] + [l for l, _ in MEANS]
    labels = [l for l in labels if any(l in r for v in cells.values() for r in v)]

    # ---- 1. 2x2 matrix ----------------------------------------------------
    print("\n=== CROSS-DYNAMICS MATRIX (dry ground) ===")
    hdr = f"{'policy':<10}{'eval backend':<14}{'kind':<10}" + "".join(f"{l:>13}" for l in labels)
    print(hdr); print("-" * len(hdr))
    for p in BACKENDS:
        for b in BACKENDS:
            runs = cells.get((p, b))
            if not runs:
                continue
            kind = "NATIVE" if p == b else "transfer"
            row = f"{p:<10}{b:<14}{kind:<10}"
            for l in labels:
                row += f"{ms([r[l] for r in runs if l in r]):>13}"
            print(row + f"   n={len(runs)}")

    # ---- 2. transfer drop vs each policy's OWN native cell -----------------
    print("\n=== TRANSFER EFFECT (cross - own native; paired by seed) ===")
    print(f"{'direction':<24}{'metric':<14}{'delta':>10}{'sd':>9}{'t':>8}")
    for p in BACKENDS:
        other = "bicycle" if p == "physx" else "physx"
        nat, cross = seedmap.get((p, p)), seedmap.get((p, other))
        if not nat or not cross:
            continue
        seeds = sorted(set(nat) & set(cross))
        if not seeds:
            continue
        for l in labels:
            dif = [cross[s][l] - nat[s][l] for s in seeds if l in cross[s] and l in nat[s]]
            if len(dif) < 2:
                continue
            md, sd = st.mean(dif), st.stdev(dif)
            t = md / (sd / len(dif) ** 0.5) if sd else float("nan")
            flag = " *" if abs(t) > 2.78 else ""
            print(f"{f'{p} -> {other}':<24}{l:<14}{md:>10.2f}{sd:>9.2f}{t:>8.2f}{flag}")
    print("  paired t, df=n-1; * marks |t|>2.78 (p<0.05 at n=5)")

    # ---- 3. goal-reaching vs safety --------------------------------------
    print("\n=== GOAL-REACHING vs SAFETY (bm7r Q5) ===")
    print("A policy can keep high SR after transfer while colliding far more.")
    print(f"{'cell':<26}{'SR %':>12}{'CR %':>12}{'crash %':>12}")
    for p in BACKENDS:
        for b in BACKENDS:
            runs = cells.get((p, b))
            if not runs:
                continue
            tag = f"{p}->{b}" + ("  (native)" if p == b else "")
            print(f"{tag:<26}{ms([r['SR %'] for r in runs if 'SR %' in r]):>12}"
                  f"{ms([r['CR %'] for r in runs if 'CR %' in r]):>12}"
                  f"{ms([r['crash %'] for r in runs if 'crash %' in r]):>12}")

    if a.csv:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["policy", "backend", "kind", "n_seeds"]
                       + [f"{l}_mean" for l in labels] + [f"{l}_sd" for l in labels])
            for p in BACKENDS:
                for b in BACKENDS:
                    runs = cells.get((p, b))
                    if not runs:
                        continue
                    mean_c, sd_c = [], []
                    for l in labels:
                        v = [r[l] for r in runs if l in r]
                        mean_c.append(f"{st.mean(v):.4f}" if v else "")
                        sd_c.append(f"{(st.pstdev(v) if len(v) > 1 else 0.0):.4f}" if v else "")
                    w.writerow([p, b, "native" if p == b else "transfer", len(runs)] + mean_c + sd_c)
        print(f"\n[write] {a.csv}")

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
        print(f"[write] {a.world_csv}  ({len(world_rows)} rows, one per world)")


if __name__ == "__main__":
    main()
