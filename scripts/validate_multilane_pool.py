"""Validate a multilane workzone scene pool by empirically spawning the convoy.

For each scene in --pool_dir, call the runtime multilane merge OD sampler with the
optimizer's params and record how many of the convoy actually spawns. Scenes that
seat >= --min_spawn cars (at --num_agents) are copied into <pool_dir>_validated and
listed in a manifest. This is the standalone counterpart to building the check into
the generator; it screens out scenes with too little upstream (spawn) or downstream
(goal) room around the box.
"""
from __future__ import annotations
import argparse, json, math
from collections import Counter
from pathlib import Path

from src.trfc import compute_multilane_capacity, sample_multilane_merge_start_goal_pairs


def _bake_capacity(scene: dict, *, spawn_spacing_m: float, bounds_size_m: float) -> int:
    """Stamp per-map capacity into the scene's workzone box (in place).

    Idempotent: overwrites `capacity` + `capacity_per_lane` on
    `workzone.keepout_boxes[0]` from the geometry each call. Returns the capacity
    (0 if the scene carries no merge contract). Single source of truth:
    src/trfc/lane_center_sampler.compute_multilane_capacity.
    """
    boxes = (scene.get("zones", {}) or {}).get("keepout_boxes", []) or []
    if not boxes:
        return 0
    cap = compute_multilane_capacity(
        scene, spawn_spacing_m=spawn_spacing_m, bounds_size_m=bounds_size_m
    )
    boxes[0]["capacity"] = int(cap["capacity"])
    boxes[0]["capacity_per_lane"] = {str(k): int(v) for k, v in cap["per_lane"].items()}
    return int(cap["capacity"])


def _spawn_count(scene: dict, *, num_agents: int, bounds: float, approach_gap: float,
                 spawn_spacing: float, goal_clearance: float, return_goal_clearance: float,
                 seed: int) -> int:
    try:
        s = sample_multilane_merge_start_goal_pairs(
            scene, num_agents=num_agents, bounds_size_m=bounds, approach_gap_m=approach_gap,
            spawn_spacing_m=spawn_spacing, goal_clearance_m=goal_clearance,
            return_goal_clearance_m=return_goal_clearance, merge_return_frac=1.0, seed=seed,
        )
        return len(s)
    except Exception:
        return 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool_dir", type=Path, default=Path("data/processed/workzone_variants_multilane_pool"))
    p.add_argument("--num_agents", type=int, default=16)
    p.add_argument("--min_spawn", type=int, default=14, help="Keep scenes spawning >= this many (of num_agents).")
    p.add_argument("--bounds_size_m", type=float, default=200.0)
    p.add_argument("--approach_gap_m", type=float, default=33.0)
    p.add_argument("--spawn_spacing_m", type=float, default=8.0)
    p.add_argument("--goal_clearance_m", type=float, default=15.0)
    p.add_argument("--return_goal_clearance_m", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument(
        "--stamp_existing", type=Path, default=None,
        help="Idempotently bake capacity into every scene_*.json in this ALREADY-validated "
             "dir (in place) and exit; skips re-running the spawn validation. Use to upgrade "
             "an existing pool to carry baked capacity.",
    )
    args = p.parse_args()

    # Idempotent one-off: stamp capacity into an existing validated pool in place.
    if args.stamp_existing is not None:
        stamp_dir = args.stamp_existing
        sfiles = sorted(f for f in stamp_dir.glob("scene_*.json"))
        if not sfiles:
            print(f"[ERROR] no scene_*.json in {stamp_dir}")
            return
        stamped = 0
        for f in sfiles:
            scene = json.loads(f.read_text())
            cap = _bake_capacity(scene, spawn_spacing_m=args.spawn_spacing_m,
                                 bounds_size_m=args.bounds_size_m)
            f.write_text(json.dumps(scene, indent=2))
            stamped += 1
            print(f"  {f.name}: capacity={cap}")
        print(f"stamped capacity into {stamped} scenes in {stamp_dir}")
        return

    files = sorted(f for f in args.pool_dir.glob("scene_*.json") if f.name != "variants_manifest.json")
    if not files:
        print(f"[ERROR] no scene_*.json in {args.pool_dir}")
        return

    out_dir = args.pool_dir.parent / (args.pool_dir.name + "_validated")
    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(num_agents=args.num_agents, bounds=args.bounds_size_m, approach_gap=args.approach_gap_m,
                  spawn_spacing=args.spawn_spacing_m, goal_clearance=args.goal_clearance_m,
                  return_goal_clearance=args.return_goal_clearance_m, seed=args.seed)

    hist: Counter = Counter()
    kept: list[dict] = []
    for f in files:
        try:
            scene = json.loads(f.read_text())
        except Exception:
            hist[-1] += 1
            continue
        n = _spawn_count(scene, **kwargs)
        hist[n] += 1
        if n >= args.min_spawn:
            # Bake the per-map capacity into the copied scene so the env/optimizer
            # read it directly (single source of truth).
            cap = _bake_capacity(scene, spawn_spacing_m=args.spawn_spacing_m,
                                 bounds_size_m=args.bounds_size_m)
            (out_dir / f.name).write_text(json.dumps(scene, indent=2))
            kept.append({"scene": f.name, "spawned": n, "capacity": cap})

    manifest = {
        "pool_dir": str(args.pool_dir),
        "validated_dir": str(out_dir),
        "num_agents": args.num_agents,
        "min_spawn": args.min_spawn,
        "params": {k: v for k, v in kwargs.items() if k not in ("num_agents", "bounds")},
        "total_scenes": len(files),
        "kept": len(kept),
        "spawn_histogram": {str(k): v for k, v in sorted(hist.items())},
        "scenes": kept,
    }
    (out_dir / "validated_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"scanned {len(files)} scenes at num_agents={args.num_agents}")
    print("spawn-count histogram (spawned -> #scenes):")
    for k in sorted(hist):
        bar = "#" * min(60, hist[k])
        print(f"  {k:>3}: {hist[k]:>4}  {bar}")
    full = hist.get(args.num_agents, 0)
    print(f"full convoy ({args.num_agents}/{args.num_agents}): {full}")
    print(f"kept (>= {args.min_spawn}): {len(kept)}/{len(files)} -> {out_dir}")
    print(f"manifest: {out_dir / 'validated_manifest.json'}")


if __name__ == "__main__":
    main()
