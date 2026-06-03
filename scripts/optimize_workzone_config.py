"""Workzone configuration optimizer using Cross-Entropy Method (CEM).

Outer optimization loop:
  1. Sample a population of workzone parameter vectors.
  2. Generate scene JSONs for each candidate.
  3. Run headless policy eval for each candidate (subprocess).
  4. Read per-world safety metrics from the eval summary JSON.
  5. Compute a safety score and select elites.
  6. Update Gaussian distribution → repeat.

Usage
-----
python scripts/optimize_workzone_config.py \\
    --checkpoint <path/to/model.pt> \\
    --config    configs/scene_factory/workzone_train.yaml \\
    --scene_dir data/workzone_scenes \\
    --output_dir artifacts/workzone_optimize \\
    --cem_iterations 20 \\
    --population_size 16 \\
    --elite_fraction 0.25
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------

@dataclass
class WorkzoneParamBounds:
    taper_length_m_lo: float = 20.0
    taper_length_m_hi: float = 120.0
    cone_spacing_m_lo: float = 3.0
    cone_spacing_m_hi: float = 20.0
    exit_taper_length_m_lo: float = 10.0
    exit_taper_length_m_hi: float = 80.0
    speed_limit_mps_lo: float = 5.6   # ~20 km/h
    speed_limit_mps_hi: float = 22.2  # ~80 km/h


PARAM_NAMES = [
    "taper_length_m",
    "cone_spacing_m",
    "exit_taper_length_m",
    "speed_limit_mps",
]


def _bounds_vectors(bounds: WorkzoneParamBounds) -> tuple[list[float], list[float]]:
    lo = [
        bounds.taper_length_m_lo,
        bounds.cone_spacing_m_lo,
        bounds.exit_taper_length_m_lo,
        bounds.speed_limit_mps_lo,
    ]
    hi = [
        bounds.taper_length_m_hi,
        bounds.cone_spacing_m_hi,
        bounds.exit_taper_length_m_hi,
        bounds.speed_limit_mps_hi,
    ]
    return lo, hi


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Safety score (lower = safer; optimizer minimises this)
# ---------------------------------------------------------------------------

def safety_score(
    summary: dict[str, Any],
    *,
    w_cone_near_miss: float = 1.0,
    w_speed_violation: float = 0.5,
    w_collision: float = 2.0,
    w_goal_reach: float = -0.5,
) -> float:
    """Compute a scalar safety cost for a single workzone parameter config.

    Lower score = safer / better configuration.
    """
    cone_nm = float(summary.get("cone_near_miss_rate", 0.0))
    speed_viol = float(summary.get("speed_violation_rate", 0.0))
    collision = float(summary.get("collision_rate", 0.0))
    success = float(summary.get("success_rate", 0.0))
    # Clamp negatives (absent / non-workzone worlds return -1.0)
    cone_nm = max(0.0, cone_nm)
    speed_viol = max(0.0, speed_viol)
    return (
        w_cone_near_miss * cone_nm
        + w_speed_violation * speed_viol
        + w_collision * collision
        + w_goal_reach * success
    )


# ---------------------------------------------------------------------------
# CEM helpers
# ---------------------------------------------------------------------------

@dataclass
class CEMState:
    mean: list[float]
    std: list[float]


def _sample_population(
    cem: CEMState,
    lo: list[float],
    hi: list[float],
    n: int,
    rng: random.Random,
) -> list[list[float]]:
    population: list[list[float]] = []
    for _ in range(n):
        vec = [
            _clamp(rng.gauss(m, s), l, h)
            for m, s, l, h in zip(cem.mean, cem.std, lo, hi)
        ]
        population.append(vec)
    return population


def _update_cem(
    cem: CEMState,
    elites: list[list[float]],
    lo: list[float],
    hi: list[float],
    min_std_fraction: float,
) -> None:
    """Update CEM mean/std in-place from elite vectors."""
    k = len(PARAM_NAMES)
    new_mean = [
        sum(e[i] for e in elites) / len(elites)
        for i in range(k)
    ]
    new_std = [
        max(
            (hi[i] - lo[i]) * min_std_fraction,
            math.sqrt(sum((e[i] - new_mean[i]) ** 2 for e in elites) / max(1, len(elites) - 1)),
        )
        for i in range(k)
    ]
    cem.mean[:] = new_mean
    cem.std[:] = new_std


# ---------------------------------------------------------------------------
# Scene generation + eval
# ---------------------------------------------------------------------------

def _generate_scenes(param_vec: list[float], scene_dir: Path, n_scenes: int) -> None:
    """Generate workzone scene JSONs from a parameter vector."""
    # Import here so the script can be imported without IsaacLab dependencies
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.workzone_scene_generator import WorkzoneConeConfig, write_workzone_scenes

    cfg = WorkzoneConeConfig(
        taper_length_m=param_vec[0],
        cone_spacing_m=param_vec[1],
        exit_taper_length_m=param_vec[2],
        speed_limit_mps=param_vec[3],
    )
    scene_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale scenes
    for f in scene_dir.glob("scene_workzone_*.json"):
        f.unlink()
    write_workzone_scenes(scene_dir, cone_cfg=cfg, n_scenes=n_scenes)


def _run_eval(
    *,
    checkpoint: str,
    config: str,
    scene_dir: Path,
    run_dir: Path,
    python_bin: str,
    extra_args: list[str],
    timeout_s: int,
) -> Path | None:
    """Launch headless eval subprocess; return path to summary JSON or None on failure."""
    summary_path = run_dir / "scene_factory_policy_eval_summary.json"
    cmd = [
        python_bin,
        "-u",
        "src/train_student_vehicle_goal_multiagent_rsl_rl.py",
        "--config", config,
        "--test_mode", "scene_factory_policy_eval",
        "--checkpoint_path", checkpoint,
        "--headless",
        "--no-use_fabric",
        "--invincible",
    ] + extra_args
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parent.parent)
    # Point the scene dir via env var (parsed in workzone_train.yaml io.scene_json_dir)
    env["WORKZONE_SCENE_DIR"] = str(scene_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "eval.log"
    print(f"  [Eval] Launching eval → {log_path} ...", flush=True)
    try:
        with open(log_path, "w") as log_fh:
            proc = subprocess.run(
                cmd,
                env=env,
                cwd=str(Path(__file__).parent.parent),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
            )
        if proc.returncode != 0:
            print(f"  [Eval] Non-zero exit {proc.returncode}. See {log_path}", flush=True)
            return None
    except subprocess.TimeoutExpired:
        print(f"  [Eval] Timed out after {timeout_s}s.", flush=True)
        return None
    except Exception as exc:
        print(f"  [Eval] Exception: {exc}", flush=True)
        return None

    # The eval writes summary to run_dir / "scene_factory_policy_eval_summary.json"
    if not summary_path.is_file():
        print(f"  [Eval] Summary not found at {summary_path}", flush=True)
        return None
    return summary_path


def _aggregate_summary(summary_path: Path) -> dict[str, float]:
    """Extract aggregate safety metrics from an eval summary JSON."""
    with open(summary_path) as fh:
        data = json.load(fh)

    # Aggregate per-world cone_near_miss_rate / speed_violation_rate
    worlds = data.get("per_world_summary_sorted_by_least_success", [])
    wz_worlds = [w for w in worlds if float(w.get("cone_near_miss_rate", -1.0)) >= 0.0]
    if wz_worlds:
        cone_nm = sum(float(w["cone_near_miss_rate"]) for w in wz_worlds) / len(wz_worlds)
        speed_viol = sum(float(w["speed_violation_rate"]) for w in wz_worlds) / len(wz_worlds)
    else:
        cone_nm = 0.0
        speed_viol = 0.0

    return {
        "cone_near_miss_rate": cone_nm,
        "speed_violation_rate": speed_viol,
        "collision_rate": float(data.get("collision_rate", 0.0)),
        "success_rate": float(data.get("success_rate", 0.0)),
        "crash_rate": float(data.get("crash_rate", 0.0)),
    }


# ---------------------------------------------------------------------------
# Main optimizer loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CEM optimizer for workzone configuration safety."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to trained policy checkpoint (.pt).")
    parser.add_argument("--config", default="configs/scene_factory/workzone_train.yaml",
                        help="YAML config for eval (workzone_train.yaml).")
    parser.add_argument("--scene_dir", default="data/workzone_scenes",
                        help="Directory where generated scene JSONs are written.")
    parser.add_argument("--output_dir", default="artifacts/workzone_optimize",
                        help="Output directory for optimization results.")
    parser.add_argument("--n_scenes", type=int, default=8,
                        help="Number of workzone scenes to generate per candidate.")
    parser.add_argument("--cem_iterations", type=int, default=20)
    parser.add_argument("--population_size", type=int, default=16)
    parser.add_argument("--elite_fraction", type=float, default=0.25)
    parser.add_argument("--initial_std_fraction", type=float, default=0.30)
    parser.add_argument("--min_std_fraction", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python_bin", default=sys.executable,
                        help="Python interpreter (default: current interpreter).")
    parser.add_argument("--eval_timeout_s", type=int, default=600,
                        help="Per-candidate eval timeout in seconds.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--w_cone_near_miss", type=float, default=1.0)
    parser.add_argument("--w_speed_violation", type=float, default=0.5)
    parser.add_argument("--w_collision", type=float, default=2.0)
    parser.add_argument("--w_goal_reach", type=float, default=-0.5)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    bounds = WorkzoneParamBounds()
    lo, hi = _bounds_vectors(bounds)

    # Initialise CEM at bounds midpoint
    init_mean = [(l + h) / 2.0 for l, h in zip(lo, hi)]
    init_std = [(h - l) * float(args.initial_std_fraction) for l, h in zip(lo, hi)]
    cem = CEMState(mean=init_mean[:], std=init_std[:])

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_dir = Path(args.scene_dir).expanduser().resolve()

    all_results: list[dict[str, Any]] = []
    best_score = math.inf
    best_params: dict[str, float] | None = None

    history_path = output_dir / "optimization_history.jsonl"
    summary_out_path = output_dir / "best_config.json"

    print(f"[WZ-OPT] Starting CEM optimization: {args.cem_iterations} iterations × {args.population_size} candidates.")
    print(f"[WZ-OPT] Output dir: {output_dir}", flush=True)

    for iteration in range(args.cem_iterations):
        print(f"\n[WZ-OPT] === Iteration {iteration + 1}/{args.cem_iterations} ===", flush=True)
        population = _sample_population(cem, lo, hi, args.population_size, rng)
        elite_count = max(1, int(math.ceil(args.population_size * args.elite_fraction)))

        iter_results: list[tuple[float, list[float], dict[str, float]]] = []

        for cand_idx, param_vec in enumerate(population):
            params_dict = dict(zip(PARAM_NAMES, param_vec))
            print(
                f"  [Cand {cand_idx + 1}/{args.population_size}] "
                + " ".join(f"{k}={v:.3f}" for k, v in params_dict.items()),
                flush=True,
            )

            # 1. Generate scenes
            try:
                _generate_scenes(param_vec, scene_dir, args.n_scenes)
            except Exception as exc:
                print(f"  [Gen] Failed to generate scenes: {exc}", flush=True)
                continue

            # 2. Run eval
            cand_run_dir = output_dir / f"iter{iteration:03d}_cand{cand_idx:03d}"
            summary_path = _run_eval(
                checkpoint=str(Path(args.checkpoint).expanduser().resolve()),
                config=args.config,
                scene_dir=scene_dir,
                run_dir=cand_run_dir,
                python_bin=args.python_bin,
                extra_args=["--device", args.device],
                timeout_s=args.eval_timeout_s,
            )
            if summary_path is None:
                print(f"  [Cand {cand_idx + 1}] Eval failed — skipping.", flush=True)
                continue

            # 3. Score
            metrics = _aggregate_summary(summary_path)
            score = safety_score(
                metrics,
                w_cone_near_miss=args.w_cone_near_miss,
                w_speed_violation=args.w_speed_violation,
                w_collision=args.w_collision,
                w_goal_reach=args.w_goal_reach,
            )
            print(
                f"  [Score] {score:.4f}  "
                + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
                flush=True,
            )
            iter_results.append((score, param_vec, metrics))

            record = {
                "iteration": iteration,
                "candidate": cand_idx,
                "params": params_dict,
                "metrics": metrics,
                "score": score,
            }
            all_results.append(record)
            with open(history_path, "a") as fh:
                fh.write(json.dumps(record) + "\n")

            if score < best_score:
                best_score = score
                best_params = params_dict.copy()
                best_params["_score"] = score
                best_params.update({f"metric_{k}": v for k, v in metrics.items()})
                summary_out_path.write_text(json.dumps(best_params, indent=2) + "\n")
                print(f"  [Best] New best score {best_score:.4f}: {best_params}", flush=True)

        if not iter_results:
            print("  [CEM] No successful candidates — keeping current distribution.", flush=True)
            continue

        # 4. Elite selection and distribution update
        iter_results.sort(key=lambda x: x[0])
        elites = [vec for _, vec, _ in iter_results[:elite_count]]
        _update_cem(cem, elites, lo, hi, args.min_std_fraction)
        print(
            f"  [CEM] Updated distribution from {len(elites)} elites. "
            f"mean={[f'{m:.3f}' for m in cem.mean]} "
            f"std={[f'{s:.3f}' for s in cem.std]}",
            flush=True,
        )

    print(f"\n[WZ-OPT] Optimization complete. Best score: {best_score:.4f}", flush=True)
    if best_params:
        print(f"[WZ-OPT] Best params: {best_params}", flush=True)
    print(f"[WZ-OPT] History: {history_path}", flush=True)
    print(f"[WZ-OPT] Best config: {summary_out_path}", flush=True)


if __name__ == "__main__":
    main()
