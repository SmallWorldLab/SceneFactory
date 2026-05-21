"""workzone_scene_generator.py
Generates synthetic straight-road scene JSONs compatible with ChocolateBarConstructor,
each with a fixed workzone and a randomized taper-zone cone layout.

Scene JSON schema (matches Waymo scene format used by chocolate_waymo_builder):
  {
    "meta": {...},
    "road": {
      "polylines": [
        {"type": <int>, "id": <int>, "n": <int>, "xyz": [[x,y,z], ...]},
        ...
      ]
    },
    "agents": []
  }

Polyline types used:
  2  = lane center (surface street)
  15 = road edge boundary
  6  = broken lane divider

Workzone cone layout per world:
  - Fixed closed workzone box (e.g. 20m x 6m) at center of road
  - Randomized taper: approach taper length, cone spacing, lateral offset jitter
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Types matching Waymo road polyline format
# ---------------------------------------------------------------------------
LANE_CENTER_TYPE   = 2   # surface-street lane center
ROAD_EDGE_TYPE     = 15  # road edge / boundary
LANE_DIVIDER_TYPE  = 6   # broken white lane divider


@dataclass
class WorkzoneConeConfig:
    """Randomizable parameters for the taper zone."""
    taper_length_m: float = 30.0        # length of approach taper before workzone
    cone_spacing_m: float = 5.0         # longitudinal spacing between cones
    cone_lateral_jitter_m: float = 0.0  # ±lateral random offset per cone
    exit_taper_length_m: float = 20.0   # taper on exit side
    workzone_length_m: float = 30.0     # length of closed workzone box
    lane_width_m: float = 3.7           # single lane width


@dataclass
class TaperCone:
    x: float
    y: float
    z: float = 0.05


def _straight_polyline(
    x_start: float,
    x_end: float,
    y: float,
    z: float = 0.0,
    n_points: int = 20,
    pid: int = 0,
    ptype: int = 2,
) -> dict[str, Any]:
    """Straight horizontal polyline along X axis."""
    xs = [x_start + (x_end - x_start) * i / max(1, n_points - 1) for i in range(n_points)]
    xyz = [[x, y, z] for x in xs]
    return {"type": ptype, "id": pid, "n": len(xyz), "xyz": xyz}


def generate_straight_road_scene(
    road_length_m: float = 120.0,
    road_half_width_m: float = 7.4,   # two lanes total
    lane_width_m: float = 3.7,
    z: float = 0.0,
) -> dict[str, Any]:
    """
    Generate the static road geometry: two lane centers + two road edges + center divider.
    Road runs along the X axis, centered at origin.
    """
    half = road_length_m / 2.0
    n = 30

    polylines = []
    pid = 0

    # Road edges (type 15)
    polylines.append(_straight_polyline(-half, half, -road_half_width_m, z, n, pid, ROAD_EDGE_TYPE))
    pid += 1
    polylines.append(_straight_polyline(-half, half,  road_half_width_m, z, n, pid, ROAD_EDGE_TYPE))
    pid += 1

    # Lane centers — right lane (positive Y) and left lane (negative Y)
    right_lane_y = lane_width_m / 2.0
    left_lane_y  = -lane_width_m / 2.0
    polylines.append(_straight_polyline(-half, half, right_lane_y, z, n, pid, LANE_CENTER_TYPE))
    pid += 1
    polylines.append(_straight_polyline(-half, half, left_lane_y, z, n, pid, LANE_CENTER_TYPE))
    pid += 1

    # Center broken-line divider
    polylines.append(_straight_polyline(-half, half, 0.0, z, n, pid, LANE_DIVIDER_TYPE))
    pid += 1

    return {
        "meta": {
            "source": "workzone_scene_generator",
            "road_length_m": road_length_m,
            "road_half_width_m": road_half_width_m,
        },
        "road": {"polylines": polylines},
        "agents": [],
    }


def generate_taper_cones(
    cfg: WorkzoneConeConfig,
    rng: random.Random,
    workzone_center_x: float = 0.0,
    road_half_width_m: float = 7.4,
) -> list[TaperCone]:
    """
    Generate approach + exit taper cones for a right-lane workzone closure.

    Layout (looking along +X / direction of travel):
      Approach taper: cones sweep from road edge inward to workzone boundary
      Workzone closed lane: cones along right boundary of left lane
      Exit taper: cones sweep back out to road edge
    """
    cones: list[TaperCone] = []

    wz_x_start = workzone_center_x - cfg.workzone_length_m / 2.0
    wz_x_end   = workzone_center_x + cfg.workzone_length_m / 2.0

    # Right lane outer edge Y (edge of road)
    outer_y = road_half_width_m - 0.3  # just inside road edge
    # Inner closure line Y (right edge of left lane = center + lane_width/2 * side)
    inner_y = cfg.lane_width_m / 2.0 + 0.3

    # --- Approach taper ---
    taper_start_x = wz_x_start - cfg.taper_length_m
    n_taper = max(2, int(cfg.taper_length_m / cfg.cone_spacing_m))
    for i in range(n_taper + 1):
        t = i / max(1, n_taper)
        x = taper_start_x + t * cfg.taper_length_m
        y = outer_y - t * (outer_y - inner_y)   # sweep inward
        jitter = rng.uniform(-cfg.cone_lateral_jitter_m, cfg.cone_lateral_jitter_m)
        cones.append(TaperCone(x=x, y=y + jitter))

    # --- Workzone side line ---
    n_side = max(1, int(cfg.workzone_length_m / cfg.cone_spacing_m))
    for i in range(1, n_side):
        x = wz_x_start + i * (cfg.workzone_length_m / n_side)
        jitter = rng.uniform(-cfg.cone_lateral_jitter_m, cfg.cone_lateral_jitter_m)
        cones.append(TaperCone(x=x, y=inner_y + jitter))

    # --- Exit taper ---
    n_exit = max(2, int(cfg.exit_taper_length_m / cfg.cone_spacing_m))
    for i in range(1, n_exit + 1):
        t = i / max(1, n_exit)
        x = wz_x_end + t * cfg.exit_taper_length_m
        y = inner_y + t * (outer_y - inner_y)   # sweep back out
        jitter = rng.uniform(-cfg.cone_lateral_jitter_m, cfg.cone_lateral_jitter_m)
        cones.append(TaperCone(x=x, y=y + jitter))

    return cones


def randomize_workzone_config(rng: random.Random, lane_width_m: float = 3.7) -> WorkzoneConeConfig:
    """Sample a randomized taper zone design."""
    return WorkzoneConeConfig(
        taper_length_m=rng.uniform(20.0, 50.0),
        cone_spacing_m=rng.uniform(3.0, 8.0),
        cone_lateral_jitter_m=rng.uniform(0.0, 0.25),
        exit_taper_length_m=rng.uniform(15.0, 30.0),
        workzone_length_m=rng.uniform(20.0, 40.0),
        lane_width_m=lane_width_m,
    )


def generate_workzone_scene(
    world_index: int,
    seed: int,
    road_length_m: float = 200.0,
    road_half_width_m: float = 7.4,
    lane_width_m: float = 3.7,
) -> dict[str, Any]:
    """Generate a complete scene JSON for one world: road + workzone cone positions."""
    rng = random.Random(seed + world_index * 31337)
    scene = generate_straight_road_scene(road_length_m, road_half_width_m, lane_width_m)

    cone_cfg = randomize_workzone_config(rng, lane_width_m)
    cones = generate_taper_cones(cone_cfg, rng, workzone_center_x=0.0,
                                  road_half_width_m=road_half_width_m)

    scene["workzone"] = {
        "cone_cfg": {
            "taper_length_m": cone_cfg.taper_length_m,
            "cone_spacing_m": cone_cfg.cone_spacing_m,
            "cone_lateral_jitter_m": cone_cfg.cone_lateral_jitter_m,
            "exit_taper_length_m": cone_cfg.exit_taper_length_m,
            "workzone_length_m": cone_cfg.workzone_length_m,
            "lane_width_m": cone_cfg.lane_width_m,
        },
        "cones": [{"x": c.x, "y": c.y, "z": c.z} for c in cones],
    }

    scene["meta"]["world_index"] = world_index
    scene["meta"]["seed"] = seed
    return scene


def write_workzone_scenes(
    output_dir: str | Path,
    num_worlds: int = 6,
    seed: int = 42,
    road_length_m: float = 200.0,
) -> list[Path]:
    """Write N synthetic workzone scene JSON files and return their paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(num_worlds):
        scene = generate_workzone_scene(i, seed=seed, road_length_m=road_length_m)
        p = out / f"workzone_scene_{i:03d}.json"
        p.write_text(json.dumps(scene, indent=2))
        paths.append(p)
        n_cones = len(scene["workzone"]["cones"])
        cfg = scene["workzone"]["cone_cfg"]
        print(f"  world {i:02d}: taper={cfg['taper_length_m']:.1f}m  "
              f"spacing={cfg['cone_spacing_m']:.1f}m  "
              f"wz_len={cfg['workzone_length_m']:.1f}m  "
              f"cones={n_cones}")
    return paths


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="data/processed/workzone_scenes_json")
    p.add_argument("--num_worlds", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--road_length_m", type=float, default=200.0)
    args = p.parse_args()
    print(f"Generating {args.num_worlds} workzone scenes → {args.output_dir}")
    write_workzone_scenes(args.output_dir, args.num_worlds, args.seed, args.road_length_m)
    print("Done.")
