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
    "agents": {"items": [{"agent_id": 10000, "start": {...}, "end": {...}}, ...]}
  }

Polyline types used:
  2  = lane center (surface street)
  15 = road edge boundary
  6  = broken lane divider

Workzone cone layout per world:
  - Fixed closed workzone box (same position/length across all worlds — represents
    the maintenance area we cannot control)
  - Randomized taper DESIGN: approach taper length, cone spacing, lateral offset jitter
    (these are the parameters being optimized)
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
    """Randomizable taper DESIGN parameters (what we are optimizing).
    The workzone box itself (length/position) is fixed and passed separately.
    """
    taper_length_m: float = 30.0        # length of approach taper before workzone
    cone_spacing_m: float = 5.0         # longitudinal spacing between cones
    cone_lateral_jitter_m: float = 0.0  # ±lateral random offset per cone
    exit_taper_length_m: float = 20.0   # taper on exit side
    lane_width_m: float = 3.7           # single lane width
    speed_limit_mps: float | None = None  # posted workzone speed limit (None = no limit)
    # NOTE: workzone_length_m is intentionally NOT here — it is fixed across worlds


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
        "agents": {"items": []},
    }


def generate_taper_cones(
    cfg: WorkzoneConeConfig,
    rng: random.Random,
    workzone_center_x: float = 0.0,
    road_half_width_m: float = 7.4,
    workzone_length_m: float = 30.0,  # fixed — same across all worlds
) -> list[TaperCone]:
    """
    Generate approach + exit taper cones for a right-lane workzone closure.

    Layout (looking along +X / direction of travel):
      Approach taper: cones sweep from road edge inward to workzone boundary
      Workzone closed lane: cones along right boundary of left lane
      Exit taper: cones sweep back out to road edge
    """
    cones: list[TaperCone] = []

    wz_x_start = workzone_center_x - workzone_length_m / 2.0
    wz_x_end   = workzone_center_x + workzone_length_m / 2.0

    # Close the RIGHT INNER lane (y: 0 → lane_width_m) — the lane agents actually drive in.
    # Agents spawn at y ≈ +lane_width/2 (right lane center) and must merge left to y ≈ -lane_width/2.
    lane_width_m = road_half_width_m / 2.0
    outer_y = lane_width_m - 0.3   # just inside right lane outer boundary
    inner_y = 0.3                   # just right of center divider — forces full right-lane closure

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
    # Use ceil so actual spacing <= cone_spacing_m (no under-counting).
    # i=0 places a cone at wz_x_start, i=n_side places one at wz_x_end,
    # guaranteeing the full workzone boundary is covered without gaps.
    n_side = max(1, math.ceil(workzone_length_m / cfg.cone_spacing_m))
    for i in range(0, n_side + 1):
        x = wz_x_start + i * (workzone_length_m / n_side)
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
    """Sample a randomized taper zone DESIGN (workzone box length is fixed separately)."""
    return WorkzoneConeConfig(
        taper_length_m=rng.uniform(15.0, 80.0),
        cone_spacing_m=rng.uniform(2.0, 10.0),
        cone_lateral_jitter_m=rng.uniform(0.0, 0.40),
        exit_taper_length_m=rng.uniform(10.0, 40.0),
        lane_width_m=rng.uniform(3.5, 4.0),
    )


def generate_agent_spawns(
    cone_cfg: WorkzoneConeConfig,
    road_length_m: float = 200.0,
    road_half_width_m: float = 7.4,
    workzone_center_x: float = 0.0,
    workzone_length_m: float = 30.0,
    num_agents: int = 4,
) -> list[dict[str, Any]]:
    """Generate synthetic start/goal pairs for vehicles approaching the workzone.

    Agents start before the approach taper and have goals after the exit taper,
    spread across both lanes. All positions are in world coordinates (not local).
    """
    half_road = road_length_m / 2.0
    right_lane_y = cone_cfg.lane_width_m / 2.0
    left_lane_y  = -cone_cfg.lane_width_m / 2.0
    lane_ys = [right_lane_y, left_lane_y]

    wz_x_start = workzone_center_x - workzone_length_m / 2.0
    wz_x_end   = workzone_center_x + workzone_length_m / 2.0
    taper_start_x = wz_x_start - cone_cfg.taper_length_m
    exit_end_x    = wz_x_end   + cone_cfg.exit_taper_length_m

    # Keep spawns comfortably before the taper and goals after the exit taper,
    # clamped to within the road bounds.
    spawn_x_base = max(-half_road + 5.0, taper_start_x - 15.0)
    goal_x_base  = min( half_road - 5.0, exit_end_x   + 15.0)

    items = []
    for i in range(num_agents):
        lane_y = lane_ys[i % len(lane_ys)]
        # Stagger starts by 5m per pair so agents don't spawn on top of each other
        offset = (i // len(lane_ys)) * 5.0
        sx = max(-half_road + 5.0, spawn_x_base - offset)
        gx = min( half_road - 5.0, goal_x_base  + offset)
        items.append({
            "track_idx": i,
            "is_sdc": False,
            "agent_type": 1,
            "agent_id": 10000 + i,
            "start": {"x": float(sx), "y": float(lane_y), "z": 0.0, "yaw": 0.0},
            "end":   {"x": float(gx), "y": float(lane_y), "z": 0.0, "yaw": 0.0},
        })
    return items


def generate_workzone_scene(
    world_index: int,
    seed: int,
    road_length_m: float = 200.0,
    road_half_width_m: float = 7.4,
    lane_width_m: float = 3.7,
    workzone_length_m: float | None = None,  # None = randomize per scene
) -> dict[str, Any]:
    """Generate a complete scene JSON for one world: road + workzone cone positions.

    When workzone_length_m is None (default for training), the workzone length is
    randomised per scene (20–80 m) to increase scene diversity. Pass an explicit
    value to fix it (used by the CEM optimizer so all worlds share the same layout).
    """
    rng = random.Random(seed + world_index * 31337)

    # Randomise road length and workzone length for training diversity.
    actual_road_length = rng.uniform(160.0, 260.0) if road_length_m == 200.0 else road_length_m
    actual_wz_length = rng.uniform(20.0, 80.0) if workzone_length_m is None else workzone_length_m

    scene = generate_straight_road_scene(actual_road_length, road_half_width_m, lane_width_m)

    cone_cfg = randomize_workzone_config(rng, lane_width_m)
    cones = generate_taper_cones(cone_cfg, rng, workzone_center_x=0.0,
                                  road_half_width_m=road_half_width_m,
                                  workzone_length_m=actual_wz_length)

    scene["workzone"] = {
        # Workzone geometry (fixed per scene, randomised across scenes)
        "workzone_length_m": actual_wz_length,
        "workzone_center_x": 0.0,
        # Randomized taper design (varies per world — this is what we optimize)
        "cone_cfg": {
            "taper_length_m": cone_cfg.taper_length_m,
            "cone_spacing_m": cone_cfg.cone_spacing_m,
            "cone_lateral_jitter_m": cone_cfg.cone_lateral_jitter_m,
            "exit_taper_length_m": cone_cfg.exit_taper_length_m,
            "lane_width_m": cone_cfg.lane_width_m,
            "speed_limit_mps": cone_cfg.speed_limit_mps,
        },
        "cones": [{"x": c.x, "y": c.y, "z": c.z} for c in cones],
        "speed_limit_mps": cone_cfg.speed_limit_mps,
    }

    scene["agents"]["items"] = generate_agent_spawns(
        cone_cfg,
        road_length_m=actual_road_length,
        road_half_width_m=road_half_width_m,
        workzone_center_x=0.0,
        workzone_length_m=actual_wz_length,
        num_agents=4,
    )

    scene["meta"]["world_index"] = world_index
    scene["meta"]["seed"] = seed
    scene["meta"]["road_length_m"] = actual_road_length
    scene["meta"]["workzone_length_m"] = actual_wz_length
    return scene


def write_workzone_scenes(
    output_dir: str | Path,
    num_worlds: int = 6,
    seed: int = 42,
    road_length_m: float = 200.0,
    workzone_length_m: float | None = None,  # None = randomize per scene
    cone_cfg: "WorkzoneConeConfig | None" = None,
) -> list[Path]:
    """Write N synthetic workzone scene JSON files and return their paths.

    Args:
        workzone_length_m: Fixed workzone length for all scenes. None (default)
            randomises per scene (20–80 m) for training diversity.
        cone_cfg: When provided, use this fixed config for all worlds instead of
                  randomising per world.  Used by the CEM optimizer so each
                  candidate uses a single, deterministic parameter vector.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    paths = []
    for i in range(num_worlds):
        if cone_cfg is not None:
            # Fixed config: use caller-provided params, but still jitter cone positions
            _cfg = cone_cfg
            fixed_wz_len = workzone_length_m if workzone_length_m is not None else 30.0
            scene = generate_workzone_scene(i, seed=seed, road_length_m=road_length_m,
                                            workzone_length_m=fixed_wz_len)
            cones = generate_taper_cones(_cfg, rng, workzone_center_x=0.0,
                                         workzone_length_m=fixed_wz_len)
            scene["workzone"] = {
                "workzone_length_m": float(fixed_wz_len),
                "workzone_center_x": 0.0,
                "speed_limit_mps": _cfg.speed_limit_mps,
                "cones": [{"x": c.x, "y": c.y, "z": c.z} for c in cones],
                "cone_cfg": {
                    "taper_length_m": _cfg.taper_length_m,
                    "cone_spacing_m": _cfg.cone_spacing_m,
                    "cone_lateral_jitter_m": _cfg.cone_lateral_jitter_m,
                    "exit_taper_length_m": _cfg.exit_taper_length_m,
                    "lane_width_m": _cfg.lane_width_m,
                    "speed_limit_mps": _cfg.speed_limit_mps,
                },
            }
            scene["agents"]["items"] = generate_agent_spawns(
                _cfg,
                road_length_m=scene["meta"]["road_length_m"],
                workzone_center_x=0.0,
                workzone_length_m=fixed_wz_len,
                num_agents=4,
            )
        else:
            scene = generate_workzone_scene(i, seed=seed, road_length_m=road_length_m,
                                            workzone_length_m=workzone_length_m)
            _cfg = None
        p = out / f"scene_workzone_{i:03d}.json"
        p.write_text(json.dumps(scene, indent=2))
        paths.append(p)
        n_cones = len(scene["workzone"]["cones"])
        cfg_info = scene["workzone"]["cone_cfg"]
        wz_len = scene["workzone"]["workzone_length_m"]
        print(f"  world {i:03d}: workzone={wz_len:.1f}m  "
              f"taper={cfg_info['taper_length_m']:.1f}m  "
              f"spacing={cfg_info['cone_spacing_m']:.1f}m  "
              f"cones={n_cones}")
    return paths


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="data/processed/workzone_scenes_json")
    p.add_argument("--num_worlds", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--road_length_m", type=float, default=200.0)
    args = p.parse_args()
    print(f"Generating {args.num_worlds} workzone scenes → {args.output_dir}")
    write_workzone_scenes(args.output_dir, args.num_worlds, args.seed, args.road_length_m)
    print("Done.")
