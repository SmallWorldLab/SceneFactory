"""
obstacle_randomizer.py
==================
Two utilities for training driving policies with random cone obstacles:

  ensure_obstacle_buffer(env, min_slots)
      Expands the env's cone tensors to at least min_slots WITHOUT touching USD.
      Call once after env setup for any scene type (Waymo or workzone) that was
      loaded without workzone cone metadata (i.e. _obstacle_points_xy_m is empty).

  RandomObstacleDropper
      Writes N random cone positions into the env's cone tensors at each episode
      reset.  Cones are sampled from the existing road-point pool so they land
      on the road surface, with small lateral noise.  Register via:

          dropper = RandomObstacleDropper(env, num_obstacles=20)
          env._obstacle_drop_fn = dropper.drop

      The env will then call dropper.drop(env_ids) automatically at the start of
      every _reset_idx, re-randomizing cones for those worlds.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

# Integer type label used for cone points in the road KNN pool.
_OBSTACLE_POINT_TYPE = 10

# Default number of obstacle slots to pre-allocate in the road-point pool.
#
# The pool is a fixed-width tensor, so the slot count has to be decided here,
# before any scene is loaded, and it must be the worst case over every obstacle
# layout a caller might place at runtime -- a caller that needs more slots than
# this cannot get them later.
#
# DO NOT change this casually: the obstacle slots are part of the road-point
# pool, so the buffer size feeds the observation vector width.  Changing it
# invalidates every existing checkpoint.
DEFAULT_OBSTACLE_SLOTS = 303


def ensure_obstacle_buffer(env: object, min_slots: int) -> None:
    """Expand all cone-related tensors to at least min_slots entries per env.

    For Waymo scenes loaded without a workzone block, _obstacle_points_xy_m has
    shape [E, 0, 2] and the road-point pool has no cone slice.  This function
    expands both in-place so that RandomObstacleDropper (or HotSwapConeBuilder) can
    write cone positions and agents will observe them via KNN.

    All new slots are initialised to zeros / False (invalid) — no cones are
    placed until the dropper runs.  No USD prims are created.

    Parameters
    ----------
    env      : StudentVehicleMultiAgentGoalEnv (already set up)
    min_slots: minimum number of cone slots required (e.g. MAX_CONE_SLOTS = 128)
    """
    cur = int(env._obstacle_points_xy_m.shape[1])
    if cur >= min_slots:
        return

    E   = env.num_envs
    add = min_slots - cur
    dev = env.device

    # ── Expand _obstacle_points_xy_m and _obstacle_points_valid ──────────────
    new_xy = torch.zeros((E, min_slots, 2), dtype=torch.float32, device=dev)
    new_valid = torch.zeros((E, min_slots), dtype=torch.bool, device=dev)
    if cur > 0:
        new_xy[:, :cur]    = env._obstacle_points_xy_m
        new_valid[:, :cur] = env._obstacle_points_valid
    env._obstacle_points_xy_m  = new_xy
    env._obstacle_points_valid = new_valid

    # ── Expand the road-point pool (lane_touch tensors) ────────────────────
    # Append `add` zero/invalid/type-10 slots at the end of each tensor.
    env._lane_touch_points_xy_m = torch.cat([
        env._lane_touch_points_xy_m,
        torch.zeros((E, add, 2), dtype=torch.float32, device=dev),
    ], dim=1)
    env._lane_touch_dirs_xy = torch.cat([
        env._lane_touch_dirs_xy,
        torch.zeros((E, add, 2), dtype=torch.float32, device=dev),
    ], dim=1)
    env._lane_touch_half_lengths_m = torch.cat([
        env._lane_touch_half_lengths_m,
        torch.zeros((E, add), dtype=torch.float32, device=dev),
    ], dim=1)
    env._lane_touch_half_widths_m = torch.cat([
        env._lane_touch_half_widths_m,
        torch.zeros((E, add), dtype=torch.float32, device=dev),
    ], dim=1)
    env._lane_touch_types = torch.cat([
        env._lane_touch_types,
        torch.full((E, add), _OBSTACLE_POINT_TYPE, dtype=torch.long, device=dev),
    ], dim=1)
    env._lane_touch_valid = torch.cat([
        env._lane_touch_valid,
        torch.zeros((E, add), dtype=torch.bool, device=dev),
    ], dim=1)

    # ── Rebuild type one-hot and lane-touch mask ────────────────────────────
    type_dim = max(int(env._lane_touch_type_dim), _OBSTACLE_POINT_TYPE + 1)
    env._lane_touch_type_dim = type_dim
    env._lane_touch_type_one_hot = F.one_hot(
        env._lane_touch_types.clamp(min=0),
        num_classes=type_dim,
    ).to(dtype=torch.bool)
    env._lane_touch_type_one_hot &= env._lane_touch_valid.unsqueeze(-1)

    A = env._num_agents
    env._lane_touch_mask = torch.zeros(
        (A, E, type_dim), dtype=torch.bool, device=dev
    )
    env._lane_touch_mask_cache_valid = False

    print(
        f"[ensure_obstacle_buffer] Expanded cone slots {cur} → {min_slots} "
        f"(+{add} slots, no USD prims created)",
        flush=True,
    )


class RandomObstacleDropper:
    """Drops N cones into the env at each episode reset.

    Two placement modes:

    od_corridor=True  (default)
        Cones are sampled from road points that lie within `corridor_width_m`
        of the straight line connecting each agent's spawn to its goal.  This
        guarantees agents must interact with cones en-route, producing a strong
        avoidance training signal.

    od_corridor=False
        Legacy mode: cones are sampled uniformly from all road points.

    Register with the env after calling ensure_obstacle_buffer:

        dropper = RandomObstacleDropper(env, num_obstacles=20)
        env._obstacle_drop_fn = dropper.drop

    Parameters
    ----------
    env              : StudentVehicleMultiAgentGoalEnv (with cone buffer sized)
    num_obstacles        : cones placed per env per episode
    noise_std_m      : Gaussian XY jitter on sampled road positions (m)
    seed             : RNG seed
    od_corridor      : if True, place cones along agent OD paths
    corridor_width_m : half-width of the OD corridor (m)
    """

    def __init__(
        self,
        env: object,
        num_obstacles: int = 20,
        noise_std_m: float = 0.6,
        seed: int = 42,
        od_corridor: bool = True,
        corridor_width_m: float = 4.0,
    ) -> None:
        self.env              = env
        self.num_obstacles        = num_obstacles
        self.noise_std_m      = noise_std_m
        self.od_corridor      = od_corridor
        self.corridor_width_m = corridor_width_m

        self._g = torch.Generator(device=env.device)
        self._g.manual_seed(seed)

        total   = int(env._lane_touch_points_xy_m.shape[1])
        max_c   = int(env._obstacle_points_xy_m.shape[1])
        self._obstacle_offset = total - max_c
        self._max_cones   = max_c
        self._road_n      = self._obstacle_offset

        if self._road_n <= 0:
            raise ValueError(
                "RandomObstacleDropper: no road points found in _lane_touch_points_xy_m. "
                "Call ensure_obstacle_buffer after the env is fully set up."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _od_corridor_weights(self, env_ids_t: torch.Tensor, B: int) -> torch.Tensor:
        """Return [B, N] float weights: 1.0 for road points inside any agent's OD
        corridor, 0.0 outside.  Falls back to all-ones when a corridor is empty."""
        env  = self.env
        dev  = env.device
        A    = env._num_agents
        w    = self.corridor_width_m

        # Env-local spawn XY:  [A, B, 2]
        spawn_xy = env._previous_root_pos_xy[:, env_ids_t, :]

        # Env-local goal XY:  [A, B, 2]
        #   _goal_pos_w is world-frame [A, E, 3]; subtract env origin to get local.
        goal_xy_w   = env._goal_pos_w[:, env_ids_t, :2]           # [A, B, 2]
        env_orig_xy = env.scene.env_origins[env_ids_t, :2]        # [B, 2]
        goal_xy     = goal_xy_w - env_orig_xy.unsqueeze(0)        # [A, B, 2]

        # Which agents actually spawned this episode:  [A, B]
        #   Non-spawned agents sit at env-local (0,0) with a stale parking goal,
        #   so their corridor would pull cones to the origin — exclude them.
        spawned = env._spawned_agent_mask[:, env_ids_t]           # [A, B] bool

        # Road points (env-local):  [B, N, 2]  and their validity mask [B, N].
        #   The [:_road_n] prefix contains zero-padding (invalid) slots at
        #   env-local (0,0); they must never be sampled as cone positions.
        road_pts   = env._lane_touch_points_xy_m[env_ids_t, :self._road_n, :]
        road_valid = env._lane_touch_valid[env_ids_t, :self._road_n]   # [B, N] bool
        N = road_pts.shape[1]

        # Build corridor mask via loop over agents (avoids [B, A, N] alloc).
        in_corridor = torch.zeros((B, N), dtype=torch.bool, device=dev)
        for a in range(A):
            spawned_a = spawned[a]                 # [B] — skip non-spawned agents
            if not bool(spawned_a.any()):
                continue
            S = spawn_xy[a].unsqueeze(1)          # [B, 1, 2]
            G = goal_xy[a].unsqueeze(1)            # [B, 1, 2]
            d = G - S                              # [B, 1, 2]
            # clamp so zero-length segments degenerate gracefully
            d_len_sq = (d * d).sum(-1, keepdim=True).clamp(min=1.0)  # [B, 1, 1]
            # project each road point onto the segment, clamp t in [0,1]
            t = ((road_pts - S) * d).sum(-1, keepdim=True) / d_len_sq  # [B, N, 1]
            t = t.clamp(0.0, 1.0)
            proj = S + t * d                      # [B, N, 2]
            dist = torch.linalg.norm(road_pts - proj, dim=-1)  # [B, N]
            # only envs where this agent spawned contribute to the corridor
            in_corridor |= (dist < w) & spawned_a.unsqueeze(1)

        # Restrict to valid road points BEFORE the empty-row check, so multinomial
        # can never see an all-zero weight row.
        in_corridor &= road_valid

        # Fallback: if an env has zero in-corridor points, sample from any VALID
        # road point (never the invalid origin padding).
        empty = in_corridor.sum(dim=1) == 0       # [B]
        if empty.any():
            in_corridor[empty] = road_valid[empty]

        return in_corridor.float()                # [B, N]

    def _sample_cone_xy(self, target, env_ids_t: torch.Tensor, B: int) -> torch.Tensor:
        """Return [B, n, 2] cone positions in env-local frame."""
        env = self.env
        dev = env.device
        n   = min(self.num_obstacles, self._max_cones)

        road_pts = env._lane_touch_points_xy_m[target, :self._road_n, :]  # [B, N, 2]

        if self.od_corridor:
            weights = self._od_corridor_weights(env_ids_t, B)              # [B, N]
        else:
            # Uniform over VALID road points only (skip zero-padding slots).
            weights = env._lane_touch_valid[env_ids_t, :self._road_n].float()  # [B, N]

        # Defensive guard: any env with an all-zero weight row (a scene with no
        # valid road points at all) would crash multinomial — fall back to
        # uniform over all slots for just those rows.
        zero_rows = weights.sum(dim=1) == 0                                # [B]
        if bool(zero_rows.any()):
            weights[zero_rows] = 1.0

        # Sample n road points per env according to weights.
        sample_idx = torch.multinomial(weights, n, replacement=True)       # [B, n]
        obstacle_xy    = road_pts.gather(1, sample_idx.unsqueeze(-1).expand(-1, -1, 2))

        return obstacle_xy  # [B, n, 2]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def drop(self, env_ids: Sequence[int] | None = None) -> None:
        """Resample cone positions for the given env indices (or all envs if None)."""
        env = self.env
        E   = env.num_envs
        n   = min(self.num_obstacles, self._max_cones)
        dev = env.device

        if env_ids is None or len(env_ids) == E:
            target    = slice(None)
            B         = E
            env_ids_t = torch.arange(E, dtype=torch.long, device=dev)
        else:
            # Convert to plain Python ints so a list of two env indices like
            # [20, 158] is treated as a 1-D fancy index into dim-0 of the env
            # tensors, not as per-dimension indices (which would try to index
            # dim-1 with 158, hitting the 128-slot cone dimension).
            target    = [int(i) for i in env_ids]
            B         = len(target)
            env_ids_t = torch.tensor(target, dtype=torch.long, device=dev)

        obstacle_xy = self._sample_cone_xy(target, env_ids_t, B)  # [B, n, 2]

        # Add small Gaussian noise so cones don't sit exactly on lane centres.
        if self.noise_std_m > 0.0:
            noise   = torch.randn(B, n, 2, generator=self._g, device=dev) * self.noise_std_m
            obstacle_xy = obstacle_xy + noise

        # Clear all cone slots for the target envs.
        # Use assignment (not zero_()/fill_()) because fancy indexing returns a
        # copy in PyTorch — in-place ops on it are silently no-ops.
        env._obstacle_points_xy_m[target] = 0.0
        env._obstacle_points_valid[target] = False
        env._lane_touch_points_xy_m[target, self._obstacle_offset:] = 0.0
        env._lane_touch_valid[target, self._obstacle_offset:] = False

        # Write new cone positions.
        env._obstacle_points_xy_m[target, :n]  = obstacle_xy
        env._obstacle_points_valid[target, :n]  = True
        env._lane_touch_points_xy_m[target, self._obstacle_offset:self._obstacle_offset + n] = obstacle_xy
        env._lane_touch_valid[target, self._obstacle_offset:self._obstacle_offset + n] = True

        # Update type one-hot for the cone slice (types are already set to _OBSTACLE_POINT_TYPE).
        # Only the validity dimension changed, so patch just the cone rows.
        type_dim = int(env._lane_touch_type_dim)
        env._lane_touch_type_one_hot[target, self._obstacle_offset:, :] = False
        # Set valid slots to True at the cone type index.
        if _OBSTACLE_POINT_TYPE < type_dim:
            valid_slice = env._lane_touch_valid[target, self._obstacle_offset:self._obstacle_offset + n]
            env._lane_touch_type_one_hot[
                target,
                self._obstacle_offset:self._obstacle_offset + n,
                _OBSTACLE_POINT_TYPE,
            ] = valid_slice

        # Mask cache must be rebuilt before the next observation.
        env._lane_touch_mask_cache_valid = False
