"""Mutate a live environment's obstacles and keep-out regions.

Why this exists
---------------
Research code needs to change what is in a world *without* rebuilding USD --
that is what makes a design optimizer affordable, since a rebuild costs far more
than a rollout.  Until now that was done by reaching into the environment and
assigning to its private tensors directly.  Two problems with that:

1. There is an invariant the caller had to know.  Obstacles live in **two**
   places: the obstacle registry (``_obstacle_points_xy_m`` /
   ``_obstacle_points_valid``, used for collision and proximity) and a slice at
   the tail of the road-point pool (``_lane_touch_points_xy_m``, which is what
   the policy actually perceives via k-NN).  Write one and forget the other and
   the agent collides with something it cannot see.  Callers also had to compute
   the slice offset themselves as ``total_points - max_obstacles``.

2. Every such write is an undeclared dependency on a private name.  Renaming a
   backbone tensor broke research code silently, hours into a GPU rollout.

This module is the one place that knows the invariant.  It is deliberately
small: it does not decide *what* to place, only how to place it consistently.
Deciding the layout -- a taper, a debris field, a parked row -- belongs to the
research line.

All functions take the environment first and are safe to call every step.
"""

from __future__ import annotations

from typing import Sequence

import torch

# Point type written into the road-point pool for obstacles, so the policy can
# tell them apart from lane centres and road edges.
OBSTACLE_POINT_TYPE = 10

# Default number of obstacle slots to pre-allocate.  Kept equal to the workzone
# taper builder's worst case; see src/obstacle_randomizer.DEFAULT_OBSTACLE_SLOTS.
from src.obstacle_randomizer import (  # noqa: E402
    DEFAULT_OBSTACLE_SLOTS,
    ensure_obstacle_buffer,
)

__all__ = [
    "OBSTACLE_POINT_TYPE",
    "DEFAULT_OBSTACLE_SLOTS",
    "obstacle_capacity",
    "obstacle_slice",
    "ensure_obstacle_capacity",
    "set_obstacles",
    "clear_obstacles",
    "keepout_capacity",
    "ensure_keepout_capacity",
    "set_keepout_boxes",
    "set_keepout_slot",
    "clear_keepout_boxes",
]


# --------------------------------------------------------------------------- #
# obstacles
# --------------------------------------------------------------------------- #
def obstacle_capacity(env) -> int:
    """How many obstacle slots each world currently has."""
    return int(env._obstacle_points_xy_m.shape[1])


def obstacle_slice(env) -> slice:
    """Where the obstacles sit inside the road-point pool.

    The obstacle slice is the tail of the pool, so the offset is
    ``total_points - obstacle_capacity``.  Callers should never compute this
    themselves -- that is exactly the duplicated knowledge this module removes.
    """
    total = int(env._lane_touch_points_xy_m.shape[1])
    return slice(total - obstacle_capacity(env), total)


def ensure_obstacle_capacity(env, min_slots: int = DEFAULT_OBSTACLE_SLOTS) -> int:
    """Grow the obstacle buffers (and the pool slice) to hold ``min_slots``.

    No-op when capacity already suffices.  Returns the resulting capacity.
    Creates no USD prims: obstacles placed this way are perceived and collided
    with, but not rendered.
    """
    ensure_obstacle_buffer(env, int(min_slots))
    return obstacle_capacity(env)


def set_obstacles(
    env,
    xy: torch.Tensor,
    *,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    counts: Sequence[int] | torch.Tensor | None = None,
) -> None:
    """Replace the obstacles of the given worlds.

    Parameters
    ----------
    xy
        ``[E, N, 2]`` positions in world metres, or ``[N, 2]`` to apply the same
        layout to every selected world.  ``N`` may exceed capacity, in which case
        the surplus is dropped rather than raising -- a design that asks for more
        cones than the buffer holds should degrade, not crash a rollout.
    env_ids
        Which worlds to write.  ``None`` means all of them.
    counts
        Per-world count of valid rows in ``xy``.  ``None`` means "all N rows".

    Writes the obstacle registry and the road-point pool slice together, so the
    two cannot disagree.
    """
    if xy.dim() == 2:
        xy = xy.unsqueeze(0).expand(env.num_envs if env_ids is None else len(env_ids), -1, -1)
    device = env._obstacle_points_xy_m.device
    xy = xy.to(device)

    ids = (
        torch.arange(env.num_envs, device=device)
        if env_ids is None
        else torch.as_tensor(list(env_ids), dtype=torch.long, device=device)
    )
    cap = obstacle_capacity(env)
    sl = obstacle_slice(env)
    n_avail = int(xy.shape[1])

    for row, env_idx in enumerate(ids.tolist()):
        n = n_avail if counts is None else int(counts[row])
        n = max(0, min(n, cap, n_avail))
        env._obstacle_points_xy_m[env_idx].zero_()
        env._obstacle_points_valid[env_idx].fill_(False)
        env._lane_touch_points_xy_m[env_idx, sl, :].zero_()
        if n == 0:
            continue
        pts = xy[row, :n]
        env._obstacle_points_xy_m[env_idx, :n] = pts
        env._obstacle_points_valid[env_idx, :n] = True
        env._lane_touch_points_xy_m[env_idx, sl.start:sl.start + n, :] = pts


def clear_obstacles(env, *, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
    """Remove every obstacle from the given worlds (all worlds if ``env_ids`` is None)."""
    empty = torch.zeros(
        (1, 0, 2), dtype=env._obstacle_points_xy_m.dtype, device=env._obstacle_points_xy_m.device
    )
    n = env.num_envs if env_ids is None else len(list(env_ids))
    set_obstacles(env, empty.expand(n, 0, 2), env_ids=env_ids, counts=[0] * n)


# --------------------------------------------------------------------------- #
# keep-out regions
# --------------------------------------------------------------------------- #
def keepout_capacity(env) -> int:
    """How many keep-out box slots each world currently has."""
    return int(env._keepout_boxes_m.shape[1])


def ensure_keepout_capacity(env, min_boxes: int) -> int:
    """Grow the keep-out box buffer to hold ``min_boxes`` per world.

    Existing boxes are preserved.  Returns the resulting capacity.
    """
    cur = keepout_capacity(env)
    if cur >= int(min_boxes):
        return cur
    old = env._keepout_boxes_m
    new = torch.zeros((env.num_envs, int(min_boxes), 5), dtype=old.dtype, device=old.device)
    if cur:
        new[:, :cur, :] = old
    env._keepout_boxes_m = new
    return int(min_boxes)


def set_keepout_boxes(
    env,
    boxes: torch.Tensor,
    *,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    counts: Sequence[int] | torch.Tensor | None = None,
) -> None:
    """Replace the keep-out regions of the given worlds.

    ``boxes`` is ``[E, M, 5]`` as ``(cx, cy, half_len, half_wid, yaw_rad)``, or
    ``[M, 5]`` to apply one layout everywhere.  Grows capacity if needed.
    """
    if boxes.dim() == 2:
        n_env = env.num_envs if env_ids is None else len(list(env_ids))
        boxes = boxes.unsqueeze(0).expand(n_env, -1, -1)
    device = env._keepout_boxes_m.device
    boxes = boxes.to(device)
    ensure_keepout_capacity(env, int(boxes.shape[1]))

    ids = (
        torch.arange(env.num_envs, device=device)
        if env_ids is None
        else torch.as_tensor(list(env_ids), dtype=torch.long, device=device)
    )
    m_avail = int(boxes.shape[1])
    for row, env_idx in enumerate(ids.tolist()):
        m = m_avail if counts is None else int(counts[row])
        m = max(0, min(m, keepout_capacity(env), m_avail))
        env._keepout_boxes_m[env_idx].zero_()
        if m:
            env._keepout_boxes_m[env_idx, :m] = boxes[row, :m]
        env._keepout_boxes_count[env_idx] = m


def set_keepout_slot(env, env_idx: int, slot: int, box: Sequence[float] | torch.Tensor) -> None:
    """Write a single keep-out box into one slot of one world.

    Exists because a caller often wants to replace part of a world's layout while
    keeping the rest -- the scene's own closure in slot 0, say, with generated
    regions appended after it.  Grows capacity if ``slot`` is past the end, and
    raises the stored count so the new slot is actually consulted.
    """
    ensure_keepout_capacity(env, int(slot) + 1)
    env._keepout_boxes_m[int(env_idx), int(slot)] = torch.as_tensor(
        box, dtype=env._keepout_boxes_m.dtype, device=env._keepout_boxes_m.device
    )
    if int(env._keepout_boxes_count[int(env_idx)]) < int(slot) + 1:
        env._keepout_boxes_count[int(env_idx)] = int(slot) + 1


def clear_keepout_boxes(env, *, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
    """Remove every keep-out region from the given worlds."""
    n = env.num_envs if env_ids is None else len(list(env_ids))
    empty = torch.zeros((n, 0, 5), dtype=env._keepout_boxes_m.dtype, device=env._keepout_boxes_m.device)
    set_keepout_boxes(env, empty, env_ids=env_ids, counts=[0] * n)
