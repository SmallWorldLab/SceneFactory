"""The scenario API must keep the obstacle registry and the road-point pool in sync.

That invariant is the whole reason the module exists: obstacles live in two
tensors, and writing one without the other produces an agent that collides with
something it cannot perceive.  These tests run on a fake env -- no Isaac, no GPU.
"""

import torch

from src.scene_factory_scenario_api import (
    clear_keepout_boxes,
    clear_obstacles,
    ensure_keepout_capacity,
    keepout_capacity,
    obstacle_capacity,
    obstacle_slice,
    set_keepout_boxes,
    set_obstacles,
)


class FakeEnv:
    """Minimal stand-in exposing only what the scenario API touches."""

    def __init__(self, num_envs=3, n_road=20, n_obstacles=8, n_boxes=2):
        self.num_envs = num_envs
        self._obstacle_points_xy_m = torch.zeros(num_envs, n_obstacles, 2)
        self._obstacle_points_valid = torch.zeros(num_envs, n_obstacles, dtype=torch.bool)
        # pool = road points followed by the obstacle slice at the tail
        self._lane_touch_points_xy_m = torch.zeros(num_envs, n_road + n_obstacles, 2)
        self._keepout_boxes_m = torch.zeros(num_envs, n_boxes, 5)
        self._keepout_boxes_count = torch.zeros(num_envs, dtype=torch.long)


def test_obstacle_slice_is_the_tail_of_the_pool():
    env = FakeEnv(n_road=20, n_obstacles=8)
    assert obstacle_slice(env) == slice(20, 28)
    assert obstacle_capacity(env) == 8


def test_set_obstacles_writes_registry_and_pool_together():
    env = FakeEnv()
    xy = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    set_obstacles(env, xy, env_ids=[1])
    sl = obstacle_slice(env)

    assert torch.equal(env._obstacle_points_xy_m[1, :2], xy)
    assert env._obstacle_points_valid[1, :2].all()
    # the perceived copy must match the collided copy
    assert torch.equal(env._lane_touch_points_xy_m[1, sl.start:sl.start + 2], xy)
    # untouched worlds stay empty
    assert not env._obstacle_points_valid[0].any()
    assert env._lane_touch_points_xy_m[0, sl].abs().sum() == 0


def test_writing_again_clears_the_previous_layout_in_both_places():
    env = FakeEnv()
    set_obstacles(env, torch.tensor([[9.0, 9.0]] * 5), env_ids=[0])
    set_obstacles(env, torch.tensor([[1.0, 1.0]]), env_ids=[0])
    sl = obstacle_slice(env)
    assert int(env._obstacle_points_valid[0].sum()) == 1
    # stale rows must not linger in the perceived pool
    assert env._lane_touch_points_xy_m[0, sl.start + 1:sl.stop].abs().sum() == 0


def test_surplus_obstacles_are_dropped_not_raised():
    env = FakeEnv(n_obstacles=4)
    set_obstacles(env, torch.ones(10, 2), env_ids=[0])
    assert int(env._obstacle_points_valid[0].sum()) == 4


def test_counts_limit_what_is_marked_valid():
    env = FakeEnv()
    xy = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    set_obstacles(env, xy, env_ids=[2], counts=[3])
    assert int(env._obstacle_points_valid[2].sum()) == 3


def test_clear_obstacles_empties_both_tensors():
    env = FakeEnv()
    set_obstacles(env, torch.ones(3, 2), env_ids=[0])
    clear_obstacles(env, env_ids=[0])
    sl = obstacle_slice(env)
    assert not env._obstacle_points_valid[0].any()
    assert env._lane_touch_points_xy_m[0, sl].abs().sum() == 0


def test_broadcast_layout_applies_to_every_world():
    env = FakeEnv(num_envs=3)
    set_obstacles(env, torch.tensor([[5.0, 5.0]]))
    assert int(env._obstacle_points_valid.sum()) == 3


def test_keepout_capacity_grows_and_preserves_existing_boxes():
    env = FakeEnv(n_boxes=1)
    env._keepout_boxes_m[0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0, 0.5])
    assert ensure_keepout_capacity(env, 4) == 4
    assert keepout_capacity(env) == 4
    assert torch.equal(env._keepout_boxes_m[0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0, 0.5]))


def test_set_keepout_boxes_updates_count_and_grows_if_needed():
    env = FakeEnv(n_boxes=1)
    boxes = torch.zeros(1, 3, 5)
    boxes[0, :, 0] = torch.tensor([1.0, 2.0, 3.0])
    set_keepout_boxes(env, boxes, env_ids=[0])
    assert keepout_capacity(env) >= 3
    assert int(env._keepout_boxes_count[0]) == 3
    assert env._keepout_boxes_m[0, 2, 0] == 3.0


def test_clear_keepout_boxes_zeroes_the_count():
    env = FakeEnv(n_boxes=3)
    set_keepout_boxes(env, torch.zeros(1, 2, 5), env_ids=[0])
    clear_keepout_boxes(env, env_ids=[0])
    assert int(env._keepout_boxes_count[0]) == 0
