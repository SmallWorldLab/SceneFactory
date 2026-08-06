"""Launcher for src/policy_action_probe.py.

Boots the ordinary trainer in --test_mode scene_factory_policy_eval (so the env,
config parsing, scene pool, checkpoint load and reset are byte-identical to the
transfer eval) and then swaps ONLY the rollout function for the probe. No file
that a running job depends on is modified.

Usage (args after `--` are passed through to the trainer verbatim):
    PYTHONPATH=. python scripts/policy_action_probe_launch.py -- \
        --config ... --checkpoint_path ... --dynamics_mode physx ...
"""
from __future__ import annotations

import sys

if "--" in sys.argv:
    passthrough = sys.argv[sys.argv.index("--") + 1:]
else:
    passthrough = sys.argv[1:]

sys.argv = ["src/train_student_vehicle_goal_multiagent_rsl_rl.py"] + passthrough

import src.train_student_vehicle_goal_multiagent_rsl_rl as T  # noqa: E402  (launches app)
from src.policy_action_probe import run_policy_action_probe  # noqa: E402

T._run_scene_factory_policy_eval = run_policy_action_probe
T.main()
T.simulation_app.close()
