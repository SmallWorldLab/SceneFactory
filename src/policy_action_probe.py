"""
Policy-action probe: what does a trained policy actually COMMAND, and what does
the vehicle DO with it?

WHY THIS EXISTS
---------------
A policy trained with the kinematic-bicycle backend scores 99.8% on its own
backend and 1.6% on PhysX, with ep_steps pinned at the episode cap and low
crash rates -- the car is not crashing, it is not moving. Speed calibration
(bicycle_max_speed_mps 15.0 vs 4.5) and ground/traction health have both been
ruled out. What has NOT been measured is the command itself.

This probe loads a checkpoint through the ORDINARY scene_factory_policy_eval
boot path (same env, config, scene pool, reset) and then, instead of only
scoring episodes, records per step:

    raw policy output   (pre-clamp, all 3 channels)
    semantic action     (post-clamp: throttle>=0, brake>=0) -- what the env applies
    wheel omega         (PhysX joint_vel on the 4 wheel joints)
    body planar speed
    slip = 1 - v / (omega * r),  r = 0.35
    goal distance

over the SPAWNED, NOT-DONE agent population only (parked slots and finished
agents are masked out of every statistic; their actions are force-zeroed by
_pre_physics_step and would dilute the mean toward zero for a reason that has
nothing to do with the policy).

THE DISCRIMINATING SIGNAL
-------------------------
  throttle high, omega high, speed ~0, slip -> 1   wheels spin, no traction
  throttle ~0,   omega ~0,   speed ~0              the policy is not asking to
                                                   move. Under PhysX, throttle=0
                                                   is not a coast: the wheel
                                                   velocity target is 0 and the
                                                   implicit actuator actively
                                                   drags omega to zero.
  throttle high, omega high, speed high            healthy.

VALIDATION REQUIREMENT
----------------------
Run the physx policy on the physx backend FIRST. That cell is known to give
~94% SR and ~4.4 m/s. If the probe does not reproduce a moving car there, the
probe is wrong and nothing it says about the bicycle policy counts.

Launch: scripts/policy_action_probe_launch.py  (monkeypatches the eval entry
point of the ordinary trainer; nothing under src/ that a running job uses is
modified).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

WHEEL_RADIUS_M = 0.35
PROBE_STEPS = int(os.environ.get("PAP_STEPS", "600"))
TRACE_EVERY = int(os.environ.get("PAP_TRACE_EVERY", "25"))

# Counterfactual action remap, applied to the RAW policy output before env.step.
#   none                        pass through
#   brake2throttle              raw_t := raw_b, raw_b := -1
#                               (semantic throttle := brake command, brake := 0)
#   brake2throttle_flipsteer    as above, and raw_s := -raw_s
# Rationale: if the policy is driving in reverse via the brake channel, then
# re-pointing that channel at PhysX's drive should make the same weights drive.
REMAP = os.environ.get("PAP_REMAP", "none").strip().lower()


def _q(x: torch.Tensor) -> dict:
    """Summary stats of a 1-D tensor."""
    if x.numel() == 0:
        return {"n": 0}
    x = x.float()
    qs = torch.quantile(x, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=x.device))
    return {
        "n": int(x.numel()),
        "mean": float(x.mean()),
        "sd": float(x.std()) if x.numel() > 1 else 0.0,
        "min": float(x.min()),
        "p05": float(qs[0]), "p25": float(qs[1]), "p50": float(qs[2]),
        "p75": float(qs[3]), "p95": float(qs[4]),
        "max": float(x.max()),
    }


def run_policy_action_probe(base_env, env, runner, run_dir: Path) -> None:
    import src.train_student_vehicle_goal_multiagent_rsl_rl as T

    ckpt = str(T.args_cli.checkpoint_path).strip()
    if not ckpt:
        raise ValueError("policy_action_probe requires --checkpoint_path")
    runner.load(ckpt, load_optimizer=False, map_location=str(runner.device))
    policy = runner.get_inference_policy(device=str(runner.device))

    mode = str(base_env.cfg.dynamics_mode)
    n_env = int(base_env.num_envs)
    n_agent = len(base_env.cfg.possible_agents)
    dt = float(base_env.cfg.sim.dt) * float(base_env.cfg.decimation)
    is_physx = mode == "physx"

    print(f"[ActionProbe] ckpt={ckpt}", flush=True)
    print(f"[ActionProbe] REMAP={REMAP}", flush=True)
    print(f"[ActionProbe] backend={mode} envs={n_env} agents={n_agent} dt={dt:.4f} "
          f"v_max={base_env.cfg.bicycle_max_speed_mps} steps={PROBE_STEPS}", flush=True)

    base_env.consume_last_reset_world_episode_summaries()
    obs, _ = env.reset()
    base_env.consume_last_reset_world_episode_summaries()
    base_env.episode_length_buf.zero_()
    if hasattr(base_env, "_steps_since_reset_buf"):
        base_env._steps_since_reset_buf.zero_()
    if hasattr(env, "_slot_episode_length_buf"):
        env._slot_episode_length_buf.zero_()
    if hasattr(env, "_slot_dead_mask") and hasattr(env, "_flatten_agent_done_mask"):
        env._slot_dead_mask = env._flatten_agent_done_mask()

    dev = base_env.device
    wheel_ids = None
    if is_physx:
        wheel_ids = [torch.as_tensor(base_env._wheel_joint_ids[a], dtype=torch.long, device=dev)
                     for a in range(n_agent)]

    # Pooled samples over the whole probe, restricted to spawned & alive agents.
    pool = {k: [] for k in ("raw_t", "raw_s", "raw_b", "sem_t", "sem_s", "sem_b",
                            "speed", "vx", "omega", "slip", "gdist", "yawrate")}
    trace: list[dict] = []
    start_pos = None
    last_pos = None

    with torch.inference_mode():
        for step in range(PROBE_STEPS):
            actions = policy(obs)
            if REMAP != "none":
                a = actions.clone()
                a[..., 0] = actions[..., 2]
                a[..., 2] = -1.0
                if REMAP == "brake2throttle_flipsteer":
                    a[..., 1] = -actions[..., 1]
                actions = a
            out = env.step(actions)
            obs = out[0]

            spawned = base_env._spawned_agent_mask.detach()          # [A, E] bool
            done = base_env._agent_done_mask.detach()                # [A, E] bool
            live = spawned & (~done)

            raw = base_env._raw_actions.detach()                     # [A, E, 3]
            sem = base_env._semantic_actions.detach()                # [A, E, 3]

            sp, om, pos, yr, vxl = [], [], [], [], []
            for a in range(n_agent):
                veh = base_env._vehicles[a]
                sp.append(torch.norm(veh.data.root_lin_vel_b.detach()[..., :2], dim=-1))
                vxl.append(veh.data.root_lin_vel_b.detach()[:, 0])   # SIGNED longitudinal
                pos.append(veh.data.root_pos_w.detach()[:, :2])
                yr.append(veh.data.root_ang_vel_w.detach()[:, 2].abs())
                if is_physx:
                    om.append(veh.data.joint_vel.detach()[:, wheel_ids[a]].abs().mean(dim=1))
                else:
                    om.append(torch.zeros(n_env, device=dev))
            speed = torch.stack(sp)          # [A, E]
            vx = torch.stack(vxl)            # [A, E] signed body-frame longitudinal
            omega = torch.stack(om)
            posxy = torch.stack(pos)         # [A, E, 2]
            yawr = torch.stack(yr)
            gdist = base_env._current_goal_distance.detach()

            if step == 0:
                start_pos = posxy.clone()
                print(f"[ActionProbe] spawned {int(spawned.sum())}/{spawned.numel()} "
                      f"({spawned.float().mean():.1%})", flush=True)
            last_pos = posxy

            if int(live.sum()) > 0:
                surf = omega * WHEEL_RADIUS_M
                slip = torch.where(surf > 0.1,
                                   1.0 - speed / surf.clamp(min=1e-6),
                                   torch.zeros_like(speed)).clamp(0.0, 1.0)
                pool["raw_t"].append(raw[..., 0][live].cpu())
                pool["raw_s"].append(raw[..., 1][live].cpu())
                pool["raw_b"].append(raw[..., 2][live].cpu())
                pool["sem_t"].append(sem[..., 0][live].cpu())
                pool["sem_s"].append(sem[..., 1][live].cpu())
                pool["sem_b"].append(sem[..., 2][live].cpu())
                pool["speed"].append(speed[live].cpu())
                pool["vx"].append(vx[live].cpu())
                pool["omega"].append(omega[live].cpu())
                pool["slip"].append(slip[live].cpu())
                pool["gdist"].append(gdist[live].cpu())
                pool["yawrate"].append(yawr[live].cpu())

                if step % TRACE_EVERY == 0:
                    row = {
                        "step": step,
                        "n_live": int(live.sum()),
                        "raw_throttle": float(raw[..., 0][live].mean()),
                        "sem_throttle": float(sem[..., 0][live].mean()),
                        "sem_brake": float(sem[..., 2][live].mean()),
                        "abs_steer": float(sem[..., 1][live].abs().mean()),
                        "speed": float(speed[live].mean()),
                        "omega": float(omega[live].mean()),
                        "slip": float(slip[live].mean()),
                        "gdist": float(gdist[live].mean()),
                    }
                    trace.append(row)
                    print(f"[ActionProbe] s{step:>4} live={row['n_live']:>4} "
                          f"raw_thr={row['raw_throttle']:+.3f} sem_thr={row['sem_throttle']:.3f} "
                          f"brk={row['sem_brake']:.3f} |steer|={row['abs_steer']:.3f} "
                          f"v={row['speed']:.3f} w={row['omega']:.2f} slip={row['slip']:.3f} "
                          f"gd={row['gdist']:.2f}", flush=True)

    stats = {k: _q(torch.cat(v)) if v else {"n": 0} for k, v in pool.items()}

    # Extra discrete fractions on the throttle channel.
    extra = {}
    if pool["raw_t"]:
        rt = torch.cat(pool["raw_t"])
        st = torch.cat(pool["sem_t"])
        sb = torch.cat(pool["sem_b"])
        sv = torch.cat(pool["speed"])
        extra = {
            "frac_raw_throttle_negative": float((rt < 0).float().mean()),
            "frac_sem_throttle_zero": float((st <= 1e-4).float().mean()),
            "frac_sem_throttle_gt_0p5": float((st > 0.5).float().mean()),
            "frac_sem_throttle_gt_0p95": float((st > 0.95).float().mean()),
            "frac_brake_gt_0p5": float((sb > 0.5).float().mean()),
            "frac_speed_lt_0p5": float((sv < 0.5).float().mean()),
            "frac_speed_lt_2p5": float((sv < 2.5).float().mean()),  # idle threshold
            "frac_vx_negative": float((torch.cat(pool["vx"]) < -0.1).float().mean()),
            "mean_vx_signed": float(torch.cat(pool["vx"]).mean()),
        }

    disp = torch.norm(last_pos - start_pos, dim=-1)
    spawned = base_env._spawned_agent_mask.detach().cpu()
    disp_spawned = disp.cpu()[spawned]

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": ckpt,
        "backend": mode,
        "bicycle_max_speed_mps": float(base_env.cfg.bicycle_max_speed_mps),
        "num_envs": n_env, "num_agents": n_agent, "dt_s": dt,
        "probe_steps": PROBE_STEPS,
        "stats": stats,
        "fractions": extra,
        "displacement_m": _q(disp_spawned),
        "trace": trace,
    }
    (run_dir / "policy_action_probe.json").write_text(json.dumps(report, indent=2))

    print("\n===== POLICY ACTION PROBE =====", flush=True)
    print(f"backend={mode}  ckpt={Path(ckpt).parent.name}/{Path(ckpt).name}", flush=True)
    hdr = f"{'channel':>12} {'mean':>9} {'sd':>8} {'p05':>8} {'p50':>8} {'p95':>8} {'max':>8}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for k in ("raw_t", "raw_s", "raw_b", "sem_t", "sem_s", "sem_b",
              "speed", "vx", "omega", "slip", "yawrate", "gdist"):
        s = stats[k]
        if not s.get("n"):
            continue
        print(f"{k:>12} {s['mean']:>9.4f} {s['sd']:>8.4f} {s['p05']:>8.4f} "
              f"{s['p50']:>8.4f} {s['p95']:>8.4f} {s['max']:>8.4f}", flush=True)
    print("", flush=True)
    for k, v in extra.items():
        print(f"  {k:<32} {v:.4f}", flush=True)
    d = report["displacement_m"]
    if d.get("n"):
        print(f"\n  displacement over {PROBE_STEPS} steps ({PROBE_STEPS*dt:.1f} s): "
              f"mean {d['mean']:.2f} m  median {d['p50']:.2f} m  max {d['max']:.2f} m", flush=True)
    print(f"\n[ActionProbe] wrote {run_dir/'policy_action_probe.json'}", flush=True)
