"""
Traction probe: does every agent actually drive when told to?

WHY THIS EXISTS
---------------
Two weather-ablation training runs (alldry / wet0to12) showed idle_penalty 39x
worse than the E backbone at the same iteration, with LOWER crash rates and
episode length pinned at the timeout cap -- i.e. the cars are not moving. The
configs differ from E's in exactly two keys: ``ground_cuboid_size_m`` (300 vs the
1000 default) and ``wheel_friction_cap`` (1.2 vs the USD-baked 1.0).

This probe tests the mechanism directly instead of inferring it from reward
curves. It spawns vehicles through the ORDINARY training path -- same env class,
same config, same scene pool, same multi-agent spawn -- then drives them with a
scripted constant throttle and records whether the body actually moves.

THE DISCRIMINATING SIGNAL is wheel omega vs body speed:

  wheels spinning, body still   -> wheels carry no normal force. Ground contact
                                   defect. slip -> 1.0.
  wheels still,    body still   -> the drive command never reached the joints.
                                   Not a traction problem at all.
  wheels spinning, body moving  -> healthy.

The other thing this settles is whether the failure is agent-index dependent.
The documented penetration bug (CLAUDE.md) claims agents AFTER agent 0 tunnel
past the contact band on the spawn drop, which single-agent tests never catch.
If that is what is happening, agent 0 drives and agents 1..k do not.

NOTHING about the env is modified except ``invincible = True``, so a car that
tips or drifts is not silently teleported back to spawn mid-measurement. Physics,
spawn logic, ground, and friction are exactly what training uses.

Output
------
run_dir/traction_probe.csv      one row per (env, agent)
run_dir/traction_probe.json     summary + run metadata

Launch:
    bash run_traction_probe.sh <variant>
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

SETTLE_STEPS = int(os.environ.get("TP_SETTLE_STEPS", "60"))   # spawn drop + suspension settle
RAMP_STEPS = int(os.environ.get("TP_RAMP_STEPS", "20"))       # throttle ramp (suppress launch impulse)
DRIVE_STEPS = int(os.environ.get("TP_DRIVE_STEPS", "600"))    # scripted full throttle
# 240 steps (~8 s) was too short to tell "healthy but still accelerating" from
# "stuck at a low speed": the vehicle had not reached its plateau, so the mean
# speed understated it. 600 (~20 s) lets the speed trace flatten.
IDLE_SPEED_MPS = 0.5                                          # matches choco_idle_speed_threshold_mps
WHEEL_RADIUS_M = 0.35


def _planar_speed(vel_b: torch.Tensor) -> torch.Tensor:
    return torch.norm(vel_b[..., :2], dim=-1)


def _build_action_dict(env, throttle: torch.Tensor, steer: torch.Tensor,
                       brake: torch.Tensor) -> dict:
    """throttle/steer/brake are [num_agents, num_envs]."""
    action_dict: dict[str, torch.Tensor] = {}
    for agent_idx, agent_id in enumerate(env.cfg.possible_agents):
        action_dict[agent_id] = torch.stack(
            [throttle[agent_idx], steer[agent_idx], brake[agent_idx]], dim=-1
        ).to(device=env.device, dtype=torch.float32)
    return action_dict


def run_traction_probe(env, run_dir: Path) -> None:
    device = env.device
    n_env = env.num_envs
    n_agent = len(env.cfg.possible_agents)
    dt = float(env.cfg.sim.dt) * float(env.cfg.decimation)

    print(f"[TractionProbe] {n_env} envs x {n_agent} agents, dt={dt:.4f}s, "
          f"settle={SETTLE_STEPS} ramp={RAMP_STEPS} drive={DRIVE_STEPS}", flush=True)
    print(f"[TractionProbe] ground_cuboid_size_m={getattr(env.cfg, 'ground_cuboid_size_m', None)} "
          f"wheel_friction_cap={getattr(env.cfg, 'wheel_friction_cap', None)}", flush=True)

    # MUST reset before stepping. Scene-derived spawn placement happens in the
    # reset path; without this every vehicle sits at its default cloned pose at
    # the env origin, all 16 overlapping, and PhysX ejects them into a vertical
    # tower -- an artifact of the harness, not of the environment.
    # (braking_validation.py:124 and physics_validation.py:109 do the same.)
    env.reset()

    zeros = torch.zeros((n_agent, n_env), device=device)

    # Per-agent accumulators, all [n_agent, n_env]
    max_speed = torch.zeros((n_agent, n_env))
    sum_speed = torch.zeros((n_agent, n_env))
    max_omega = torch.zeros((n_agent, n_env))
    sum_omega = torch.zeros((n_agent, n_env))
    n_samples = 0
    speed_trace: list[float] = []
    path_len = torch.zeros_like(sum_speed)
    max_dist_origin = torch.zeros((n_agent, n_env))
    z_trace: list[tuple[int, float, float, float]] = []
    spawn_xy: torch.Tensor | None = None
    start_xy: torch.Tensor | None = None
    z_settled: torch.Tensor | None = None
    z_end: torch.Tensor | None = None
    last_xy: torch.Tensor | None = None

    def _read_state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> speed, wheel_omega, xy, z ; each [n_agent, n_env]."""
        sp, om, xy, z = [], [], [], []
        for a in range(n_agent):
            veh = env._vehicles[a]
            wid = torch.as_tensor(env._wheel_joint_ids[a], dtype=torch.long, device=device)
            sp.append(_planar_speed(veh.data.root_lin_vel_b.detach()).cpu())
            om.append(veh.data.joint_vel.detach()[:, wid].abs().mean(dim=1).cpu())
            pos = veh.data.root_pos_w.detach().cpu()
            xy.append(pos[:, :2])
            z.append(pos[:, 2])
        return torch.stack(sp), torch.stack(om), torch.stack(xy), torch.stack(z)

    total = SETTLE_STEPS + RAMP_STEPS + DRIVE_STEPS
    for step in range(total):
        if step < SETTLE_STEPS:
            throttle = zeros
        elif step < SETTLE_STEPS + RAMP_STEPS:
            throttle = torch.full_like(zeros, (step - SETTLE_STEPS + 1) / RAMP_STEPS)
        else:
            throttle = torch.ones_like(zeros)

        env.step(_build_action_dict(env, throttle, zeros, zeros))

        speed, omega, xy, z = _read_state()

        # Settle-window z trace. Spawn z is env_origin_z + spawn_height_m for every
        # agent (verified in the reset path), so any height above that is acquired
        # AFTER spawn. Rising z during zero-action settle => agents are being pushed
        # up, i.e. interpenetration resolution, not a spawn-position bug.
        if step == 0:
            spawn_xy = xy.clone()
            # CRITICAL: not every agent slot is an active participant. Scenes carry
            # fewer than num_agents_per_env valid agents; the rest are PARKED
            # (_park_done_vehicle) and are excluded from every training metric via
            # _spawned_agent_mask. Measuring them as if they were drivers is
            # meaningless -- report spawned and parked separately.
            sm = env._spawned_agent_mask.detach().cpu().clone()          # [A, n_env] bool
            print(f"[TractionProbe] spawned agents: {int(sm.sum())}/{sm.numel()} "
                  f"({sm.float().mean():.1%})  per-env: "
                  f"{[int(c) for c in sm.sum(dim=0)]}", flush=True)
            for lbl, m in (("spawned", sm), ("parked", ~sm)):
                if int(m.sum()) == 0:
                    continue
                zz = z[m]
                print(f"[TractionProbe]   {lbl:<8} n={int(m.sum()):>4}  "
                      f"z at step0: mean {zz.mean():.3f} min {zz.min():.3f} max {zz.max():.3f}",
                      flush=True)
            # closest neighbour in XY within each env, per agent
            d = torch.cdist(xy.transpose(0, 1), xy.transpose(0, 1))       # [n_env, A, A]
            d += torch.eye(n_agent).unsqueeze(0) * 1e6
            nn_dist = d.min(dim=-1).values.transpose(0, 1)                 # [A, n_env]
            print(f"[TractionProbe] spawn XY nearest-neighbour distance: "
                  f"min {nn_dist.min():.3f} m  median {nn_dist.median():.3f} m  "
                  f"frac < 5 m: {(nn_dist < 5.0).float().mean():.1%}", flush=True)
        if step <= SETTLE_STEPS:
            sm = env._spawned_agent_mask.detach().cpu()
            zs = z[sm] if int(sm.sum()) else z
            z_trace.append((step, float(zs.mean()), float(zs.max()), float(z[~sm].mean())
                            if int((~sm).sum()) else float("nan")))

        if step == SETTLE_STEPS - 1:
            start_xy = xy.clone()
            z_settled = z.clone()

        # Distance from own env origin. Training terminates at
        # max_distance_from_origin_m; the probe runs invincible=True, so agents
        # that would have been reset instead keep going and can leave the ground
        # slab. Only agents that stay inside that radius are representative of
        # what training actually sees.
        eo = env.scene.env_origins.detach().cpu()[:, :2]                 # [n_env, 2]
        dist_origin = torch.norm(xy - eo.unsqueeze(0), dim=-1)           # [A, n_env]
        max_dist_origin = torch.maximum(max_dist_origin, dist_origin)

        if step >= SETTLE_STEPS + RAMP_STEPS:
            max_speed = torch.maximum(max_speed, speed)
            sum_speed += speed
            max_omega = torch.maximum(max_omega, omega)
            sum_omega += omega
            n_samples += 1
            # Trajectory quality: a speed trace that is still rising at the end
            # means the probe stopped too early, not that the vehicle is slow.
            # Path length vs net displacement says whether it drove straight or
            # milled around.
            speed_trace.append(float(speed[env._spawned_agent_mask.detach().cpu()].mean())
                               if int(env._spawned_agent_mask.sum()) else float(speed.mean()))
            if last_xy is not None:
                path_len += torch.norm(xy - last_xy, dim=-1)

        last_xy, z_end = xy, z

        if step % 60 == 0:
            print(f"[TractionProbe] step {step:>4}/{total}  "
                  f"mean|v|={speed.mean():.3f} m/s  mean|omega|={omega.mean():.2f} rad/s  "
                  f"mean z={z.mean():.3f} m", flush=True)

    assert start_xy is not None and z_settled is not None and last_xy is not None
    # Quartile means of the drive-window speed trace, and whether it plateaued.
    _q = max(1, len(speed_trace) // 4)
    q_means = [sum(speed_trace[i * _q:(i + 1) * _q]) / max(1, len(speed_trace[i * _q:(i + 1) * _q]))
               for i in range(4)] if speed_trace else [0.0] * 4
    # Plateaued if the last quarter is within 5% of the third quarter. Direction
    # matters: a FALLING trace is not "needs a longer window", it means the
    # vehicle slowed down (collision, leaving the slab, spinning out).
    _drift = ((q_means[3] - q_means[2]) / q_means[2]) if q_means[2] > 0 else 0.0
    plateaued = bool(abs(_drift) < 0.05)
    speed_trend = "plateau" if plateaued else ("rising" if _drift > 0 else "falling")
    mean_speed = sum_speed / max(1, n_samples)
    mean_omega = sum_omega / max(1, n_samples)
    displacement = torch.norm(last_xy - start_xy, dim=-1)
    # Slip: 1 - v / (omega*r). ~1 means wheels spin with no forward motion.
    surface_speed = mean_omega * WHEEL_RADIUS_M
    slip = torch.where(surface_speed > 0.1, 1.0 - mean_speed / surface_speed.clamp(min=1e-6),
                       torch.zeros_like(mean_speed)).clamp(0.0, 1.0)
    idle = (max_speed < IDLE_SPEED_MPS)

    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "traction_probe.csv").open("w") as f:
        f.write("env,agent,mean_speed_mps,max_speed_mps,mean_wheel_omega_radps,"
                "slip,displacement_m,z_settled_m,z_end_m,idle\n")
        for a in range(n_agent):
            for e in range(n_env):
                f.write(f"{e},{a},{mean_speed[a,e]:.4f},{max_speed[a,e]:.4f},"
                        f"{mean_omega[a,e]:.4f},{slip[a,e]:.4f},{displacement[a,e]:.4f},"
                        f"{z_settled[a,e]:.4f},{z_end[a,e]:.4f},{int(idle[a,e])}\n")

    # ---- console report ----------------------------------------------------
    print("\n=== SETTLE-WINDOW Z TRACE (zero action; spawn z = origin + spawn_height_m) ===")
    print(f"{'step':>5} {'z SPAWNED':>11} {'max':>9} {'z parked':>11}")
    for s, zm, zx, zp in z_trace:
        if s % 10 == 0 or s == SETTLE_STEPS:
            print(f"{s:>5} {zm:>11.3f} {zx:>9.3f} {zp:>11.3f}")

    print("\n=== TRACTION PROBE: per-agent-index (averaged over envs) ===")
    print(f"{'agent':>5} {'mean v':>9} {'max v':>9} {'omega':>9} {'slip':>7} "
          f"{'disp m':>9} {'z set':>8} {'z end':>8} {'idle%':>7}")
    for a in range(n_agent):
        print(f"{a:>5} {mean_speed[a].mean():>9.3f} {max_speed[a].mean():>9.3f} "
              f"{mean_omega[a].mean():>9.2f} {slip[a].mean():>7.3f} "
              f"{displacement[a].mean():>9.3f} {z_settled[a].mean():>8.3f} "
              f"{z_end[a].mean():>8.3f} {100.0*idle[a].float().mean():>7.1f}")

    # THE number that matters: parked/inactive slots are excluded from every
    # training metric via _spawned_agent_mask, and on a small ground cuboid they
    # sit off the slab edge and free-fall -- which drags any all-slot average into
    # nonsense. Report spawned-only separately and treat it as authoritative.
    sm = env._spawned_agent_mask.detach().cpu()
    rad = float(getattr(env.cfg, "max_distance_from_origin_m", 100.0))
    in_rad = sm & (max_dist_origin <= rad)
    print(f"\n>>> THE NUMBER THAT MATTERS <<<  spawned AND within the {rad:.0f} m "
          f"termination radius, i.e. the population training actually keeps:")
    if int(in_rad.sum()):
        print(f"    n={int(in_rad.sum())}/{int(sm.sum())} spawned   "
              f"idle {100*idle[in_rad].float().mean():.1f}%   "
              f"mean v {mean_speed[in_rad].mean():.3f} m/s   "
              f"slip {slip[in_rad].mean():.3f}   "
              f"z_end {z_end[in_rad].mean():.3f} m")
    else:
        print("    n=0 -- every spawned agent left the radius")
    for lbl, m in (("SPAWNED (all)", sm), ("parked (ignore)", ~sm)):
        if int(m.sum()) == 0:
            continue
        print(f"\n{lbl}: n={int(m.sum())}  idle {100*idle[m].float().mean():.1f}%  "
              f"mean v {mean_speed[m].mean():.3f} m/s  slip {slip[m].mean():.3f}  "
              f"z_end mean {z_end[m].mean():.3f} m  "
              f"frac z_end<-1m {100*(z_end[m] < -1).float().mean():.1f}%")

    a0, rest = idle[0].float().mean().item(), (idle[1:].float().mean().item() if n_agent > 1 else float("nan"))
    print(f"\nidle fraction  agent 0: {100*a0:.1f}%   agents 1..{n_agent-1}: {100*rest:.1f}%")
    print(f"overall idle   {100*idle.float().mean():.1f}%   "
          f"mean slip {slip.mean():.3f}   mean speed {mean_speed.mean():.3f} m/s")

    if mean_omega.mean() > 3.0 and mean_speed.mean() < 0.5:
        verdict = "WHEELS SPIN, BODY STILL -> zero normal force. Ground contact defect."
    elif mean_omega.mean() <= 3.0 and mean_speed.mean() < 0.5:
        verdict = "WHEELS ALSO STILL -> drive command never reached the joints. Not traction."
    elif mean_speed.mean() >= 0.5:
        verdict = "VEHICLES DRIVE -> no traction defect under this config."
    else:
        verdict = "INCONCLUSIVE."
    print(f"\nVERDICT: {verdict}\n")

    summary = {
        "n_envs": n_env, "n_agents": n_agent, "dt_s": dt,
        "settle_steps": SETTLE_STEPS, "ramp_steps": RAMP_STEPS, "drive_steps": DRIVE_STEPS,
        "ground_mode": getattr(env.cfg, "ground_mode", None),
        "ground_cuboid_size_m": getattr(env.cfg, "ground_cuboid_size_m", None),
        "ground_contact_offset_m": getattr(env.cfg, "ground_contact_offset_m", None),
        "env_spacing_m": getattr(env.cfg, "env_spacing", None),
        "wheel_friction_cap": getattr(env.cfg, "wheel_friction_cap", None),
        "scene_pool": getattr(env.cfg, "scene_factory_config_path", None),
        "mean_speed_mps": float(mean_speed.mean()),
        "mean_wheel_omega_radps": float(mean_omega.mean()),
        "mean_slip": float(slip.mean()),
        "idle_fraction_overall": float(idle.float().mean()),
        "idle_fraction_agent0": a0,
        "idle_fraction_agents_1plus": rest,
        "per_agent_mean_speed": [float(x) for x in mean_speed.mean(dim=1)],
        "per_agent_idle_fraction": [float(x) for x in idle.float().mean(dim=1)],
        "drive_steps": DRIVE_STEPS,
        "speed_quartile_means_mps": q_means,
        "speed_plateaued": plateaued,
        "speed_trend": speed_trend,
        "speed_trend_pct": round(100.0 * _drift, 1),
        "mean_path_length_m": float(path_len.mean()),
        "mean_net_displacement_m": float(displacement.mean()),
        "path_straightness": float((displacement / path_len.clamp(min=1e-6)).mean()),
        "verdict": verdict,
    }
    (run_dir / "traction_probe.json").write_text(json.dumps(summary, indent=2))

    # Human-readable sibling of the JSON. The JSON is for tooling; this is what
    # a person opens.
    q = summary["speed_quartile_means_mps"]
    lines = [
        "=" * 66,
        "  TRACTION PROBE",
        "=" * 66,
        "",
        f"  Question: does every agent actually drive, or only agent 0?",
        f"  Config:   {n_env} worlds x {n_agent} agents, {DRIVE_STEPS} drive steps "
        f"({DRIVE_STEPS * dt:.1f} s)",
        f"            ground_mode={summary['ground_mode']}  env_spacing={summary['env_spacing_m']} m",
        (f"            cuboid={summary['ground_cuboid_size_m']} m  "
         f"contact_offset={summary['ground_contact_offset_m']} m"
         if str(summary["ground_mode"]).lower() == "cuboid"
         else "            NOTE: ground_mode=plane -- cuboid size and contact_offset are "
              "NOT in play; this run does not exercise the contact_offset fix"),
        "",
        "  PER-AGENT (averaged over worlds) -- these must not diverge",
        f"    {'agent':>6} {'mean v (m/s)':>14} {'idle %':>9}",
    ]
    for a in range(n_agent):
        lines.append(f"    {a:>6} {float(mean_speed[a].mean()):>14.3f} "
                     f"{100.0 * float(idle[a].float().mean()):>9.1f}")
    spread = (max(summary["per_agent_mean_speed"]) - min(summary["per_agent_mean_speed"])) \
        if summary["per_agent_mean_speed"] else 0.0
    lines += [
        f"    spread across agents: {spread:.3f} m/s"
        f"{'  <-- agent-index dependence, investigate' if spread > 0.5 else '  (uniform)'}",
        "",
        "  SPEED TRACE over the drive window (is it still accelerating?)",
        f"    Q1 {q[0]:.2f}   Q2 {q[1]:.2f}   Q3 {q[2]:.2f}   Q4 {q[3]:.2f}  m/s",
        f"    trend Q3->Q4: {summary['speed_trend']} ({summary['speed_trend_pct']:+.1f}%)"
        + ("" if summary["speed_plateaued"] else
           ("   <-- still accelerating; raise TP_DRIVE_STEPS"
            if summary["speed_trend"] == "rising" else
            "   <-- DECELERATING; check for collisions or agents leaving the slab")),
        "",
        "  TRAJECTORY",
        f"    path length      {summary['mean_path_length_m']:.1f} m",
        f"    net displacement {summary['mean_net_displacement_m']:.1f} m",
        f"    straightness     {summary['path_straightness']:.3f}  (1.0 = straight line)",
        "",
        f"  slip {summary['mean_slip']:.3f}   "
        f"(near 1.0 = wheels spinning with no grip)",
        "",
        "-" * 66,
        f"  VERDICT: {verdict}",
        "-" * 66,
        "",
    ]
    (run_dir / "traction_probe.txt").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    print(f"[TractionProbe] wrote {run_dir/'traction_probe.csv'} and traction_probe.json", flush=True)
