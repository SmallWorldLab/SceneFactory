"""
NHTSA-style braking validation for the SceneFactory PhysX articulated vehicle.

Validates that the *same* vehicle used to train the workzone (dynamics_mode=physx,
sysid params comprehensive_fwd_v1_cem_v4) produces realistic straight-line braking
distances, for the paper rebuttal (external validation vs NHTSA 100->0 km/h).

Design
------
- N worlds, 1 agent each, flat cuboid ground, steering=0.
- Per-world friction mu via friction_ruler_mode (see MU_VALUES).
- Each world throttles up to a target entry speed (v0 = 27.78 m/s = 100 km/h),
  then applies scripted FULL BRAKE (throttle=0, steer=0, brake=1.0) until stopped.
- The PhysX drive is velocity-controlled and normally clamps body speed at
  ~15 m/s; the launch config raises bicycle_max_speed_mps to 30 so the target
  entry speed is reachable. Only the drive clamp changes -- mass, brake torque,
  tires, suspension are the trained vehicle's. The ACTUAL entry speed is recorded
  per world (low-mu worlds may plateau below target) and used in every metric.
- Brake trigger is STAGGERED per world (worlds accelerate at different rates).

Output
------
run_dir/braking_validation.csv  (one row per mu) with columns:
  mu, v0_entry_mps, stopping_distance_m, stopping_time_s, mean_decel_mps2,
  peak_decel_mps2, peak_wheel_slip, theoretical_distance_m, distance_ratio
run_dir/braking_validation_report.json  (same data + run metadata)

Launch via:
    bash run_braking_validation.sh
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch


# -- Physical / experiment constants ------------------------------------------
V_TARGET_MPS   = 27.78          # 100 km/h entry speed
WHEEL_RADIUS_M = 0.35           # PHYSX_WIZARD_DEFAULT_WHEEL_RADIUS_M
G              = 9.80665        # m/s^2

# One mu per world. MUST match friction_ruler_mu_values in
# configs/scene_factory/braking_validation.yaml (same order).
MU_VALUES = [0.30, 0.40, 0.50, 0.60, 0.66, 0.70, 0.86, 1.00, 1.10]

# -- Schedule -----------------------------------------------------------------
import os as _os
SETTLE_STEPS    = 60      # zero-action warmup: drop from spawn height + settle suspension
RAMP_STEPS      = 30      # throttle ramp-up (suppress drive impulse)
MAX_DRIVE_STEPS = int(_os.environ.get("BV_MAX_DRIVE_STEPS", "3000"))  # cap to reach v0
MAX_BRAKE_STEPS = int(_os.environ.get("BV_MAX_BRAKE_STEPS", "3000"))  # cap for the stop
STOP_SPEED_MPS  = 0.15    # below this the vehicle is considered stopped
SLIP_MIN_SPEED  = 1.0     # only measure slip ratio while body speed > this (avoids blowup)
DEBUG_BRAKE     = _os.environ.get("BV_DEBUG_BRAKE", "0") == "1"  # verbose per-step brake trace
DEBUG_BRAKE_N   = int(_os.environ.get("BV_DEBUG_BRAKE_N", "40")) # steps to trace after engage


def _planar_speed(vel_b: torch.Tensor) -> torch.Tensor:
    """Planar speed (m/s) from body-frame velocity [..., 3]."""
    return torch.norm(vel_b[..., :2], dim=-1)


def _build_action_dict(env, throttle: torch.Tensor, steer: torch.Tensor,
                       brake: torch.Tensor) -> dict:
    """Build action dict from per-agent-per-env components [num_agents, num_envs]."""
    action_dict: dict[str, torch.Tensor] = {}
    for agent_idx, agent_id in enumerate(env.cfg.possible_agents):
        action_dict[agent_id] = torch.stack(
            [throttle[agent_idx], steer[agent_idx], brake[agent_idx]], dim=-1
        ).to(device=env.device, dtype=torch.float32)
    return action_dict


def _wheel_joint_ids(env) -> torch.Tensor:
    """LongTensor of the 4 wheel-joint ids for agent 0 (front/rear L/R)."""
    ids = env._wheel_joint_ids[0]
    return torch.as_tensor(ids, dtype=torch.long, device=env.device)


def run_braking_validation(env, run_dir: Path) -> None:
    """
    Run the braking validation test. The env must be configured with:
      - num_envs = len(MU_VALUES)
      - num_agents_per_env = 1
      - dynamics_mode = "physx"
      - friction_ruler_mode = True with friction_ruler_mu_values == MU_VALUES
      - bicycle_max_speed_mps >= 30 (so v0 = 27.78 m/s is reachable by throttle)
    """
    device = env.device
    N      = env.num_envs
    dt     = float(env.cfg.sim.dt) * float(env.cfg.decimation)  # s per env step

    # One mu per world, taken from the actual env config (supports N=1 for a
    # single-world diagnostic, or N=9 for the full sweep).
    mu_list = [float(x) for x in str(env.cfg.friction_ruler_mu_values).split(",")
               if x.strip()]
    assert N == len(mu_list), (
        f"braking_validation: num_envs ({N}) must equal the number of "
        f"friction_ruler_mu_values ({len(mu_list)}): {mu_list}"
    )

    vehicle    = env._vehicles[0]
    wheel_ids  = _wheel_joint_ids(env)

    # Per-world state machine: 0=settle, 1=drive, 2=brake, 3=done
    phase          = [0] * N
    brake_start_pos = [(0.0, 0.0)] * N
    last_pos        = [(0.0, 0.0)] * N
    v0_entry        = [0.0] * N          # actual speed at brake engage
    cum_dist        = [0.0] * N          # cumulative XY travel since brake engage
    stop_time       = [float("nan")] * N
    peak_decel      = [0.0] * N
    peak_slip       = [0.0] * N
    prev_speed      = [0.0] * N          # for per-step deceleration
    brake_step      = [0] * N            # step at which brake engaged
    stopped_step    = [-1] * N

    print(f"[BrakingValidation] Resetting {N} worlds (1 agent each), dt={dt:.4f}s "
          f"mu={mu_list}", flush=True)
    env.reset()

    max_total = SETTLE_STEPS + MAX_DRIVE_STEPS + MAX_BRAKE_STEPS
    print(f"[BrakingValidation] v0_target={V_TARGET_MPS:.2f} m/s, "
          f"full-brake stop; up to {max_total} steps.", flush=True)

    for step in range(max_total):
        throttle = torch.zeros(1, N, device=device)
        steer    = torch.zeros(1, N, device=device)
        brake    = torch.zeros(1, N, device=device)

        # ---- Decide commanded action from current phase --------------------
        drive_elapsed = step - SETTLE_STEPS
        for gi in range(N):
            if phase[gi] == 0:                       # settle
                if step >= SETTLE_STEPS:
                    phase[gi] = 1
            if phase[gi] == 1:                       # drive up to v0
                ramp = min(1.0, (drive_elapsed + 1) / max(1, RAMP_STEPS))
                throttle[0, gi] = 1.0 * ramp
            elif phase[gi] == 2:                     # full brake
                brake[0, gi] = 1.0

        env.step(_build_action_dict(env, throttle, steer, brake))

        if step < SETTLE_STEPS:
            continue

        # ---- Read authoritative PhysX state --------------------------------
        vel_b   = vehicle.data.root_lin_vel_b.detach()          # [N, 3]
        pos_w   = vehicle.data.root_pos_w.detach()               # [N, 3]
        jv      = vehicle.data.joint_vel.detach()[:, wheel_ids]  # [N, 4] rad/s
        speeds  = _planar_speed(vel_b).cpu()                     # [N]
        omega   = jv.abs().mean(dim=1).cpu()                     # [N] mean |wheel rate|
        pos_xy  = pos_w[:, :2].cpu()                             # [N, 2]
        if DEBUG_BRAKE:
            vel_b_c  = vel_b.cpu()                                # [N,3] body-frame lin vel
            angv_c   = vehicle.data.root_ang_vel_b.detach().cpu() # [N,3] body-frame ang vel
            posz_c   = pos_w[:, 2].cpu()                          # [N] world z
            jv_c     = jv.cpu()                                   # [N,4] wheel omega

        for gi in range(N):
            v = float(speeds[gi])

            if phase[gi] == 1:
                # Trigger brake when target reached OR drive cap hit.
                if v >= V_TARGET_MPS or drive_elapsed >= MAX_DRIVE_STEPS:
                    phase[gi]           = 2
                    v0_entry[gi]        = v
                    xy                  = (float(pos_xy[gi, 0]), float(pos_xy[gi, 1]))
                    brake_start_pos[gi] = xy
                    last_pos[gi]        = xy
                    brake_step[gi]      = step
                    prev_speed[gi]      = v

            elif phase[gi] == 2:
                xy = (float(pos_xy[gi, 0]), float(pos_xy[gi, 1]))
                cum_dist[gi] = math.hypot(xy[0] - brake_start_pos[gi][0],
                                          xy[1] - brake_start_pos[gi][1])
                last_pos[gi] = xy

                # Instantaneous deceleration (positive = slowing down).
                decel = (prev_speed[gi] - v) / dt
                if decel > peak_decel[gi]:
                    peak_decel[gi] = decel
                prev_speed[gi] = v

                # Longitudinal slip ratio (locked wheel -> 1). Only while moving.
                if v > SLIP_MIN_SPEED:
                    slip = (v - float(omega[gi]) * WHEEL_RADIUS_M) / v
                    slip = max(0.0, min(1.0, slip))
                    if slip > peak_slip[gi]:
                        peak_slip[gi] = slip

                # Brake window cap or full stop.
                brake_elapsed = step - brake_step[gi]

                if DEBUG_BRAKE and brake_elapsed <= DEBUG_BRAKE_N:
                    wheels = " ".join(f"{float(jv_c[gi, k]):+7.1f}" for k in range(4))
                    print(f"    [BRK e{gi} k={brake_elapsed:03d}] "
                          f"v={v:8.3f} vx={float(vel_b_c[gi,0]):+8.3f} "
                          f"vy={float(vel_b_c[gi,1]):+7.3f} vz={float(vel_b_c[gi,2]):+7.3f} "
                          f"z={float(posz_c[gi]):+6.3f} wz={float(angv_c[gi,2]):+6.2f} "
                          f"wheel_w=[{wheels}]", flush=True)
                if v < STOP_SPEED_MPS or brake_elapsed >= MAX_BRAKE_STEPS:
                    phase[gi]     = 3
                    stopped_step[gi] = step
                    stop_time[gi] = brake_elapsed * dt

        if (step + 1) % 200 == 0:
            ph = "".join(str(p) for p in phase)
            spd = " ".join(f"{float(speeds[i]):5.1f}" for i in range(N))
            xps = " ".join(f"{float(pos_xy[i,0]):6.0f}" for i in range(N))
            print(f"  step {step+1:5d}/{max_total}  phase=[{ph}]  v=[{spd}]  x=[{xps}]",
                  flush=True)

        if all(p == 3 for p in phase):
            print(f"[BrakingValidation] All worlds stopped at step {step+1}.",
                  flush=True)
            break

    # ---- Analyse + write ---------------------------------------------------
    rows = []
    for gi in range(N):
        mu   = mu_list[gi]
        v0   = v0_entry[gi]
        d    = cum_dist[gi]
        t    = stop_time[gi]
        mean_decel = (v0 / t) if (t and t > 0 and not math.isnan(t)) else float("nan")
        d_theory   = (v0 * v0) / (2.0 * mu * G) if mu > 0 else float("inf")
        ratio      = (d / d_theory) if d_theory > 0 else float("nan")
        rows.append({
            "mu":                    round(mu, 3),
            "v0_entry_mps":          round(v0, 3),
            "stopping_distance_m":   round(d, 3),
            "stopping_time_s":       round(t, 3) if not math.isnan(t) else None,
            "mean_decel_mps2":       round(mean_decel, 4) if not math.isnan(mean_decel) else None,
            "peak_decel_mps2":       round(peak_decel[gi], 4),
            "peak_wheel_slip":       round(peak_slip[gi], 4),
            "theoretical_distance_m": round(d_theory, 3),
            "distance_ratio":        round(ratio, 4) if not math.isnan(ratio) else None,
            "reached_target":        v0 >= V_TARGET_MPS - 0.5,
            "stopped":               stopped_step[gi] >= 0,
        })

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "braking_validation.csv"
    fields = ["mu", "stopping_distance_m", "stopping_time_s", "mean_decel_mps2",
              "peak_decel_mps2", "peak_wheel_slip", "v0_entry_mps",
              "theoretical_distance_m", "distance_ratio", "reached_target", "stopped"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    report = {
        "experiment": "nhtsa_braking_validation",
        "v0_target_mps": V_TARGET_MPS,
        "wheel_radius_m": WHEEL_RADIUS_M,
        "g": G,
        "dt_s": dt,
        "bicycle_max_speed_mps": float(env.cfg.bicycle_max_speed_mps),
        "rows": rows,
    }
    (run_dir / "braking_validation_report.json").write_text(json.dumps(report, indent=2))

    # ---- Print a table -----------------------------------------------------
    print("\n===== NHTSA Braking Validation =====", flush=True)
    hdr = (f"{'mu':>5} {'v0':>6} {'stop_d':>8} {'theory':>8} {'ratio':>6} "
           f"{'stop_t':>7} {'mean_a':>7} {'peak_a':>7} {'pk_slip':>7}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for r in rows:
        print(f"{r['mu']:>5} {r['v0_entry_mps']:>6.2f} "
              f"{r['stopping_distance_m']:>8.2f} {r['theoretical_distance_m']:>8.2f} "
              f"{(r['distance_ratio'] if r['distance_ratio'] is not None else float('nan')):>6.2f} "
              f"{(r['stopping_time_s'] if r['stopping_time_s'] is not None else float('nan')):>7.2f} "
              f"{(r['mean_decel_mps2'] if r['mean_decel_mps2'] is not None else float('nan')):>7.3f} "
              f"{r['peak_decel_mps2']:>7.3f} {r['peak_wheel_slip']:>7.3f}", flush=True)

    # Highlight the two mu the user asked to read first.
    for target_mu in (0.86, 0.66):
        for r in rows:
            if abs(r["mu"] - target_mu) < 1e-6:
                print(f"\n>>> mu={target_mu}: stop = {r['stopping_distance_m']:.2f} m "
                      f"(theory {r['theoretical_distance_m']:.2f} m, "
                      f"ratio {r['distance_ratio']}) at v0={r['v0_entry_mps']:.2f} m/s",
                      flush=True)
    print(f"\n[BrakingValidation] CSV    -> {csv_path}", flush=True)
    print(f"[BrakingValidation] Report -> {run_dir / 'braking_validation_report.json'}",
          flush=True)
