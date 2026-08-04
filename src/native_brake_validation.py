"""
Native-speed brake validation: dry vs wet, one car, full trajectory.

Runs the SAME PhysX articulated vehicle used to train the workzone through an
accelerate-then-brake maneuver on two flat surfaces in parallel:

  world 0  = DRY  surface (high mu)
  world 1  = WET  surface (low  mu)

Each world: settle -> full throttle up to a native target speed (~6 m/s, the
vehicle's throttle plateau) -> scripted FULL BRAKE to a stop. The whole run stays
short so the car remains grounded (avoids the long-drive ground-penetration sink;
see tasks/braking_validation_nhtsa.md).

Why native ~6 m/s (not 100 km/h): the velocity-controlled drive plateaus the body
near 6 m/s under throttle -- the vehicle physically cannot be driven to highway
speed (see memory vehicle-top-speed-brake). This validates braking behaviour and
the dry->wet friction effect at the speed the vehicle actually reaches.

Per-world friction is set via friction_ruler_mode (static = mu, dynamic = 0.8*mu),
so DRY and WET are two different mu on two coplanar cuboid grounds.

Outputs (in run_dir):
  native_brake_dry_trajectory.csv   per-step: time, phase, x, y, speed, accel, omega, slip
  native_brake_wet_trajectory.csv   (same schema)
  native_brake_summary.csv          one row per condition (v0, stop dist/time, decel, slip)
  native_brake_summary.json         same + run metadata
  native_brake_compare.png          speed-vs-time + decel-vs-time, dry vs wet (if matplotlib)

Launch:
    bash run_native_brake_validation.sh
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import torch


# -- Physical constants -------------------------------------------------------
WHEEL_RADIUS_M = 0.35
G              = 9.80665

# -- Experiment schedule (env-var overridable) --------------------------------
TARGET_MPS      = float(os.environ.get("NBV_TARGET_MPS", "6.0"))   # native throttle plateau
SETTLE_STEPS    = int(os.environ.get("NBV_SETTLE_STEPS", "60"))    # drop + suspension settle
RAMP_STEPS      = 30                                               # throttle ramp-up
MAX_DRIVE_STEPS = int(os.environ.get("NBV_MAX_DRIVE_STEPS", "1200"))  # cap: stay grounded (< ~2000)
MAX_BRAKE_STEPS = int(os.environ.get("NBV_MAX_BRAKE_STEPS", "600"))
STOP_SPEED_MPS  = 0.15
SLIP_MIN_SPEED  = 1.0

CONDITIONS = ["dry", "wet"]  # world 0, world 1 -- order matches mu values in config


def _planar_speed(vel_b: torch.Tensor) -> torch.Tensor:
    return torch.norm(vel_b[..., :2], dim=-1)


def _build_action_dict(env, throttle, steer, brake) -> dict:
    action_dict = {}
    for agent_idx, agent_id in enumerate(env.cfg.possible_agents):
        action_dict[agent_id] = torch.stack(
            [throttle[agent_idx], steer[agent_idx], brake[agent_idx]], dim=-1
        ).to(device=env.device, dtype=torch.float32)
    return action_dict


def run_native_brake_validation(env, run_dir: Path) -> None:
    """
    Env must be configured with:
      - num_envs = 2 (world 0 = dry, world 1 = wet), num_agents_per_env = 1
      - dynamics_mode = "physx", friction_ruler_mode = True with two mu values
      - invincible = True (scripted test: never auto-reset)
    """
    device = env.device
    N      = env.num_envs
    dt     = float(env.cfg.sim.dt) * float(env.cfg.decimation)

    mu_list = [float(x) for x in str(env.cfg.friction_ruler_mu_values).split(",") if x.strip()]
    assert N == len(mu_list) == len(CONDITIONS), (
        f"native_brake_validation expects {len(CONDITIONS)} worlds "
        f"({CONDITIONS}); got num_envs={N}, mu={mu_list}"
    )

    vehicle   = env._vehicles[0]
    wheel_ids = torch.as_tensor(env._wheel_joint_ids[0], dtype=torch.long, device=device)

    # Per-world trajectory: list of per-step records.
    traj = [[] for _ in range(N)]           # each: dict(time, phase, x, y, speed, accel, omega, slip)
    phase        = [0] * N                    # 0 settle, 1 accel, 2 brake, 3 done
    brake_step   = [0] * N
    brake_x0     = [0.0] * N
    v0_entry     = [0.0] * N
    top_speed    = [0.0] * N
    prev_speed   = [0.0] * N
    stop_time    = [float("nan")] * N
    stop_dist    = [float("nan")] * N
    peak_decel   = [0.0] * N
    peak_slip    = [0.0] * N
    stopped      = [False] * N

    print(f"[NativeBrake] {N} worlds {CONDITIONS} mu={mu_list} dt={dt:.4f}s "
          f"target={TARGET_MPS:.1f} m/s", flush=True)
    env.reset()

    max_total = SETTLE_STEPS + MAX_DRIVE_STEPS + MAX_BRAKE_STEPS
    for step in range(max_total):
        throttle = torch.zeros(1, N, device=device)
        steer    = torch.zeros(1, N, device=device)
        brake    = torch.zeros(1, N, device=device)

        drive_elapsed = step - SETTLE_STEPS
        for gi in range(N):
            if phase[gi] == 0 and step >= SETTLE_STEPS:
                phase[gi] = 1
            if phase[gi] == 1:
                ramp = min(1.0, (drive_elapsed + 1) / max(1, RAMP_STEPS))
                throttle[0, gi] = 1.0 * ramp
            elif phase[gi] == 2:
                brake[0, gi] = 1.0

        env.step(_build_action_dict(env, throttle, steer, brake))
        if step < SETTLE_STEPS:
            continue

        vel_b  = vehicle.data.root_lin_vel_b.detach()
        pos_w  = vehicle.data.root_pos_w.detach()
        jv     = vehicle.data.joint_vel.detach()[:, wheel_ids]
        speeds = _planar_speed(vel_b).cpu()
        omega  = jv.abs().mean(dim=1).cpu()
        pos    = pos_w[:, :2].cpu()

        t = (step - SETTLE_STEPS) * dt
        for gi in range(N):
            if phase[gi] == 3:
                continue
            v  = float(speeds[gi])
            om = float(omega[gi])
            x  = float(pos[gi, 0])
            y  = float(pos[gi, 1])
            top_speed[gi] = max(top_speed[gi], v)

            accel = (v - prev_speed[gi]) / dt       # signed: + accelerating, - braking
            slip = 0.0
            if v > SLIP_MIN_SPEED:
                slip = max(0.0, min(1.0, (v - om * WHEEL_RADIUS_M) / v))

            traj[gi].append({
                "time_s": round(t, 4), "phase": phase[gi],
                "x_m": round(x, 4), "y_m": round(y, 4),
                "speed_mps": round(v, 4), "accel_mps2": round(accel, 4),
                "wheel_omega_radps": round(om, 4), "wheel_slip": round(slip, 4),
            })

            if phase[gi] == 1:
                if v >= TARGET_MPS or drive_elapsed >= MAX_DRIVE_STEPS:
                    phase[gi]     = 2
                    v0_entry[gi]  = v
                    brake_step[gi] = step
                    brake_x0[gi]  = x
                    prev_speed[gi] = v
            elif phase[gi] == 2:
                if accel < 0 and -accel > peak_decel[gi]:
                    peak_decel[gi] = -accel
                if slip > peak_slip[gi]:
                    peak_slip[gi] = slip
                be = step - brake_step[gi]
                if v < STOP_SPEED_MPS or be >= MAX_BRAKE_STEPS:
                    phase[gi]     = 3
                    stopped[gi]   = v < STOP_SPEED_MPS
                    stop_time[gi] = be * dt
                    stop_dist[gi] = abs(x - brake_x0[gi])
            prev_speed[gi] = v

        if (step + 1) % 100 == 0:
            info = "  ".join(
                f"{CONDITIONS[i]}:v={float(speeds[i]):4.1f} ph{phase[i]}" for i in range(N))
            print(f"  step {step+1:4d}/{max_total}  {info}", flush=True)
        if all(p == 3 for p in phase):
            print(f"[NativeBrake] both worlds stopped at step {step+1}.", flush=True)
            break

    # ---- Write per-condition trajectory CSVs -------------------------------
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_fields = ["time_s", "phase", "x_m", "y_m", "speed_mps",
                   "accel_mps2", "wheel_omega_radps", "wheel_slip"]
    for gi, cond in enumerate(CONDITIONS):
        p = run_dir / f"native_brake_{cond}_trajectory.csv"
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=traj_fields)
            w.writeheader()
            w.writerows(traj[gi])

    # ---- Summary -----------------------------------------------------------
    summary_rows = []
    for gi, cond in enumerate(CONDITIONS):
        mu = mu_list[gi]
        v0 = v0_entry[gi]
        d  = stop_dist[gi]
        tt = stop_time[gi]
        mean_decel = (v0 / tt) if (tt and tt > 0 and not math.isnan(tt)) else float("nan")
        # accel time to reach v0 (drive phase duration up to brake engage)
        accel_time = next((r["time_s"] for r in traj[gi] if r["phase"] == 2), float("nan"))
        summary_rows.append({
            "condition": cond, "mu": round(mu, 3),
            "v0_entry_mps": round(v0, 3), "top_speed_mps": round(top_speed[gi], 3),
            "accel_time_to_v0_s": round(accel_time, 3) if not math.isnan(accel_time) else None,
            "stopping_distance_m": round(d, 3) if not math.isnan(d) else None,
            "stopping_time_s": round(tt, 3) if not math.isnan(tt) else None,
            "mean_decel_mps2": round(mean_decel, 4) if not math.isnan(mean_decel) else None,
            "peak_decel_mps2": round(peak_decel[gi], 4),
            "peak_wheel_slip": round(peak_slip[gi], 4),
            "stopped": stopped[gi],
        })

    sfields = ["condition", "mu", "v0_entry_mps", "top_speed_mps", "accel_time_to_v0_s",
               "stopping_distance_m", "stopping_time_s", "mean_decel_mps2",
               "peak_decel_mps2", "peak_wheel_slip", "stopped"]
    with (run_dir / "native_brake_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader()
        w.writerows(summary_rows)
    (run_dir / "native_brake_summary.json").write_text(json.dumps({
        "experiment": "native_brake_validation_dry_vs_wet",
        "dt_s": dt, "target_mps": TARGET_MPS, "wheel_radius_m": WHEEL_RADIUS_M,
        "bicycle_max_speed_mps": float(env.cfg.bicycle_max_speed_mps),
        "rows": summary_rows,
    }, indent=2))

    # ---- Print table -------------------------------------------------------
    print("\n===== Native-speed Brake Validation (dry vs wet) =====", flush=True)
    hdr = (f"{'cond':>5} {'mu':>5} {'v0':>6} {'stop_d':>8} {'stop_t':>7} "
           f"{'mean_a':>7} {'peak_a':>7} {'pk_slip':>7}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for r in summary_rows:
        def _g(k):
            return r[k] if r[k] is not None else float("nan")
        print(f"{r['condition']:>5} {r['mu']:>5} {r['v0_entry_mps']:>6.2f} "
              f"{_g('stopping_distance_m'):>8.2f} {_g('stopping_time_s'):>7.2f} "
              f"{_g('mean_decel_mps2'):>7.3f} {r['peak_decel_mps2']:>7.3f} "
              f"{r['peak_wheel_slip']:>7.3f}", flush=True)

    _try_plot(run_dir, traj, summary_rows, dt)
    print(f"\n[NativeBrake] trajectories + summary -> {run_dir}", flush=True)


def _try_plot(run_dir, traj, summary_rows, dt):
    """Speed-vs-time and decel-vs-time comparison; skips silently if no matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[NativeBrake] matplotlib unavailable, skipping plot ({e}).", flush=True)
        return

    colors = {"dry": "#1f77b4", "wet": "#d62728"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=False)

    for gi, cond in enumerate(CONDITIONS):
        rows = traj[gi]
        tt = [r["time_s"] for r in rows]
        sp = [r["speed_mps"] for r in rows]
        c  = colors.get(cond, None)
        mu = summary_rows[gi]["mu"]
        ax1.plot(tt, sp, color=c, lw=1.8, label=f"{cond} (μ={mu})")
        # mark brake onset
        b = next((r["time_s"] for r in rows if r["phase"] == 2), None)
        if b is not None:
            ax1.axvline(b, color=c, ls=":", lw=1.0, alpha=0.6)

    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("speed (m/s)")
    ax1.set_title("Accelerate → brake: speed vs time (same vehicle)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Deceleration during brake phase (time since brake onset)
    for gi, cond in enumerate(CONDITIONS):
        rows = [r for r in traj[gi] if r["phase"] == 2]
        if not rows:
            continue
        t0 = rows[0]["time_s"]
        tb = [r["time_s"] - t0 for r in rows]
        dec = [-r["accel_mps2"] for r in rows]  # positive = deceleration
        c  = colors.get(cond, None)
        ax2.plot(tb, dec, color=c, lw=1.5, label=f"{cond}")
    ax2.axhline(G, color="gray", ls="--", lw=0.8, alpha=0.6, label="1 g")
    ax2.set_xlabel("time since brake onset (s)")
    ax2.set_ylabel("deceleration (m/s²)")
    ax2.set_title("Braking deceleration vs time")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    out = run_dir / "native_brake_compare.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[NativeBrake] plot -> {out}", flush=True)
