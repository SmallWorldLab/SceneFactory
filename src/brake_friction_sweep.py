"""
Braking friction sweep: consistent entry speed, mu = 1.0 -> 0.0, one car.

Runs the SAME PhysX articulated vehicle used to train the workzone through a
brake-to-stop maneuver at a CONSISTENT entry speed across a sweep of surface
friction values (one mu per world, run in parallel):

    mu = 1.0, 0.9, 0.8, ... , 0.2, 0.1   (10 worlds by default)

Entry speed is made consistent by VELOCITY INJECTION (not throttle): after a short
settle, the body root velocity is set to v0 and the wheels are pre-spun to rolling
(omega = v0 / r), then scripted FULL BRAKE is applied to a stop. Injection is the
only way to give every world the same v0 -- at mu=0 the car cannot accelerate by
throttle at all. No long drive => the car stays grounded (no sink; see
tasks/braking_validation_nhtsa.md).

Caveat: the velocity-controlled drive commands wheels->0 at throttle=0, adding
engine braking on top of the friction brake, so measured stops are SHORTER than
pure-friction theory v0^2/(2*mu*g), especially at high mu. The `theoretical_*`
and `distance_ratio` columns expose this. mu=0 never stops (censored at cap).

Outputs (in run_dir):
  brake_sweep_summary.csv / .json   one row per mu (v0, stop dist/time, decel, slip, theory)
  brake_sweep_trajectory.csv        per-step, all worlds (mu, time, speed, accel, slip, x)
  brake_sweep_distance_vs_mu.png    measured vs theoretical stopping distance (if matplotlib)

Launch:
    bash run_brake_friction_sweep.sh
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import torch


WHEEL_RADIUS_M = 0.35
G              = 9.80665

V0_MPS          = float(os.environ.get("BFS_V0_MPS", "6.0"))       # consistent entry speed
SETTLE_STEPS    = int(os.environ.get("BFS_SETTLE_STEPS", "60"))    # drop + suspension settle
INJECT_RAMP_STEPS = int(os.environ.get("BFS_INJECT_RAMP_STEPS", "25"))  # ramp v0 (avoid contact shock)
MAX_BRAKE_STEPS = int(os.environ.get("BFS_MAX_BRAKE_STEPS", "1500"))  # cap (low mu / mu=0)
STOP_SPEED_MPS  = 0.15
SLIP_MIN_SPEED  = 1.0
TRAJ_STRIDE     = int(os.environ.get("BFS_TRAJ_STRIDE", "2"))      # write every Nth step


def _planar_speed(vel_b: torch.Tensor) -> torch.Tensor:
    return torch.norm(vel_b[..., :2], dim=-1)


def _build_action_dict(env, throttle, steer, brake) -> dict:
    action_dict = {}
    for agent_idx, agent_id in enumerate(env.cfg.possible_agents):
        action_dict[agent_id] = torch.stack(
            [throttle[agent_idx], steer[agent_idx], brake[agent_idx]], dim=-1
        ).to(device=env.device, dtype=torch.float32)
    return action_dict


def run_brake_friction_sweep(env, run_dir: Path) -> None:
    """
    Env must be configured with:
      - num_envs = number of mu values, num_agents_per_env = 1
      - dynamics_mode = "physx", friction_ruler_mode = True with the mu sweep
      - invincible = True (scripted test: never auto-reset)
    """
    device = env.device
    N      = env.num_envs
    dt     = float(env.cfg.sim.dt) * float(env.cfg.decimation)

    mu_list = [float(x) for x in str(env.cfg.friction_ruler_mu_values).split(",") if x.strip()]
    assert N == len(mu_list), (
        f"brake_friction_sweep: num_envs ({N}) must equal number of mu values "
        f"({len(mu_list)}): {mu_list}"
    )

    vehicle   = env._vehicles[0]
    wheel_ids = torch.as_tensor(env._wheel_joint_ids[0], dtype=torch.long, device=device)

    traj_rows   = []                    # flat per-step records across all worlds
    braking     = [False] * N           # has brake engaged (post-injection)
    done        = [False] * N
    v0_entry    = [0.0] * N
    brake_x0    = [0.0] * N
    prev_speed  = [0.0] * N
    stop_time   = [float("nan")] * N
    stop_dist   = [float("nan")] * N
    peak_decel  = [0.0] * N
    peak_slip   = [0.0] * N
    stopped     = [False] * N

    print(f"[BrakeSweep] {N} worlds mu={mu_list} dt={dt:.4f}s "
          f"v0={V0_MPS:.2f} m/s (injected)", flush=True)
    env.reset()

    max_total = SETTLE_STEPS + MAX_BRAKE_STEPS
    for step in range(max_total):
        throttle = torch.zeros(1, N, device=device)
        steer    = torch.zeros(1, N, device=device)
        brake    = torch.zeros(1, N, device=device)

        # Bring the body up to the consistent entry speed by RAMPING the root velocity
        # over INJECT_RAMP_STEPS (no pre-spin, no brake yet). A one-step teleport to
        # highway speed shocks the contact solver and blows up certain env indices
        # (peak decel 30+ g, v0 collapse); a gradual ramp avoids the impulse. Wheels
        # stay near rest (locked-wheel skid); brake begins once the ramp completes.
        inject_end = SETTLE_STEPS + INJECT_RAMP_STEPS
        if SETTLE_STEPS <= step < inject_end:
            env_ids = torch.arange(N, device=device)
            frac = (step - SETTLE_STEPS + 1) / float(INJECT_RAMP_STEPS)
            root_vel = torch.zeros(N, 6, device=device)
            root_vel[:, 0] = V0_MPS * frac                # world +x (spawn yaw = 0)
            vehicle.write_root_velocity_to_sim(root_vel, env_ids=env_ids)

        # Brake once the ramp is done. brake_elapsed==0 (step == inject_end) captures
        # the clean injected entry state (v0 ~= V0_MPS, position = brake_x0).
        if step >= inject_end:
            for gi in range(N):
                if not done[gi]:
                    brake[0, gi] = 1.0

        env.step(_build_action_dict(env, throttle, steer, brake))
        if step < inject_end:
            continue

        vel_b  = vehicle.data.root_lin_vel_b.detach()
        pos_w  = vehicle.data.root_pos_w.detach()
        jvw    = vehicle.data.joint_vel.detach()[:, wheel_ids]
        speeds = _planar_speed(vel_b).cpu()
        omega  = jvw.abs().mean(dim=1).cpu()
        pos    = pos_w[:, :2].cpu()

        brake_elapsed = step - inject_end
        t = brake_elapsed * dt
        for gi in range(N):
            if done[gi]:
                continue
            v  = float(speeds[gi])
            om = float(omega[gi])
            x  = float(pos[gi, 0])

            if brake_elapsed == 0:               # first post-injection reading = entry speed
                v0_entry[gi] = v
                brake_x0[gi] = x

            accel = (prev_speed[gi] - v) / dt    # positive = deceleration
            if accel > peak_decel[gi]:
                peak_decel[gi] = accel
            slip = 0.0
            if v > SLIP_MIN_SPEED:
                slip = max(0.0, min(1.0, (v - om * WHEEL_RADIUS_M) / v))
                if slip > peak_slip[gi]:
                    peak_slip[gi] = slip
            prev_speed[gi] = v

            if brake_elapsed % TRAJ_STRIDE == 0:
                traj_rows.append({
                    "mu": round(mu_list[gi], 3), "time_s": round(t, 4),
                    "speed_mps": round(v, 4), "decel_mps2": round(accel, 4),
                    "wheel_slip": round(slip, 4), "x_m": round(x, 4),
                })

            if v < STOP_SPEED_MPS or brake_elapsed >= MAX_BRAKE_STEPS:
                done[gi]      = True
                stopped[gi]   = v < STOP_SPEED_MPS
                stop_time[gi] = brake_elapsed * dt
                stop_dist[gi] = abs(x - brake_x0[gi])

        if (step + 1) % 100 == 0:
            info = " ".join(f"{mu_list[i]:.1f}:{float(speeds[i]):4.1f}" for i in range(N))
            print(f"  step {step+1:4d}/{max_total}  v=[{info}]", flush=True)
        if all(done):
            print(f"[BrakeSweep] all worlds settled (stopped or capped) at step {step+1}.",
                  flush=True)
            break

    # ---- Summary -----------------------------------------------------------
    rows = []
    for gi in range(N):
        mu = mu_list[gi]
        v0 = v0_entry[gi]
        d  = stop_dist[gi]
        tt = stop_time[gi]
        mean_decel = (v0 / tt) if (tt and tt > 0 and not math.isnan(tt)) else float("nan")
        d_theory   = (v0 * v0) / (2.0 * mu * G) if mu > 0 else float("inf")
        ratio      = (d / d_theory) if (d_theory not in (0.0, float("inf")) and not math.isnan(d)) else float("nan")
        rows.append({
            "mu": round(mu, 3), "v0_entry_mps": round(v0, 3),
            "stopping_distance_m": round(d, 3) if not math.isnan(d) else None,
            "stopping_time_s": round(tt, 3) if not math.isnan(tt) else None,
            "mean_decel_mps2": round(mean_decel, 4) if not math.isnan(mean_decel) else None,
            "peak_decel_mps2": round(peak_decel[gi], 4),
            "peak_wheel_slip": round(peak_slip[gi], 4),
            "theoretical_distance_m": (round(d_theory, 3) if d_theory != float("inf") else None),
            "distance_ratio": round(ratio, 4) if not math.isnan(ratio) else None,
            "stopped": stopped[gi],
        })

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    sfields = ["mu", "v0_entry_mps", "stopping_distance_m", "stopping_time_s",
               "mean_decel_mps2", "peak_decel_mps2", "peak_wheel_slip",
               "theoretical_distance_m", "distance_ratio", "stopped"]
    with (run_dir / "brake_sweep_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader()
        w.writerows(rows)
    with (run_dir / "brake_sweep_trajectory.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mu", "time_s", "speed_mps", "decel_mps2",
                                          "wheel_slip", "x_m"])
        w.writeheader()
        w.writerows(traj_rows)
    (run_dir / "brake_sweep_summary.json").write_text(json.dumps({
        "experiment": "brake_friction_sweep",
        "dt_s": dt, "v0_mps": V0_MPS, "wheel_radius_m": WHEEL_RADIUS_M, "g": G,
        "bicycle_max_speed_mps": float(env.cfg.bicycle_max_speed_mps),
        "note": "throttle=0 adds engine braking (wheels->0); measured < theory at high mu.",
        "rows": rows,
    }, indent=2))

    # ---- Print table -------------------------------------------------------
    print("\n===== Braking Friction Sweep (v0 injected, full brake) =====", flush=True)
    hdr = (f"{'mu':>5} {'v0':>6} {'stop_d':>8} {'theory':>8} {'ratio':>6} "
           f"{'stop_t':>7} {'mean_a':>7} {'peak_a':>7} {'pk_slip':>7} {'stopped':>7}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for r in rows:
        def _g(k):
            return r[k] if r[k] is not None else float("nan")
        print(f"{r['mu']:>5} {r['v0_entry_mps']:>6.2f} {_g('stopping_distance_m'):>8.2f} "
              f"{_g('theoretical_distance_m'):>8.2f} {_g('distance_ratio'):>6.2f} "
              f"{_g('stopping_time_s'):>7.2f} {_g('mean_decel_mps2'):>7.3f} "
              f"{r['peak_decel_mps2']:>7.3f} {r['peak_wheel_slip']:>7.3f} "
              f"{str(r['stopped']):>7}", flush=True)

    _try_plot(run_dir, rows, traj_rows)
    print(f"\n[BrakeSweep] outputs -> {run_dir}", flush=True)


def _try_plot(run_dir, rows, traj_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[BrakeSweep] matplotlib unavailable, skipping plot ({e}).", flush=True)
        return

    mus       = [r["mu"] for r in rows]
    meas      = [r["stopping_distance_m"] for r in rows]
    theory    = [r["theoretical_distance_m"] for r in rows]
    stopped   = [r["stopped"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: stopping distance vs mu, measured vs theory.
    mm = [(m, d) for m, d, s in zip(mus, meas, stopped) if d is not None and s]
    cc = [(m, d) for m, d, s in zip(mus, meas, stopped) if d is not None and not s]
    if mm:
        ax1.plot([m for m, _ in mm], [d for _, d in mm], "o-", color="#1f77b4",
                 lw=1.8, label="measured (stopped)")
    if cc:
        ax1.plot([m for m, _ in cc], [d for _, d in cc], "x", color="#7f7f7f",
                 ms=9, label="censored (capped)")
    mt = [(m, d) for m, d in zip(mus, theory) if d is not None]
    if mt:
        ax1.plot([m for m, _ in mt], [d for _, d in mt], "--", color="#d62728",
                 lw=1.5, label="theory v₀²/(2μg)")
    ax1.set_xlabel("surface friction μ")
    ax1.set_ylabel("stopping distance (m)")
    ax1.set_title("Stopping distance vs μ")
    ax1.invert_xaxis()
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel 2: speed-vs-time family, colored by mu.
    cmap = plt.get_cmap("viridis")
    umus = sorted(set(r["mu"] for r in traj_rows), reverse=True)
    for mu in umus:
        pts = [(r["time_s"], r["speed_mps"]) for r in traj_rows if r["mu"] == mu]
        if not pts:
            continue
        color = cmap(1.0 - mu) if max(umus) <= 1.0 else None
        ax2.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=1.3,
                 label=f"μ={mu}")
    ax2.set_xlabel("time since brake onset (s)")
    ax2.set_ylabel("speed (m/s)")
    ax2.set_title("Braking speed vs time")
    ax2.grid(True, alpha=0.3)
    ax2.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    out = run_dir / "brake_sweep_distance_vs_mu.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[BrakeSweep] plot -> {out}", flush=True)
