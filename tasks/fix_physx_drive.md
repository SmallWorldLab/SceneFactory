# Task: Fix PhysX Vehicle Drive Dynamics

## Problem
The PhysX vehicle only reaches ~4.7 m/s at full throttle (target: 7–10 m/s). The suspension springs are too weak (1080 N/m sysid value) — the vehicle slams against its lower suspension limit under its own weight, creating large constraint forces that amplify joint reaction forces and cancel ~86% of wheel traction.

## Relevant files
- `src/student_vehicle_goal_env.py` lines ~96–143: `build_student_vehicle_articulation_cfg()` — ImplicitActuatorCfg for wheels (damping=50, effort_limit=1e8)
- `src/student_vehicle_multiagent_goal_env.py` lines ~1082–1115: `_setup_scene()` — writes joint viscous frictions; line ~2612–2700: `_apply_action()` — velocity-controlled wheel drive
- `src/student_vehicle_sysid.py`: `StudentTunableConfig` — contains `suspension_stiffness_n_m = 1080.8`
- `artifacts/student_vehicle_sysid/comprehensive_fwd_v1_cem_v4/best_config.json`: loaded sysid params
- `src/physx_teacher_patch_track.py`: how `_apply_runtime_student_dynamics` uses sysid params

## What NOT to touch
- Steering PD control (kp/kd) — works correctly
- Braking joint effort — works correctly, friction-dependent stopping confirmed
- Per-world friction pipeline (`_apply_per_env_tire_friction`) — untouched
- Reward/observation systems

## Acceptance criterion
```bash
bash run_physics_validation.sh
```
Phase 1 PASS: full throttle peak > 5 m/s, z_range < 0.20 m, monotonic  
Phase 2 PASS: steering symmetric, turning radii reasonable (no change needed)  
Phase 3: peak speed INCREASES with μ (not decreases as currently), stop_dist DECREASES with μ

## What's already been tried — do NOT repeat these
1. **Direct joint torque** (`set_joint_effort_target`): FWD spin-up asymmetry — left wheel over-spins, right wheel under-spins, forces cancel.
2. **External body force** on chassis: cancelled by suspension→wheel→contact reaction chain. Drive force flows chassis→suspension→wheel→ground and back.
3. **Velocity-controlled wheels** (current state, D=50): provides ~4.7 m/s but z_range=0.56 m at full throttle. Equilibrium is still dominated by articulation reaction forces (~3500 Ns/m effective drag).
4. **Zeroing all Coulomb frictions** (suspension, steer, wheel): made speed WORSE (2.9 m/s) because the suspension Coulomb friction was accidentally stiffening the suspension and improving force transmission.

## Most promising untried approach
**Fix the suspension spring strength.** The sysid value `suspension_stiffness_n_m = 1080.8 N/m` is far too weak — at vehicle mass 1920 kg, 4 wheels each carrying 4709 N, the spring can only provide 1080 × 0.175 (max travel) = 189 N. The vehicle slams to the lower limit stop immediately. This creates large impulsive constraint forces at each bounce that propagate as reaction forces.

For proper suspension at rest: K = 4709 N / 0.05 m compression = **~94,000 N/m per wheel**.

Try: in `_setup_scene()` after `_apply_runtime_student_dynamics()`, add:
```python
# Override suspension spring to properly support vehicle weight
vehicle.write_joint_stiffness_to_sim(
    torch.full((self.num_envs, len(suspension_joint_ids)), 94000.0, device=self.device),
    joint_ids=suspension_joint_ids,
)
```

Also try setting suspension target position to the equilibrium compression:
```python
vehicle.write_joint_position_to_sim(
    torch.full((self.num_envs, len(suspension_joint_ids)), -0.05, device=self.device),
    joint_ids=suspension_joint_ids,
)
```

If that alone doesn't reach 5 m/s, also try combining with the existing D=50 velocity controller but increasing the max speed (currently `bicycle_max_speed_mps = 15 m/s`, which gives target_omega = 42.9 rad/s — try keeping this but also check if the z_range improves with proper suspension).

## Expected outcome
With correct suspension spring: z_range should drop below 0.2 m at full throttle, vehicle body stays grounded, and the articulation transmits wheel traction to chassis more efficiently.
