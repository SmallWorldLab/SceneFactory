# GPU capacity benchmark

How many worlds and agents fit on a given GPU, and what throughput they sustain.

`scripts/benchmark_casps_sweep.py` sweeps agents-per-world levels, finds the
largest world count that completes PPO iterations at each level, and reports
CASPS (controlled agent steps per second) measured there.

```bash
PYTHONPATH=. python scripts/benchmark_casps_sweep.py --device cuda:0
```

Outputs land in `--out` (default `artifacts/casps_sweep/`):

| file | contents |
|---|---|
| `sweep_summary.txt` | the table below |
| `sweep_trials.csv` | every trial, including failures |
| `sweep_meta.json` | device, config, and the **observation settings** |
| `trial_logs/` | per-trial stdout/stderr, kept when anything fails |

---

## Results depend on the observation width — always report it

CASPS and the memory ceiling are both strong functions of the observation
vector, so a number is meaningless without the configuration that produced it.
`sweep_meta.json` records this for you. The two configurations shipped here:

| config | `road_points_k` | `neighbor_k` | obs dim |
|---|---|---|---|
| `demo_weather_physx_train.yaml` | 64 | 8 | **387** |
| `generated/scene_factory_256scene_0414_train_fastgoal_v7_sysid4_weather.yaml` | 350 | 24 | **1929** |

The paper's throughput figures use the second. Numbers measured under the first
are **not** comparable to them.

---

## Reproducing the reported sweeps

Two GPUs × two observation widths. Substitute your own `--device`.

```bash
# (a) demo observation width, 387 dims
PYTHONPATH=. python scripts/benchmark_casps_sweep.py \
  --device cuda:0 \
  --base-config configs/scene_factory/demo_weather_physx_train.yaml \
  --agents 1,2,4,8,16 \
  --out artifacts/casps_sweep_demo

# (b) paper observation width, 1929 dims
PYTHONPATH=. python scripts/benchmark_casps_sweep.py \
  --device cuda:0 \
  --base-config configs/scene_factory/generated/scene_factory_256scene_0414_train_fastgoal_v7_sysid4_weather.yaml \
  --agents 1,2,4,8,16 \
  --calib-envs 24 \
  --max-envs 4096 \
  --out artifacts/casps_sweep_paper
```

`--calib-envs 24` for (b): calibration trials must themselves fit, and the wider
observation costs several times more memory per agent slot. If any calibration
line reports `oom` or `incomplete`, halve it again — the fit needs at least 3 of
its 5 samples to succeed.

Requires Waymo scene data (README §5). Every world uses a single scene
(`--scene-json`, default `scene_000077.json`) so the measurement is not confounded
by scene diversity.

---

## How it works, and why

Trial cost is linear in world count — roughly `T(N) ≈ 11 + 0.23·N` seconds on an
RTX 4090, dominated by the serial per-world road build. A probe near the ceiling
therefore costs ~30× one at small `N`, which makes conventional
doubling-and-bisection search extremely expensive: its most frequent queries are
its least informative.

Peak VRAM, by contrast, is close to linear in world and agent count. So the
default `--mode model` **solves** for the ceiling instead of searching for it:

1. **Calibrate** — ~5 deliberately small trials spanning (worlds, agents).
2. **Fit** `VRAM(N, A) = c0 + c_w·N + c_a·N·A`, separating per-world cost (roads,
   USD prims) from per-agent cost (articulation, observation and rollout
   buffers). One fit covers every agent level.
3. **Confirm** — one run per level at `--undershoot` (0.93) of the prediction.
   Undershooting is deliberate: a pass yields the ceiling *and* the CASPS
   measurement, whereas an overshoot costs the same and yields only a failure.
4. Levels run cheapest-first (most agents → fewest worlds), refitting after each
   confirmation so the expensive low-agent levels are attempted last, against a
   model anchored by real large-`N` data.

`--mode search` restores conventional doubling plus bisection for comparison.

### What counts as a pass

A trial must complete the requested number of PPO iterations **and** emit a
`Perf/CASPS` scalar. Exit code 0 alone is not sufficient: Isaac Sim can fail
during setup and still exit cleanly, which otherwise produces a "passing" trial
that never trained and a ceiling far above the truth. Such trials are reported as
`incomplete`.

At least 2 iterations are required (`--iterations`) because RSL-RL allocates its
rollout storage and update buffers on the first learning step — a run that builds
all its worlds can still exhaust memory an iteration later.

### Reported CASPS

`Perf/CASPS` is written once per PPO iteration, already averaged over
`num_steps_per_env` steps. The first iterations absorb CUDA context creation,
kernel autotuning and Fabric settling, so the confirmation run uses
`--measure-iterations` (12) and discards `--warmup-iterations` (4) before
averaging.

---

## Safety

The sweep deliberately drives the device to out-of-memory. It refuses to start
if the target GPU already has significant memory in use; `--force` overrides.
Each trial is a subprocess with `CUDA_VISIBLE_DEVICES` pinned to `--device`, so
other GPUs are untouched.

---

## Interpreting the output

Where the scene cannot place every requested agent, the row is flagged. CASPS is
normalised by the agents that **actually spawned**
(`_scene_factory_spawn_valid`), so a flagged row must not be read as if the
requested agent count were live.

If the attainable world count halves exactly as agents per world double, capacity
is governed by total agent slots rather than world count — read the `agent slots`
column, which will be near-constant in that case.
