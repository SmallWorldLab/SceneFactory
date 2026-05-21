# WorkZone — Project Tracking

> Branch: `workzone`  
> Base: SceneFactory (GPU-vectorized multi-agent driving sim, Isaac Sim + IsaacLab)  
> Status: Early scaffolding — this document is the source of truth for what needs to be done.

---

## Why This Is a Natural Extension

SceneFactory already provides:
- GPU-batched PhysX vehicle dynamics (10-DOF, sysid-calibrated)
- Real Waymo road geometry as USD worlds
- Multi-agent RL environment (goal-reaching, 16 agents × 256 worlds)
- Per-world weather/friction randomization (ALL model)
- A lane-center sampler for start/goal pair generation
- A config wizard for curating scene pools

A **workzone** (construction zone) is the next logical scene type:
- It is a *perturbation* of existing road geometry, not a new simulation stack
- It demands the multi-agent framework (merge, yield, contraflow)
- It adds new surface/friction types (gravel, steel plate, fresh asphalt) the friction API can absorb
- It is a safety-critical AV edge case targeted by NHTSA and AV OEMs
- The reward shaping layer already exists — workzone just needs new behavioral terms

---

## Work Items

---

### ⚠️ Phase 0 — Road Representation (Prerequisite for Everything)

> **Ground physics (post ground-plane-isolation change):** vehicles drive on a per-env
> 1000×1000×1m kinematic cuboid (`ground_mode="cuboid"`), cloned into each env by Isaac Lab.
> This is confirmed working — scene creation succeeds, Isaac Lab cloner copies it correctly.
>
> **Road physics are NOT real.** The "road" is purely decorative:
> - `seg_width=0.10m`, `seg_height=0.01m` USD cuboids floating 2cm above the ground (`z_lift=0.02`)
> - `enable_segment_collision=False` hardcoded as the default in every call site in `chocolate_waymo_builder.py`
> - The collision API *infrastructure* exists in the builder (the parameter is there) but was
>   deliberately left off — almost certainly because tiny 10cm×1cm boxes sitting on top of
>   a ground plane would create conflicting contacts/normal directions, and the flat ground is
>   the only real driving surface
> - Waymo Z data exists but is discarded (`flatten_road_z: true` by default)
>
> **Lane-edge penalties are software reward terms, not physical boundaries.**
> **Friction is a single scalar applied globally to the ground cuboid per world — not surface-specific.**
>
> **Status of physics proof:** Ground cuboid ✅ confirmed. Road surface ❌ never enabled.
> **None of the workzone items are physically meaningful until this layer is real.**

- [ ] **P0-01** — Road surface mesh from Waymo polylines  
  Triangulate the lane polygon (left edge + right edge polylines per lane) into a proper road mesh.
  Write it as a `UsdGeom.Mesh` with a `UsdPhysics.CollisionAPI` and a bound `UsdPhysics.MaterialAPI`.
  This replaces the decorative cuboid strip segments in `chocolate_waymo_builder.py`.
  The vehicle should physically drive on the road mesh, not on an infinite flat floor below it.

- [ ] **P0-02** — Road-boundary collision geometry  
  Convert lane edge polylines into thin collision walls (curbs / raised kerbs) as USD prims with
  `UsdPhysics.CollisionAPI`. This makes lane departures a physical event, not just a reward penalty.
  The `choco_road_edge_ttc_penalty` reward term becomes a fallback, not the only guard.

- [ ] **P0-03** — Per-surface PhysX material binding to road mesh  
  Remove the global ground-plane friction scalar. Bind a `UsdPhysics.MaterialAPI` (static/dynamic
  friction, restitution) directly to each road mesh face group or road segment prim.
  This is the prerequisite for workzone surface types (gravel, steel plate, fresh asphalt)
  to have any physical effect.

- [ ] **P0-04** — Elevation support  
  Stop discarding Waymo Z. The scene JSONs carry real elevation data per polyline point.
  Set `flatten_road_z: false` as the new default and verify the road mesh and vehicle spawn
  heights track the actual terrain profile. The PhysX ground plane must either be removed or
  lowered far enough that the road mesh is the contact surface everywhere.

- [ ] **P0-05** — Ground plane removal / per-world floor strategy  
  With a real road mesh providing collision, the infinite shared ground plane is either
  redundant or actively harmful (vehicles that leave the road fall through nothing, or worse,
  the ground plane and road mesh create conflicting contact normals). Decide: keep the ground
  plane far below as a fallback, or replace with per-world bounded floor tiles that only cover
  the drivable area.

- [ ] **P0-06** — Spawn height and vehicle-road contact validation  
  Current spawn height is a config scalar (`spawn_height_m`). With road mesh elevation, spawn
  Z must be sampled from the mesh at the spawn XY, not hardcoded. Validate that the vehicle
  wheel contact points settle onto the road surface correctly within the first physics step.

- [ ] **P0-07** — Performance benchmark after road mesh introduction  
  Decorative cuboid strips are cheap. A real triangulated road mesh with collision detection is
  not. Benchmark frame time and CASPS (currently 19,250 at 256 worlds × 16 agents) after P0-01
  through P0-05. Identify whether broad-phase collision filtering or mesh simplification is
  needed to stay within throughput budget.

- [ ] **P0-08** — Road representation is a static preloaded lookup table, not perception  
  The policy does not "perceive" the road. At scene load, Waymo polyline points are stored as
  raw float arrays in USD custom metadata on a prim. At env startup, `_initialize_lane_touch_metadata()`
  reads them into a GPU tensor `_lane_touch_points_xy_m [num_envs × N_points × 2]`.
  Every step, road "observation" is a brute-force k-NN distance query against that static tensor.
  The reward edge penalty uses the same tensor.
  **This is a privileged map-based lookup, not a sensor.**
  For workzone, this is fatal: closing a lane or placing a barrel does not update the tensor —
  the map still says all lanes are open.
  Required: a runtime map-update mechanism that can mark lane segments as closed, narrow, or
  reclassified, and propagate that into `_lane_touch_points_xy_m` and the reward kernels
  before the episode begins.

---

### 🏗️ Scene Construction (Phase 1 — Workzone)
- [ ] **WZ-01** — Define a workzone scene JSON schema extension  
  Add `workzone` block to scene JSON: `taper_start_m`, `taper_length_m`, `closed_lanes: [int]`, `speed_limit_kmh`, `surface_type` (asphalt / gravel / steel_plate / fresh_asphalt), optional `worker_positions`.
  
- [ ] **WZ-02** — Workzone USD geometry builder  
  Extend `chocolate_waymo_builder.py` (or add `workzone_builder.py`) to instantiate:
  - Traffic barrel arrays along taper and buffer zones
  - Jersey barrier USD prims for positive protection
  - Speed limit sign assets at taper entry
  - Optional worker / flagger prim placements

- [ ] **WZ-03** — Procedural taper generator  
  Given a scene road centerline and `closed_lanes`, compute cone/barrel placement positions following MUTCD taper length formula: $L = WS^2/60$ (urban) or $L = WS$ (highway, $S \geq 45$ mph). Output as USD Xform array instanced into the scene.

- [ ] **WZ-04** — Workzone scene pool curation  
  Select Waymo scenes whose road topology supports a believable lane closure (multi-lane roads, merge points). Use `scene_factory_config_wizard.py` as the curation entry point. Target: 64 train + 32 test workzone scenes.

---

### 🗺️ Lane Sampler
- [ ] **WZ-05** — Workzone-aware lane center sampler  
  Extend `src/trfc/lane_center_sampler.py` to accept a `closed_lane_ids: set[int]` mask. Start/goal pairs must not originate or terminate in closed lanes. Add a `require_lane_change=True` mode that forces at least one lane transition (the merge maneuver).

- [ ] **WZ-06** — Spawn exclusion zones  
  Add `exclusion_zones: list[AABB]` to the spawn sampler so agents cannot be placed inside barrel/barrier footprints. Populated from WZ-02 geometry at scene-load time.

---

### ⚙️ Physics & Friction
- [ ] **WZ-07** — Workzone surface friction codes  
  Extend `src/trfc/friction_api.py` with new `SurfaceType` entries:
  - `GRAVEL`: μ ≈ 0.4–0.5 (dry), degrades sharply when wet
  - `STEEL_PLATE`: μ ≈ 0.3–0.4 dry, ≈ 0.15 wet
  - `FRESH_ASPHALT`: μ ≈ 0.7–0.8 (higher than aged)
  Map these through the existing per-world PhysX friction tensor pipeline.

- [ ] **WZ-08** — Friction zone geometry  
  Support per-zone friction overrides within a single world (not just per-world uniform friction). Store as a list of `(polygon, friction_coeff)` pairs; sample per-agent based on current XY position each physics step.

---

### 🎮 Reward & Observation
- [ ] **WZ-09** — Workzone reward terms  
  Add to `student_vehicle_multiagent_goal_env.py`:
  - `r_speed_compliance`: penalize speed > workzone limit
  - `r_lateral_clearance`: penalize proximity to barriers (not just agent-agent TTC)
  - `r_merge_yield`: reward yielding to the vehicle with right-of-way in merge taper
  - `r_worker_proximity`: heavy penalty within exclusion radius of worker prims

- [ ] **WZ-10** — Workzone observation tokens  
  Extend `scene_factory_obs_contract.py` and the observation vector:
  - `workzone_active` flag (0/1)
  - `speed_limit_normalized` scalar
  - `nearest_barrier_dist_m` (like TTC but lateral, to static obstacles)
  - `merge_priority` token (am I the yielding or proceeding agent?)

---

### 🏋️ Training
- [ ] **WZ-11** — Workzone curriculum  
  Implement a curriculum that starts agents in open-road SceneFactory worlds and gradually increases workzone world fraction (e.g., 0% → 25% → 50% → 100% over training). Hook into the existing `scene_factory_world_selection_mode` logic.

- [ ] **WZ-12** — Demo training config  
  Create `configs/scene_factory/demo_workzone_train.yaml`: 32 worlds, 4 agents, 4 workzone scenes, dry asphalt, no weather randomization. Mirrors `demo_weather_physx_train.yaml` in scale.

- [ ] **WZ-13** — `run_workzone_train.sh`  
  Entry-point shell script analogous to `run_demo_train.sh`.

---

### 📊 Evaluation
- [ ] **WZ-14** — Workzone-specific eval metrics  
  Add to eval output:
  - `barrel_contact_rate` (hit a cone/barrel)
  - `speed_limit_violation_rate` (exceeded posted limit)
  - `merge_success_rate` (completed merge without collision)
  - `worker_near_miss_rate` (entered worker exclusion zone)

- [ ] **WZ-15** — Transfer eval: open-road policy → workzone  
  Evaluate the existing `v8_no_weather_iter300.pt` checkpoint (trained on open roads) zero-shot in workzone worlds. Establishes baseline degradation and motivates workzone-specific training.

- [ ] **WZ-16** — `run_workzone_eval.sh` and eval config generation  
  Mirror `run_eval_pretrained.sh` structure; add to `scripts/generate_weather_eval_configs.py` or create `scripts/generate_workzone_eval_configs.py`.

---

### 🔧 Infrastructure / Polish
- [ ] **WZ-17** — USD asset acquisition / creation  
  Source or create USD assets for: traffic barrel (cone), jersey barrier, speed limit sign, flagger worker. Add to `artifacts/workzone_assets/`.

- [ ] **WZ-18** — `convert_waymo_tfrecord_to_json.py` workzone annotation pass  
  After base extraction, run a second pass that annotates scenes with `road_lane_count`, `has_merge_point`, `max_road_width_m`. Used by WZ-04 curation filter.

- [ ] **WZ-19** — Workzone visualizer  
  Extend `run_visualize_scene.sh` / `physx_teacher_rollout_visualizer.py` to overlay workzone geometry (barrel positions, closed-lane shading, speed limit zones) in the Isaac Sim GUI.

- [ ] **WZ-20** — README update  
  Add a `## WorkZone Extension` section to `README.md` documenting the new scene schema, new configs, and new eval scripts once the above items stabilize.

---

---

### � Phase -2 — SF Format (Foundation for Everything)

> Before any converter, terrain integration, or workzone feature can be built,
> the canonical interchange format must be defined and adopted. See **[SF_FORMAT.md](SF_FORMAT.md)**
> for the full specification.

- [ ] **SF-01** — SF Format spec finalized  
  Review and ratify [SF_FORMAT.md](SF_FORMAT.md). Agree on type codes, required vs optional fields,
  and versioning policy. This document becomes the contract — no downstream code changes
  without a format version bump.

- [ ] **SF-02** — `scripts/migrate_waymo_json_to_sf.py`  
  Upgrade existing `scene_XXXXXX.json` → `scene_XXXXXX.sf.json`. Safe defaults:
  `sf_version=1.0`, `coordinate_frame.type=waymo_world`, `terrain.elevation_grid=null`,
  `surfaces=[]`, `workzone=null`, `passable=true` per polyline, `road_surface=asphalt`.
  Run non-destructively (keeps old files). Update all YAML configs to point at `.sf.json`.

- [ ] **SF-03** — SF Format reader/validator in Python  
  `src/sf_format.py` — dataclass-based loader and schema validator.
  Replaces the scattered `_load_scene_cfg()` / `_load_yaml()` calls that read raw dicts.
  Validates `sf_version` compatibility, required fields, array shape consistency.
  Used by `lane_center_sampler.py`, `chocolate_waymo_builder.py`, reward kernels.

- [ ] **SF-04** — `convert_waymo_to_sf.py` (rename + upgrade of existing converter)  
  Rename `scripts/convert_waymo_tfrecord_to_json.py` → `scripts/convert_waymo_to_sf.py`.
  Emit full SF Format: preserve Z from Waymo LiDAR (stop discarding elevation),
  populate `type_name`, `passable`, `road_surface`, `half_width_m` per polyline,
  set `coordinate_frame.type=waymo_world`.

---

### �🗺️ Phase -1 — Scene Source Liberation (OSM + Real Terrain)

> SceneFactory is currently **locked to Waymo Open Motion Dataset** — a restricted-license,
> ~1,000-clip dataset requiring a signed data agreement. The `IsaacLab/` submodule directory
> is an empty placeholder (no `.gitmodules` registered, never initialized).
> The goal here is to break both locks simultaneously: replace Waymo with OpenStreetMap
> as the unlimited, freely-licensed road source, and route terrain geometry through Isaac Lab's
> `TerrainImporter` so the ground is physically real.
>
> **The scene JSON schema is the abstraction boundary.** Everything downstream
> (`lane_center_sampler.py`, obs contract, reward, policy) reads only from the JSON and the
> USD stage — it does not care how the JSON was produced. A drop-in OSM converter
> that emits the same schema makes Waymo entirely optional.

- [ ] **OSM-01** — Register IsaacLab as a proper git submodule  
  `git submodule add https://github.com/isaac-sim/IsaacLab.git IsaacLab`  
  Pin to the version in the README (`v0.54.3`). Add `git submodule update --init` to
  `quickstart.sh`. The empty `IsaacLab/` folder should be removed and replaced by the submodule.

- [ ] **OSM-02** — `scripts/convert_osm_to_scene_json.py`  
  New script mirroring `convert_waymo_tfrecord_to_json.py`. Uses `osmnx` to:
  - Accept a lat/lon bounding box or place name
  - Query OSM road graph (`drive` network), extract lane polylines per way
  - Assign lane types using OSM highway tags → SceneFactory road type integers
  - Fetch SRTM/Copernicus DEM elevation for the bbox and sample Z per polyline point
  - Write `scene_XXXXXX.json` in the existing schema (`road.polylines`, `road.edges`, etc.)
  Output is a drop-in replacement for Waymo JSONs — no downstream changes needed.

- [ ] **OSM-03** — DEM elevation integration  
  Use `elevation` library or direct SRTM tile fetch to assign real Z coordinates to OSM
  polyline points. This feeds directly into P0-04 (elevation support) and means the road mesh
  will have real terrain profile, not a flat floor.

- [ ] **OSM-04** — Isaac Lab `TerrainImporter` integration  
  In `scene_factory_multiworld_scene.py`, after `chocolate_waymo_builder.py` produces the
  per-world USD, register it via `TerrainImporterCfg(terrain_type="usd", usd_path=...)`.
  This gives: real PhysX collision on road mesh, automatic env-origin placement on terrain
  surface, and collision group isolation between worlds.
  Replace the current `GroundPlaneCfg` / cuboid floor with terrain-managed ground.

- [ ] **OSM-05** — Scene JSON schema forward-compatibility audit  
  Audit whether OSM-sourced JSONs need any new fields vs. the Waymo-sourced schema.
  Candidates: `source: "osm" | "waymo"`, `bbox_latlon`, `osm_way_ids`, `dem_resolution_m`.
  Add these as optional fields so the schema stays backward-compatible.

- [ ] **OSM-06** — Lane-level detail fallback strategy  
  OSM highway tags give road width and lane count inconsistently. Define a fallback:
  if `lanes` tag is absent, infer from `highway` type (residential=1, secondary=2,
  primary/trunk=3). This produces usable (if approximate) lane geometry for the sampler.

- [ ] **OSM-07** — Workzone tag passthrough  
  OSM has `highway=construction`, `construction=*`, and `access=no` tags on ways under
  construction. Parse these in the converter and emit a `workzone_hint: true` field in the
  scene JSON. This seeds WZ-04 scene pool curation for free from real-world data.

---

## Open Questions

1. **Worker/flagger agents** — are they static USD prims or articulated agents with their own policy? Static is tractable now; dynamic workers are a future milestone.
2. **Contraflow scenes** — MUTCD allows two-way traffic in one open lane. Does the current multi-agent reward handle head-on approach correctly? Needs investigation in `r_merge_yield`.
3. **Waymo scene sufficiency** — how many of the ~200 processed Waymo scenes have multi-lane roads suitable for WZ-04? Run a quick lane-count audit before committing to a 64-scene target.
4. **Per-zone friction (WZ-08)** — per-agent position lookup each physics step adds overhead. Benchmark against the 19,250 CASPS baseline to ensure throughput stays acceptable.
5. **Curriculum pacing (WZ-11)** — should curriculum be time-based (iteration count) or performance-based (success rate threshold)? Performance-based is more principled but adds bookkeeping.

---

## Milestone Plan

| Milestone | Items | Goal |
|-----------|-------|------|
| **M-2 — SF Format** | SF-01 → SF-04 | Canonical 3D scene format defined, existing Waymo JSONs migrated |
| **M-1 — Scene Source Liberation** | OSM-01 → OSM-07 | OSM replaces Waymo; IsaacLab submodule live; real terrain physics |
| **M0 — Real Road Surface** | P0-01 → P0-08 | Vehicle physically drives on road mesh; friction is surface-bound; elevation is real |
| **M1 — Foundation** | WZ-01, WZ-17, WZ-18 | Workzone scene schema defined, assets acquired, Waymo/OSM scenes audited |
| **M2 — Scene in Sim** | WZ-02, WZ-03, WZ-04, WZ-06 | Can visualize a workzone world in Isaac Sim |
| **M3 — Agent Can Navigate** | WZ-05, WZ-07, WZ-09, WZ-10, WZ-12, WZ-13 | Demo training runs end-to-end |
| **M4 — Evaluation** | WZ-14, WZ-15, WZ-16 | Transfer eval + workzone-trained policy compared |
| **M5 — Polish** | WZ-08, WZ-11, WZ-19, WZ-20 | Curriculum, per-zone friction, visualizer, docs |
