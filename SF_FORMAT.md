# SceneFactory Scene Format (SF Format)

**Version:** 1.0  
**File extension:** `.sf.json`  
**Encoding:** UTF-8 JSON

---

## Purpose

The SF Format is the **single canonical 3D scene representation** for SceneFactory. It is the
contract between data sources (Waymo, OpenStreetMap, synthetic generators, hand-authored scenes)
and the simulation stack (USD builder, lane sampler, observation encoder, reward kernels).

```
┌─────────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│  Waymo WOMD         │     │  OpenStreetMap +     │     │  Synthetic /         │
│  TFRecord           │     │  DEM elevation       │     │  Hand-authored       │
└────────┬────────────┘     └──────────┬───────────┘     └──────────┬───────────┘
         │ convert_waymo_to_sf.py      │ convert_osm_to_sf.py       │ (direct write)
         └────────────────────┬────────┘────────────────────────────┘
                              ▼
                    scene_XXXXXX.sf.json          ← THE SF FORMAT (this document)
                              │
              ┌───────────────┼──────────────────┐
              ▼               ▼                  ▼
       USD builder      lane_center          obs contract /
    (road mesh +        _sampler.py          reward kernels
     terrain import)
```

No downstream component reads from Waymo TFRecords, OSM queries, or any other source directly.
Everything passes through SF format.

---

## Top-Level Structure

```json
{
  "sf_version": "1.0",
  "meta": { ... },
  "coordinate_frame": { ... },
  "terrain": { ... },
  "road": { ... },
  "surfaces": [ ... ],
  "agents": { ... },
  "workzone": null
}
```

---

## Fields

### `sf_version` *(string, required)*
Format version string. Currently `"1.0"`. Readers must reject files with incompatible major versions.

---

### `meta` *(object, required)*
Provenance and identification. All fields optional except `scene_id`.

```json
"meta": {
  "scene_id": "scene_000042",
  "source": "waymo",
  "source_file": "uncompressed_scenario_training_training_tfexample.tfrecord-00000-of-01000",
  "source_scene_index": 42,
  "location_description": null,
  "bbox_latlon": null,
  "creation_date": "2026-05-18",
  "sf_converter_version": "1.0"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `scene_id` | string | Unique identifier, used as filename stem |
| `source` | string | `"waymo"` \| `"osm"` \| `"synthetic"` \| `"handcrafted"` |
| `source_file` | string \| null | Original source file path |
| `source_scene_index` | int \| null | Index within source file (Waymo scenario index, etc.) |
| `location_description` | string \| null | Human-readable place name (e.g. `"San Francisco, CA"`) |
| `bbox_latlon` | `[lat_min, lon_min, lat_max, lon_max]` \| null | WGS84 bounding box if known |
| `creation_date` | string \| null | ISO 8601 date |
| `sf_converter_version` | string \| null | Version of the converter script |

---

### `coordinate_frame` *(object, required)*
Defines the local Cartesian frame used for all XYZ coordinates in this file.

```json
"coordinate_frame": {
  "type": "local_enu",
  "origin_lat": 37.4419,
  "origin_lon": -122.1430,
  "origin_alt_m": 30.2,
  "z_up": true,
  "units": "meters"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"local_enu"` (East-North-Up) or `"waymo_world"` (Waymo's unnamed world frame) |
| `origin_lat/lon/alt_m` | float \| null | WGS84 origin of the local frame, if known |
| `z_up` | bool | Always `true` — Z is up |
| `units` | string | Always `"meters"` |

All `xyz` arrays in the file are in this frame.

---

### `terrain` *(object, required)*
Ground elevation and physical surface properties for the entire scene extent.

```json
"terrain": {
  "bounds_m": [-120.0, -120.0, 120.0, 120.0],
  "elevation_grid": null,
  "dem_source": null,
  "dem_resolution_m": null,
  "base_elevation_m": 0.0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `bounds_m` | `[xmin, ymin, xmax, ymax]` | Axis-aligned scene extent in local frame |
| `elevation_grid` | object \| null | Height-field grid (see below). `null` = flat at `base_elevation_m` |
| `dem_source` | string \| null | `"srtm"` \| `"copernicus"` \| `"waymo_lidar"` \| null |
| `dem_resolution_m` | float \| null | Grid cell size in meters |
| `base_elevation_m` | float | Fallback Z when `elevation_grid` is null. Default `0.0` |

**`elevation_grid`** (when not null):
```json
"elevation_grid": {
  "nx": 64,
  "ny": 64,
  "z_m": [[0.0, 0.1, ...], ...]
}
```
Row-major 2D array of Z values, `nx` columns × `ny` rows, covering `bounds_m`.
The USD builder samples this to set road mesh vertex Z. Isaac Lab `TerrainImporter` can consume
it directly as a height field.

---

### `road` *(object, required)*
The drivable road network. All lane geometry lives here.

```json
"road": {
  "stats": {
    "num_polylines": 47,
    "num_points_total": 1820
  },
  "polylines": [ ... ]
}
```

#### Polyline object

Each polyline is a single connected sequence of 3D points representing **one lane boundary,
lane centerline, or road edge**.

```json
{
  "id": 12,
  "type": 2,
  "type_name": "lane_center",
  "lane_id": 3,
  "n": 24,
  "xyz": [[x0,y0,z0], [x1,y1,z1], ...],
  "dir": [[dx0,dy0,dz0], ...],
  "half_width_m": 1.85,
  "half_length_m": null,
  "passable": true,
  "road_surface": "asphalt",
  "speed_limit_mps": 13.4,
  "osm_way_id": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Unique polyline ID within this scene |
| `type` | int | Numeric type code (see Type Table below) |
| `type_name` | string | Human-readable type label |
| `lane_id` | int \| null | Parent lane this polyline belongs to |
| `n` | int | Number of points |
| `xyz` | `[[x,y,z], ...]` | 3D point sequence in local frame — **Z is real elevation, never discarded** |
| `dir` | `[[dx,dy,dz], ...]` | Unit tangent direction at each point |
| `half_width_m` | float \| null | Half-width of the lane at this polyline (for mesh triangulation) |
| `half_length_m` | float \| null | Half-length of each segment (for instanced USD geometry) |
| `passable` | bool | `false` = lane is closed (workzone, barrier, etc.) |
| `road_surface` | string | `"asphalt"` \| `"concrete"` \| `"gravel"` \| `"steel_plate"` \| `"fresh_asphalt"` \| `"dirt"` |
| `speed_limit_mps` | float \| null | Posted speed limit in m/s. `null` = unknown |
| `osm_way_id` | int \| null | Source OSM way ID, if derived from OSM |

#### Polyline Type Table

| Code | Name | Description |
|------|------|-------------|
| 0 | `unknown` | Unknown / unmapped type |
| 1 | `lane_center` | Lane centerline (drivable) |
| 2 | `lane_boundary_left` | Left edge of a lane |
| 3 | `lane_boundary_right` | Right edge of a lane |
| 4 | `road_edge` | Outer road boundary (curb, edge of pavement) |
| 5 | `road_line_broken_single_white` | Broken white line |
| 6 | `road_line_solid_single_white` | Solid white line |
| 7 | `road_line_solid_double_yellow` | Double yellow (no-pass) |
| 8 | `stop_sign` | Stop line |
| 9 | `crosswalk` | Pedestrian crossing |
| 10 | `speed_bump` | Speed bump centerline |
| 11–19 | *(reserved)* | |
| 20 | `workzone_taper` | Workzone approach taper |
| 21 | `workzone_buffer` | Workzone buffer / tangent zone |
| 22 | `workzone_closed_lane` | Lane closed by workzone |
| 23 | `workzone_worker_zone` | Active worker area boundary |

Types 0–10 map directly to Waymo `roadgraph_samples/type` values.
Types 20+ are SceneFactory extensions (no Waymo equivalent).

---

### `surfaces` *(array, may be empty)*
Explicit polygon regions that override the default road surface friction.
Used for workzone surface patches (gravel, steel plate, fresh asphalt)
and for any area where friction differs from the lane-level default.

```json
"surfaces": [
  {
    "id": 0,
    "surface_type": "gravel",
    "polygon_xy": [[x0,y0], [x1,y1], [x2,y2], [x3,y3]],
    "z_m": 0.0,
    "friction_static": 0.45,
    "friction_dynamic": 0.40
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Unique surface patch ID |
| `surface_type` | string | One of the road_surface names |
| `polygon_xy` | `[[x,y], ...]` | Convex polygon boundary in local XY. 3+ points. |
| `z_m` | float | Elevation of the patch |
| `friction_static` | float | PhysX static friction coefficient |
| `friction_dynamic` | float | PhysX dynamic friction coefficient |

At runtime, the USD builder creates a `UsdGeom.Mesh` per surface patch with a bound
`UsdPhysics.MaterialAPI` — so friction is physically real, not a reward scalar.

---

### `agents` *(object, required)*
Initial traffic agent states extracted from the source (Waymo scenarios) or declared
synthetically. Used for initializing non-ego background agents or for scene classification.
Not used for RL agent spawning (that is handled by the lane sampler).

```json
"agents": {
  "count_valid": 12,
  "sdc": {
    "track_idx": 0,
    "start": {"x": 10.2, "y": -3.1, "z": 0.0, "yaw_rad": 1.57},
    "end":   {"x": 85.0, "y": 22.0, "z": 0.0, "yaw_rad": 1.60}
  },
  "items": [
    {
      "track_idx": 1,
      "agent_type": 1,
      "agent_type_name": "vehicle",
      "start": {"x": 5.0, "y": 0.0, "z": 0.0, "yaw_rad": 1.57},
      "end": null
    }
  ]
}
```

Agent type codes: `1=vehicle`, `2=pedestrian`, `3=cyclist`.

---

### `workzone` *(object \| null)*
Workzone annotation. `null` if this scene has no construction zone.

```json
"workzone": {
  "present": true,
  "source": "osm_tag",
  "taper_start_polyline_ids": [20, 21],
  "closed_lane_ids": [3, 4],
  "speed_limit_mps": 8.9,
  "worker_positions": [
    {"xyz": [45.0, 2.0, 0.0], "radius_m": 3.0}
  ],
  "description": "Two-lane closure with shoulder work"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `present` | bool | `true` if a workzone exists |
| `source` | string | `"osm_tag"` \| `"manual"` \| `"synthetic"` |
| `taper_start_polyline_ids` | `[int]` | IDs of polylines marking the taper entry |
| `closed_lane_ids` | `[int]` | Lane IDs that are `passable: false` due to this workzone |
| `speed_limit_mps` | float \| null | Reduced workzone speed limit |
| `worker_positions` | array | Flagger / worker spawn points with exclusion radius |
| `description` | string \| null | Free-text annotation |

---

## Converter Responsibilities

| Converter | Source | Adds |
|-----------|--------|------|
| `convert_waymo_to_sf.py` | Waymo TFRecord | `meta.source=waymo`, all road polylines (types 0–10), agent tracks. Z from LiDAR. `workzone=null`. |
| `convert_osm_to_sf.py` | OSM + DEM | `meta.source=osm`, road polylines from OSM ways (types 1–8), elevation from DEM raster, `bbox_latlon`, `osm_way_id`. Workzone from `highway=construction` tags. |
| `generate_synthetic_scene.py` | Procedural | Any types, full workzone support, controllable geometry. |

All converters must:
1. Set `sf_version: "1.0"`
2. Never discard Z — if elevation is unknown, set to `base_elevation_m`, do not silently zero
3. Set `passable: false` on any lane that is physically closed
4. Populate `road_surface` per polyline where known; default `"asphalt"`

---

## Backward Compatibility

The current Waymo JSON schema (`meta` / `road.polylines` / `agents`) maps to SF Format as follows:

| Current field | SF Format field |
|---------------|-----------------|
| `meta.source_file` | `meta.source_file` ✓ |
| `road.polylines[].type` | `road.polylines[].type` ✓ |
| `road.polylines[].id` | `road.polylines[].id` ✓ |
| `road.polylines[].xyz` | `road.polylines[].xyz` ✓ (Z now preserved) |
| `road.polylines[].dir` | `road.polylines[].dir` ✓ |
| `agents.sdc` | `agents.sdc` ✓ |
| `agents.items` | `agents.items` ✓ |
| *(missing)* | `sf_version` — new required field |
| *(missing)* | `coordinate_frame` — new required field |
| *(missing)* | `terrain` — new required field |
| *(missing)* | `surfaces` — new field (empty array default) |
| *(missing)* | `workzone` — new field (null default) |
| *(missing)* | `road.polylines[].passable` — new field (true default) |
| *(missing)* | `road.polylines[].road_surface` — new field |
| *(missing)* | `road.polylines[].half_width_m` — new field |
| *(missing)* | `road.polylines[].type_name` — new field |

A migration script `scripts/migrate_waymo_json_to_sf.py` will upgrade existing
`scene_XXXXXX.json` files to `scene_XXXXXX.sf.json` with safe defaults.
