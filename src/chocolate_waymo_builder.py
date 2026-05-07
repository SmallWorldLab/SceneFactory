
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

import omni.usd
import omni.kit.app
from src.trfc.lane_center_sampler import compute_scene_center_from_road

# ---- PhysX vehicle wizard imports ----
#
# The road-only mini-world builder should remain usable even when the legacy vehicle wizard
# Python module is not exposed on the current Isaac Sim install. Keep these imports optional
# and fail only if the caller actually tries to spawn a wizard vehicle.
try:
    from omni.physxvehicle.scripts.wizards import physxVehicleWizard as VehicleWizard
    from omni.physxvehicle.scripts.helpers.UnitScale import UnitScale
    from omni.physxvehicle.scripts.commands import PhysXVehicleWizardCreateCommand
except ModuleNotFoundError:
    try:
        from omni.physx.vehicle.scripts.wizards import physxVehicleWizard as VehicleWizard
        from omni.physx.vehicle.scripts.helpers.UnitScale import UnitScale
        from omni.physx.vehicle.scripts.commands import PhysXVehicleWizardCreateCommand
    except ModuleNotFoundError:
        VehicleWizard = None  # type: ignore[assignment]
        PhysXVehicleWizardCreateCommand = None  # type: ignore[assignment]

        @dataclass(frozen=True)
        class UnitScale:  # type: ignore[override]
            lengthScale: float
            massScale: float

# PhysX schema optional (for trigger API if available)
try:
    from pxr import PhysxSchema
except Exception:
    PhysxSchema = None  # type: ignore

ROOT_PATH = "/World"
SHARED_ROOT = ROOT_PATH + "/VehicleShared"


# ----------------------------
# Utilities
# ----------------------------

def _ensure_world_default_prim(stage: Usd.Stage) -> None:
    world = stage.GetPrimAtPath(ROOT_PATH)
    if not world.IsValid():
        world = UsdGeom.Xform.Define(stage, ROOT_PATH).GetPrim()
    if not stage.GetDefaultPrim().IsValid():
        stage.SetDefaultPrim(world)

def _get_unit_scale(stage: Usd.Stage) -> Tuple[UnitScale, float]:
    """Return (UnitScale, meters_per_unit)."""
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    if meters_per_unit == 0:
        meters_per_unit = 0.01  # Isaac default often cm
    length_scale = 1.0 / meters_per_unit

    kilograms_per_unit = UsdPhysics.GetStageKilogramsPerUnit(stage)
    if kilograms_per_unit == 0:
        kilograms_per_unit = 1.0
    mass_scale = 1.0 / kilograms_per_unit
    return UnitScale(length_scale, mass_scale), meters_per_unit

def _meters_per_unit(stage: Usd.Stage) -> float:
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    return float(mpu) if mpu and float(mpu) > 0 else 0.01

def _safe_int(x, default=-1) -> int:
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default

def _safe_float(x, default=None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default

def _quath_from_yaw_z(yaw_rad: float) -> Gf.Quath:
    """Half-precision quaternion rotation about +Z for USD PointInstancer orientations."""
    half = 0.5 * float(yaw_rad)
    w = float(np.cos(half))
    z = float(np.sin(half))
    try:
        return Gf.Quath(Gf.Half(w), Gf.Half(0.0), Gf.Half(0.0), Gf.Half(z))
    except Exception:
        return Gf.Quath(w, 0.0, 0.0, z)

def _reduce_polyline_xy(points_xyz: np.ndarray, area_thresh: float) -> np.ndarray:
    """Iteratively remove points with small local triangle area (XY) under threshold."""
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.shape[0] < 3 or area_thresh <= 0:
        return pts

    keep = np.ones((pts.shape[0],), dtype=bool)
    changed = True

    def tri_area_xy(a, b, c) -> float:
        abx = float(b[0] - a[0]); aby = float(b[1] - a[1])
        acx = float(c[0] - a[0]); acy = float(c[1] - a[1])
        cross = abx * acy - aby * acx
        return 0.5 * abs(cross)

    while changed:
        changed = False
        idxs = np.nonzero(keep)[0]
        if idxs.size < 3:
            break
        for k in range(1, idxs.size - 1):
            i0 = idxs[k - 1]; i1 = idxs[k]; i2 = idxs[k + 1]
            if tri_area_xy(pts[i0], pts[i1], pts[i2]) < area_thresh:
                keep[i1] = False
                changed = True
    return pts[keep]

def _get_world_translation_m(stage: Usd.Stage, prim_path: str) -> Optional[Tuple[float, float, float]]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    xf = UsdGeom.Xformable(prim)
    if not xf:
        return None
    m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = m.ExtractTranslation()  # stage units
    mpu = _meters_per_unit(stage)
    return (float(t[0] * mpu), float(t[1] * mpu), float(t[2] * mpu))
from pxr import UsdShade


def _apply_contact_report_to_colliders(root_prim: Usd.Prim) -> None:
    """Apply PhysX contact reporting to all collision shapes under root_prim."""
    if PhysxSchema is None or not root_prim.IsValid():
        return
    for prim in Usd.PrimRange(root_prim):
        try:
            if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().IsValid():
                continue
            cr = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            if hasattr(cr, "CreateThresholdAttr"):
                cr.CreateThresholdAttr().Set(0.0)
            else:
                cr.CreatePhysxContactReportThresholdAttr().Set(0.0)
        except Exception:
            continue

def _road_type_color_srgb(t: int) -> Tuple[float, float, float]:
    # Big-contrast palette (sRGB). Tune as you like.
    palette = [
        (0.95, 0.35, 0.35),  # red-ish
        (0.35, 0.75, 0.95),  # sky
        (0.35, 0.95, 0.55),  # green
        (0.95, 0.85, 0.35),  # yellow
        (0.75, 0.45, 0.95),  # purple
        (0.95, 0.55, 0.25),  # orange
        (0.55, 0.95, 0.90),  # teal
        (0.85, 0.85, 0.85),  # light gray
    ]
    if t < 0:
        return (0.7, 0.7, 0.7)
    return palette[int(t) % len(palette)]

def _srgb_to_linear(c: float) -> float:
    c = float(c)
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4

def _rgb_srgb_to_linear(rgb):
    return Gf.Vec3f(_srgb_to_linear(rgb[0]), _srgb_to_linear(rgb[1]), _srgb_to_linear(rgb[2]))

def _get_or_create_preview_material(stage: Usd.Stage, mat_path: str,
                                   rgb_srgb=(0.6, 0.6, 0.6),
                                   emissive_strength: float = 0.0) -> UsdShade.Material:
    """
    UsdPreviewSurface material.
    - rgb_srgb is in sRGB for convenience; converted to linear.
    - emissive_strength > 0 makes it pop in lit scenes.
    """
    mat = UsdShade.Material.Get(stage, mat_path)
    if not mat:
        mat = UsdShade.Material.Define(stage, mat_path)

    shader_path = mat_path + "/PreviewSurface"
    shader = UsdShade.Shader.Get(stage, shader_path)
    if not shader:
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")

    rgb_lin = _rgb_srgb_to_linear(rgb_srgb)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(rgb_lin)

    if emissive_strength and emissive_strength > 0:
        # emissive = rgb * strength (still Color3f)
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(rgb_lin[0] * emissive_strength, rgb_lin[1] * emissive_strength, rgb_lin[2] * emissive_strength)
        )

    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat

def _bind_material(prim: Usd.Prim, material: UsdShade.Material) -> None:
    try:
        UsdShade.MaterialBindingAPI(prim).Bind(material)
    except Exception:
        pass


# ----------------------------
# Goal manager (polling, no PhysX trigger subscription)
# ----------------------------

class _GoalPollingManager:
    """
    Very stable across Isaac versions:
    - subscribe to app update stream
    - each frame: check distance(car, goal_center) <= radius
    - if yes: remove both car and goal prims
    """
    def __init__(self, stage: Usd.Stage, *, remove_prims_on_reach: bool = False):
        self.stage = stage
        self.remove_prims_on_reach = bool(remove_prims_on_reach)
        self._sub = None
        # key: goal_root_path
        # val: dict(center=(x,y,z) meters, radius, car_path, agent_id)
        self.goals: Dict[str, Dict[str, Any]] = {}

    def ensure_subscription(self) -> None:
        if self._sub is not None:
            return
        stream = omni.kit.app.get_app().get_update_event_stream()
        self._sub = stream.create_subscription_to_pop(self._on_update, name="goal_polling")

    def add_goal(self, goal_root_path: str, *, center_m: Tuple[float, float, float], radius_m: float,
                 car_root_path: str, agent_id: int) -> None:
        self.goals[goal_root_path] = {
            "center_m": tuple(map(float, center_m)),
            "radius_m": float(radius_m),
            "car_root_path": str(car_root_path),
            "agent_id": int(agent_id),
        }
        self.ensure_subscription()

    def _on_update(self, e) -> None:
        if not self.goals:
            return

        stage = self.stage
        to_remove: List[str] = []

        # iterate over snapshot so we can delete
        for goal_path, info in list(self.goals.items()):
            car_path = info["car_root_path"]
            center = info["center_m"]
            r = float(info["radius_m"])
            r2 = r * r

            car_t = _get_world_translation_m(stage, car_path)
            if car_t is None:
                # car gone -> cleanup goal too
                to_remove.append(goal_path)
                continue

            dx = float(car_t[0] - center[0])
            dy = float(car_t[1] - center[1])
            dz = float(car_t[2] - center[2])
            d2 = dx * dx + dy * dy + dz * dz

            if d2 <= r2:
                # correct car reached its own goal
                # TRAINING-SAFE: do NOT delete the car or the goal.
                # Just mark the goal as reached and stop polling this pair.
                try:
                    gp = stage.GetPrimAtPath(goal_path)
                    if gp.IsValid():
                        gp.SetCustomDataByKey("reached", True)
                        gp.SetCustomDataByKey("reached_by_agent_id", int(info.get("agent_id", -1)))
                except Exception:
                    pass

                # Stop checking this goal to save overhead (but keep prims in stage)
                to_remove.append(goal_path)


        for gp in to_remove:
            self.goals.pop(gp, None)


# module-level singleton per stage (good enough for your demo)
_GOAL_MGR: Optional[_GoalPollingManager] = None
def _goal_mgr(stage: Usd.Stage) -> _GoalPollingManager:
    global _GOAL_MGR
    if _GOAL_MGR is None or _GOAL_MGR.stage != stage:
        _GOAL_MGR = _GoalPollingManager(stage)
    return _GOAL_MGR


# ----------------------------
# Mini-world builder (ONE world)
# ----------------------------

@dataclass
class LocalBounds:
    width_m: float = 200.0
    length_m: float = 200.0
    origin_xy: Tuple[float, float] = (0.0, 0.0)   # local-space center of bounds


class WaymoJsonMiniWorldBuilder:
    """
    Builds ONE mini-world under world_root.

    Behavior:
    - start_in_goal agents become PARKED CARS:
        * static collider on chassis
        * visual wheels only (no wheel colliders)
        * NO goal spawned
    - other agents remain full PhysX vehicles (wizard) + green ring goal
      goal completion is detected via cheap per-frame distance check (no trigger-report API).
    """

    def __init__(
        self,
        stage: Optional[Usd.Stage] = None,
        world_root: str = "/World/MiniWorlds/world_000",
        bounds: Optional[LocalBounds] = None,
        origin_mode: str = "center",  # "center" or "zero"
        origin_center_mode: str = "mean",  # "mean" or "bbox"
    ):
        self.stage = stage or omni.usd.get_context().get_stage()
        self.world_root = world_root
        self.bounds = bounds or LocalBounds()
        self.origin_mode = origin_mode
        self.origin_center_mode = str(origin_center_mode).strip().lower()

        self._scene_center = np.zeros((3,), dtype=np.float32)

        # ensure root prim exists + subfolders
        UsdGeom.Xform.Define(self.stage, self.world_root)
        self.road_root = f"{self.world_root}/Road"
        self.agents_root = f"{self.world_root}/Agents"
        self.goals_root = f"{self.world_root}/Goals"
        UsdGeom.Xform.Define(self.stage, self.road_root)
        UsdGeom.Xform.Define(self.stage, self.agents_root)
        UsdGeom.Xform.Define(self.stage, self.goals_root)

    # -------- bounds gating in LOCAL coords --------
    def _in_bounds_xy(self, x: float, y: float) -> bool:
        ox, oy = self.bounds.origin_xy
        hw = 0.5 * float(self.bounds.width_m)
        hl = 0.5 * float(self.bounds.length_m)
        return (ox - hw <= x <= ox + hw) and (oy - hl <= y <= oy + hl)

    def _filter_polyline_points(self, pts: np.ndarray) -> np.ndarray:
        if pts.shape[0] == 0:
            return pts
        m = np.array([self._in_bounds_xy(float(p[0]), float(p[1])) for p in pts], dtype=bool)
        return pts[m]

    # -------- origin shift --------
    def _compute_scene_center_from_road(self, polylines: List[Dict[str, Any]]) -> np.ndarray:
        if self.origin_center_mode != "bbox":
            return compute_scene_center_from_road({"road": {"polylines": polylines}})
        all_pts = []
        for pl in polylines:
            xyz = pl.get("xyz", None)
            if not xyz:
                continue
            arr = np.asarray(xyz, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 1:
                if arr.shape[1] == 2:
                    arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=np.float32)], axis=1)
                all_pts.append(arr[:, :3])
        if not all_pts:
            return np.zeros((3,), dtype=np.float32)
        pts = np.concatenate(all_pts, axis=0)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        return 0.5 * (mins + maxs)

    def _to_local_xyz(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        return (float(x - self._scene_center[0]),
                float(y - self._scene_center[1]),
                float(z - self._scene_center[2]))

    def _is_start_within_goal(self, sx: float, sy: float, sz: float, ex: float, ey: float, ez: float, radius_m: float) -> bool:
        dx = float(ex - sx)
        dy = float(ey - sy)
        dz = float(ez - sz)
        return (dx * dx + dy * dy + dz * dz) <= float(radius_m * radius_m)

    # -------- road segments grouped by type --------
    def build_road_segments_grouped(
        self,
        cfg: Dict[str, Any],
        polyline_reduction_area: float = 0.0,   # 0 disables reduction
        min_points_for_reduction: int = 10,
        jump_break_m: float = 3.0,
        seg_width: float = 0.10,
        seg_height: float = 0.10,
        z_lift: float = 0.02,
        max_segments_per_type: Optional[int] = None,
        flatten_road_z: bool = True,
        road_z_m: float = 0.0,
        enable_segment_collision: bool = False,
        trigger_enable: bool = False,
        trigger_height_m: float = 1.0,
        trigger_width_scale: float = 1.0,
        trigger_offset_z_m: float = 0.5,
        trigger_match_segment: bool = True,
        trigger_script_enable: bool = True,
        allowed_road_types: Optional[Sequence[int]] = None,
        road_render_mode: str = "point_instancer",
    ) -> None:
        road = (cfg.get("road", {}) or {})
        polylines = road.get("polylines", []) or []
        road_points_all: List[Gf.Vec3f] = []
        road_dirs_all: List[Gf.Vec3f] = []
        road_types_all: List[int] = []
        road_half_lengths_all: List[float] = []
        road_half_widths_all: List[float] = []
        allowed_types = None if allowed_road_types is None else {int(x) for x in allowed_road_types}
        render_mode = str(road_render_mode).strip().lower()
        if render_mode not in {"point_instancer", "explicit_prims"}:
            raise ValueError(f"Unsupported road_render_mode: {road_render_mode!r}")

        # compute scene center (for origin_mode="center")
        if self.origin_mode == "center":
            self._scene_center = self._compute_scene_center_from_road(polylines)
        else:
            self._scene_center = np.zeros((3,), dtype=np.float32)

        # group polylines by type
        by_type: Dict[int, List[np.ndarray]] = {}
        for pl in polylines:
            t = _safe_int(pl.get("type", -1), -1)
            if allowed_types is not None and int(t) not in allowed_types:
                continue
            xyz = pl.get("xyz", None)
            if not xyz:
                continue
            pts = np.asarray(xyz, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            if pts.shape[1] == 2:
                pts = np.concatenate([pts, np.zeros((pts.shape[0], 1), dtype=np.float32)], axis=1)

            pts_local = np.array([self._to_local_xyz(p[0], p[1], p[2]) for p in pts[:, :3]], dtype=np.float32)
            if flatten_road_z:
                pts_local[:, 2] = float(road_z_m)
            if pts_local.shape[0] < 2:
                continue

            if polyline_reduction_area > 0 and pts_local.shape[0] >= int(min_points_for_reduction):
                pts_local = _reduce_polyline_xy(pts_local, float(polyline_reduction_area))
                if pts_local.shape[0] < 2:
                    continue

            by_type.setdefault(t, []).append(pts_local)

        # Build one road group per type. Point instancers are compact but have been unstable under
        # fabric/visibility updates for these road lines, so explicit prims are supported as a
        # clean fallback for SceneFactory.
        for t, polys in sorted(by_type.items(), key=lambda kv: kv[0]):
            type_root = f"{self.road_root}/Type_{t:02d}"
            segments_root = f"{type_root}/Segments"

            UsdGeom.Xform.Define(self.stage, type_root)

            seg_positions_py = []
            seg_orients_py = []
            seg_scales_py = []
            proto_indices_py = []
            trigger_positions_py = []
            trigger_scales_py = []
            segment_yaws_py = []

            seg_count = 0
            for poly in polys:
                for i in range(poly.shape[0] - 1):
                    p0 = poly[i]
                    p1 = poly[i + 1]

                    if not (
                        self._in_bounds_xy(float(p0[0]), float(p0[1]))
                        and self._in_bounds_xy(float(p1[0]), float(p1[1]))
                    ):
                        continue

                    dx = float(p1[0] - p0[0])
                    dy = float(p1[1] - p0[1])
                    length = float(math.sqrt(dx * dx + dy * dy))
                    if length < 1e-6:
                        continue
                    if length > float(jump_break_m):
                        continue

                    z_base = float(road_z_m) if flatten_road_z else float((p0[2] + p1[2]) * 0.5)
                    mid = Gf.Vec3f(
                        float((p0[0] + p1[0]) * 0.5),
                        float((p0[1] + p1[1]) * 0.5),
                        float(z_base + z_lift),
                    )

                    yaw = float(math.atan2(dy, dx))
                    q = _quath_from_yaw_z(yaw)
                    scale = Gf.Vec3f(float(length), float(seg_width), float(seg_height))

                    seg_positions_py.append(mid)
                    seg_orients_py.append(q)
                    seg_scales_py.append(scale)
                    proto_indices_py.append(0)
                    segment_yaws_py.append(float(yaw))
                    road_points_all.append(mid)
                    road_dirs_all.append(
                        Gf.Vec3f(float(dx / length), float(dy / length), 0.0)
                    )
                    road_types_all.append(int(t))
                    road_half_lengths_all.append(0.5 * float(length))
                    road_half_widths_all.append(0.5 * float(seg_width))
                    if trigger_enable:
                        trigger_h = float(seg_height) if trigger_match_segment else float(trigger_height_m)
                        trigger_positions_py.append(
                            Gf.Vec3f(float(mid[0]), float(mid[1]), float(mid[2] + trigger_offset_z_m))
                        )
                        trigger_scales_py.append(
                            Gf.Vec3f(float(length), float(seg_width * trigger_width_scale), float(trigger_h))
                        )

                    seg_count += 1
                    if max_segments_per_type is not None and seg_count >= int(max_segments_per_type):
                        break
                if max_segments_per_type is not None and seg_count >= int(max_segments_per_type):
                    break

            if seg_count == 0:
                continue

            rgb = _road_type_color_srgb(int(t))
            mat_path = f"{type_root}/Materials/RoadType_{int(t):02d}"
            mat = _get_or_create_preview_material(self.stage, mat_path, rgb_srgb=rgb, emissive_strength=0.15)
            if render_mode == "explicit_prims":
                UsdGeom.Xform.Define(self.stage, segments_root)
                for seg_index, (mid, scale, yaw_rad) in enumerate(zip(seg_positions_py, seg_scales_py, segment_yaws_py)):
                    seg_path = f"{segments_root}/Seg_{seg_index:05d}"
                    cube = UsdGeom.Cube.Define(self.stage, seg_path)
                    cube.GetSizeAttr().Set(1.0)
                    seg_api = UsdGeom.XformCommonAPI(cube)
                    seg_api.SetTranslate(Gf.Vec3d(float(mid[0]), float(mid[1]), float(mid[2])))
                    seg_api.SetRotate(
                        Gf.Vec3f(0.0, 0.0, math.degrees(float(yaw_rad))),
                        UsdGeom.XformCommonAPI.RotationOrderXYZ,
                    )
                    seg_api.SetScale(Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])))
                    _bind_material(cube.GetPrim(), mat)
                    try:
                        UsdGeom.Gprim(cube.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
                    except Exception:
                        pass
                    if enable_segment_collision:
                        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                        try:
                            UsdPhysics.CollisionAPI(cube.GetPrim()).CreateCollisionEnabledAttr(True)
                        except Exception:
                            pass
                    cube.GetPrim().SetCustomDataByKey("road_type", int(t))

                if trigger_enable and trigger_positions_py:
                    trigger_path = f"{type_root}/Triggers"
                    UsdGeom.Xform.Define(self.stage, trigger_path)
                    for seg_index, (mid, scale, yaw_rad) in enumerate(
                        zip(trigger_positions_py, trigger_scales_py, segment_yaws_py)
                    ):
                        trig_path = f"{trigger_path}/Trig_{seg_index:05d}"
                        trig_cube = UsdGeom.Cube.Define(self.stage, trig_path)
                        trig_cube.GetSizeAttr().Set(1.0)
                        trig_api = UsdGeom.XformCommonAPI(trig_cube)
                        trig_api.SetTranslate(Gf.Vec3d(float(mid[0]), float(mid[1]), float(mid[2])))
                        trig_api.SetRotate(
                            Gf.Vec3f(0.0, 0.0, math.degrees(float(yaw_rad))),
                            UsdGeom.XformCommonAPI.RotationOrderXYZ,
                        )
                        trig_api.SetScale(Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])))
                        UsdGeom.Imageable(trig_cube.GetPrim()).MakeInvisible()
                        UsdPhysics.CollisionAPI.Apply(trig_cube.GetPrim())
                        if PhysxSchema is not None:
                            try:
                                PhysxSchema.PhysxTriggerAPI.Apply(trig_cube.GetPrim())
                            except Exception:
                                pass
                            if trigger_script_enable:
                                try:
                                    trigger_script = Path(__file__).resolve().parent / "trigger_road_contact.py"
                                    trig_api_schema = PhysxSchema.PhysxTriggerAPI.Apply(trig_cube.GetPrim())
                                    trig_api_schema.CreateEnterScriptTypeAttr().Set(PhysxSchema.Tokens.scriptFile)
                                    trig_api_schema.CreateOnEnterScriptAttr().Set(str(trigger_script))
                                    trig_api_schema.CreateLeaveScriptTypeAttr().Set(PhysxSchema.Tokens.scriptFile)
                                    trig_api_schema.CreateOnLeaveScriptAttr().Set(str(trigger_script))
                                except Exception:
                                    pass
                        trig_cube.GetPrim().SetCustomDataByKey("road_type", int(t))
            else:
                # Match the older stable chocolate authoring path and Isaac Lab's own
                # marker initialization pattern more closely:
                # 1. define the instancer
                # 2. define a prototype under it
                # 3. seed one dummy instance so Fabric initializes the prototype table
                # 4. overwrite with the final arrays
                instancer = UsdGeom.PointInstancer.Define(self.stage, segments_root)
                proto_path = f"{segments_root}/CubeProto"
                cube = UsdGeom.Cube.Define(self.stage, proto_path)
                cube.GetSizeAttr().Set(1.0)
                # For cloned envs, keep the prototype rendering path as simple as possible:
                # use displayColor on the prototype and avoid relying on a material binding relationship.
                try:
                    UsdGeom.Gprim(cube.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
                except Exception:
                    pass

                if enable_segment_collision:
                    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                    try:
                        UsdPhysics.CollisionAPI(cube.GetPrim()).CreateCollisionEnabledAttr(True)
                    except Exception:
                        pass
                cube.GetPrim().SetCustomDataByKey("road_type", int(t))

                instancer.CreatePrototypesRel().SetTargets([cube.GetPath()])
                instancer.GetProtoIndicesAttr().Set(Vt.IntArray([0]))
                instancer.GetPositionsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.0)]))
                instancer.GetOrientationsAttr().Set(Vt.QuathArray([_quath_from_yaw_z(0.0)]))
                instancer.GetScalesAttr().Set(Vt.Vec3fArray([Gf.Vec3f(1.0)]))

                instancer.GetPositionsAttr().Set(Vt.Vec3fArray(seg_positions_py))
                instancer.GetOrientationsAttr().Set(Vt.QuathArray(seg_orients_py))
                instancer.GetScalesAttr().Set(Vt.Vec3fArray(seg_scales_py))
                instancer.GetProtoIndicesAttr().Set(Vt.IntArray(proto_indices_py))
                instancer.GetPrim().SetCustomDataByKey("road_type", int(t))

                if trigger_enable and trigger_positions_py:
                    trigger_path = f"{type_root}/Triggers"
                    trig_inst = UsdGeom.PointInstancer.Define(self.stage, trigger_path)
                    trig_proto_path = f"{trigger_path}/CubeProto"
                    trig_cube = UsdGeom.Cube.Define(self.stage, trig_proto_path)
                    trig_cube.GetSizeAttr().Set(1.0)

                    UsdGeom.Imageable(trig_cube.GetPrim()).MakeInvisible()
                    UsdPhysics.CollisionAPI.Apply(trig_cube.GetPrim())
                    if PhysxSchema is not None:
                        try:
                            PhysxSchema.PhysxTriggerAPI.Apply(trig_cube.GetPrim())
                        except Exception:
                            pass
                        if trigger_script_enable:
                            try:
                                trigger_script = Path(__file__).resolve().parent / "trigger_road_contact.py"
                                trig_api = PhysxSchema.PhysxTriggerAPI.Apply(trig_cube.GetPrim())
                                trig_api.CreateEnterScriptTypeAttr().Set(PhysxSchema.Tokens.scriptFile)
                                trig_api.CreateOnEnterScriptAttr().Set(str(trigger_script))
                                trig_api.CreateLeaveScriptTypeAttr().Set(PhysxSchema.Tokens.scriptFile)
                                trig_api.CreateOnLeaveScriptAttr().Set(str(trigger_script))
                            except Exception:
                                pass

                    trig_cube.GetPrim().SetCustomDataByKey("road_type", int(t))
                    trig_inst.CreatePrototypesRel().SetTargets([trig_cube.GetPath()])
                    trig_inst.GetProtoIndicesAttr().Set(Vt.IntArray([0]))
                    trig_inst.GetPositionsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.0)]))
                    trig_inst.GetOrientationsAttr().Set(Vt.QuathArray([_quath_from_yaw_z(0.0)]))
                    trig_inst.GetScalesAttr().Set(Vt.Vec3fArray([Gf.Vec3f(1.0)]))
                    trig_inst.GetPositionsAttr().Set(Vt.Vec3fArray(trigger_positions_py))
                    trig_inst.GetOrientationsAttr().Set(Vt.QuathArray(seg_orients_py))
                    trig_inst.GetScalesAttr().Set(Vt.Vec3fArray(trigger_scales_py))
                    trig_inst.GetProtoIndicesAttr().Set(Vt.IntArray(proto_indices_py))
                    trig_inst.GetPrim().SetCustomDataByKey("road_type", int(t))

        if road_points_all:
            root_prim = self.stage.GetPrimAtPath(self.world_root)
            root_prim.SetCustomDataByKey("road_points_m", Vt.Vec3fArray(road_points_all))
            root_prim.SetCustomDataByKey("road_point_dirs", Vt.Vec3fArray(road_dirs_all))
            root_prim.SetCustomDataByKey("road_point_types", Vt.IntArray(road_types_all))
            root_prim.SetCustomDataByKey("road_point_half_lengths_m", Vt.FloatArray(road_half_lengths_all))
            root_prim.SetCustomDataByKey("road_point_half_widths_m", Vt.FloatArray(road_half_widths_all))

    # -------- vehicles + parked cars --------

    def _spawn_vehicle_wizard_under(self, parent_path: str, position_m: Tuple[float, float, float], yaw_deg: float) -> Optional[str]:
        stage = self.stage
        _ensure_world_default_prim(stage)

        if VehicleWizard is None or PhysXVehicleWizardCreateCommand is None:
            raise RuntimeError(
                "PhysX vehicle wizard Python modules are not available on this Isaac Sim install. "
                "Road-only world building still works, but wizard vehicle spawning does not."
            )

        parent_xform = UsdGeom.Xform.Define(stage, parent_path)
        xform_api = UsdGeom.XformCommonAPI(parent_xform)

        unit_scale, meters_per_unit = _get_unit_scale(stage)
        x_m, y_m, z_m = position_m
        pos_units = Gf.Vec3d(x_m / meters_per_unit, y_m / meters_per_unit, z_m / meters_per_unit)

        xform_api.SetTranslate(pos_units)
        xform_api.SetRotate(Gf.Vec3f(0.0, 0.0, float(yaw_deg)), UsdGeom.XformCommonAPI.RotationOrderXYZ)

        vehicle_data = VehicleWizard.VehicleData(
            unit_scale,
            VehicleWizard.VehicleData.AXIS_Z,
            VehicleWizard.VehicleData.AXIS_X,
        )
        vehicle_data.rootVehiclePath = parent_path + "/Vehicle"
        vehicle_data.rootSharedPath = SHARED_ROOT

        ret = PhysXVehicleWizardCreateCommand.execute(vehicle_data)
        success = bool(ret[0]) if isinstance(ret, (tuple, list)) and len(ret) > 0 else False
        if not success:
            payload = ret[1] if isinstance(ret, (tuple, list)) and len(ret) > 1 else None
            messages = payload[0] if isinstance(payload, (tuple, list)) and len(payload) > 0 else None
            print("[VEH WIZ FAIL]", parent_path, "messages:", messages)
            return None

        return vehicle_data.rootVehiclePath

    def _infer_vehicle_size_m(
        self,
        vehicle_root_path: str,
        *,
        default_size_m: Tuple[float, float, float] = (4.0, 2.0, 1.0),
    ) -> Tuple[float, float, float]:
        stage = self.stage
        mpu = _meters_per_unit(stage)
        tc = Usd.TimeCode.Default()
        bbox_cache = UsdGeom.BBoxCache(tc, [UsdGeom.Tokens.default_], useExtentsHint=True)

        candidates = [
            f"{vehicle_root_path}/Vehicle",
            vehicle_root_path,
        ]
        best = None
        best_score = -1.0
        for path in candidates:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            try:
                box = bbox_cache.ComputeLocalBound(prim)
                rng = box.GetRange()
                size = rng.GetSize()
                lx = abs(float(size[0])) * mpu
                wy = abs(float(size[1])) * mpu
                hz = abs(float(size[2])) * mpu
                if lx <= 1e-3 or wy <= 1e-3:
                    continue
                score = lx * wy * max(hz, 1e-3)
                if score > best_score:
                    best = (lx, wy, hz)
                    best_score = score
            except Exception:
                continue

        if best is None:
            return (
                float(default_size_m[0]),
                float(default_size_m[1]),
                float(default_size_m[2]),
            )
        return best

    def _set_vehicle_size_metadata(
        self,
        vehicle_prim: Usd.Prim,
        *,
        size_m: Tuple[float, float, float],
    ) -> None:
        if vehicle_prim is None or not vehicle_prim.IsValid():
            return
        lx = float(size_m[0])
        wy = float(size_m[1])
        hz = float(size_m[2])
        vehicle_prim.SetCustomDataByKey("vehicle_size_m", (lx, wy, hz))
        vehicle_prim.SetCustomDataByKey("vehicle_length_m", lx)
        vehicle_prim.SetCustomDataByKey("vehicle_width_m", wy)
        vehicle_prim.SetCustomDataByKey("vehicle_height_m", hz)

    def _spawn_parked_car_visual(
        self,
        path: str,
        *,
        position_m: Tuple[float, float, float],
        yaw_deg: float,
        chassis_size_m: Tuple[float, float, float] = (4.0, 2.0, 1.0),  # L,W,H
        wheel_radius_m: float = 0.35,
        wheel_thickness_m: float = 0.15,
        wheel_inset_x_m: float = 0.35,
        wheel_inset_y_m: float = 0.25,
        ground_clearance_m: float = 0.25,
    ) -> str:
        stage = self.stage
        mpu = _meters_per_unit(stage)

        L, W, H = chassis_size_m
        clearance = float(max(0.0, ground_clearance_m))

        root = UsdGeom.Xform.Define(stage, path)
        xapi = UsdGeom.XformCommonAPI(root)
        xapi.SetTranslate(Gf.Vec3d(position_m[0] / mpu, position_m[1] / mpu, position_m[2] / mpu))
        xapi.SetRotate(Gf.Vec3f(0.0, 0.0, float(yaw_deg)), UsdGeom.XformCommonAPI.RotationOrderXYZ)

        chassis = UsdGeom.Cube.Define(stage, path + "/Chassis")
        chassis.GetSizeAttr().Set(1.0)

        capi = UsdGeom.XformCommonAPI(chassis)
        capi.SetTranslate(Gf.Vec3d(0.0, 0.0, float((0.5 * H + clearance) / mpu)))
        capi.SetScale(Gf.Vec3f(float(L / mpu), float(W / mpu), float(H / mpu)))

        # cheap static collider
        UsdPhysics.CollisionAPI.Apply(chassis.GetPrim())

        wheels_root = UsdGeom.Xform.Define(stage, path + "/Wheels")

        half_L = 0.5 * L
        half_W = 0.5 * W
        wheel_inset_x_m = float(max(0.0, min(wheel_inset_x_m, half_L - 1e-3)))
        wheel_inset_y_m = float(max(0.0, min(wheel_inset_y_m, half_W - 1e-3)))

        wx_front = +(half_L - wheel_inset_x_m)
        wx_rear  = -(half_L - wheel_inset_x_m)
        wy_left  = +(half_W - wheel_inset_y_m)
        wy_right = -(half_W - wheel_inset_y_m)

        wz = float(wheel_radius_m)  # wheels touch ground
        wheel_rot = Gf.Vec3f(90.0, 0.0, 0.0)

        wheel_specs = {
            "FL": (wx_front, wy_left,  wz),
            "FR": (wx_front, wy_right, wz),
            "RL": (wx_rear,  wy_left,  wz),
            "RR": (wx_rear,  wy_right, wz),
        }

        for name, (wx, wy, wz_m) in wheel_specs.items():
            wpath = wheels_root.GetPath().pathString + f"/{name}"
            wheel = UsdGeom.Cylinder.Define(stage, wpath)
            wheel.CreateRadiusAttr(float(wheel_radius_m / mpu))
            wheel.CreateHeightAttr(float(wheel_thickness_m / mpu))

            wapi = UsdGeom.XformCommonAPI(wheel)
            wapi.SetTranslate(Gf.Vec3d(float(wx / mpu), float(wy / mpu), float(wz_m / mpu)))
            wapi.SetRotate(wheel_rot, UsdGeom.XformCommonAPI.RotationOrderXYZ)

        root_prim = stage.GetPrimAtPath(path)
        root_prim.SetCustomDataByKey("parked_car", True)
        return path

    # -------- goal ring (visual) + trigger collider (no RB) --------

    def _spawn_goal_ring_with_trigger(
        self,
        goal_root_path: str,
        *,
        center_m: Tuple[float, float, float],
        radius_m: float,
        ring_tube_radius_m: float = 0.12,
        trigger_height_m: float = 0.6,
        trigger_radius_scale: float = 1.05,
        color_rgb: Tuple[float, float, float] = (0.1, 0.9, 0.2),
        z_on_ground: Optional[float] = 0.0,   # enforce z
    ) -> str:
        """
        Creates:
          - Visual ring (green) at z=z_on_ground (or center_m[2] if None)
          - Trigger collider cylinder (invisible) that overlaps when car reaches goal
        We DO NOT rely on PhysX trigger report API; collision is optional for debug only.
        """
        stage = self.stage
        mpu = _meters_per_unit(stage)

        cx, cy, cz = map(float, center_m)
        if z_on_ground is not None:
            cz = float(z_on_ground)

        root = UsdGeom.Xform.Define(stage, goal_root_path)
        UsdGeom.XformCommonAPI(root).SetTranslate(Gf.Vec3d(cx / mpu, cy / mpu, cz / mpu))

        # Visual ring:
        # Prefer UsdGeom.Torus if available; otherwise fall back to a thin cylinder "disk".
        ring_path = goal_root_path + "/Ring"
        if hasattr(UsdGeom, "Torus"):
            torus = UsdGeom.Torus.Define(stage, ring_path)
            # Torus attrs are in stage units; keep it simple with scale.
            torus.CreateMajorRadiusAttr(float(radius_m / mpu))
            torus.CreateMinorRadiusAttr(float(ring_tube_radius_m / mpu))
            # Rotate so it lies on ground (torus is usually around Z already, but keep consistent)
            UsdGeom.XformCommonAPI(torus).SetRotate(Gf.Vec3f(0.0, 90.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
            ring_prim = torus.GetPrim()
        else:
            # Fallback visual: very thin cylinder
            cyl = UsdGeom.Cylinder.Define(stage, ring_path)
            cyl.CreateRadiusAttr(float(radius_m / mpu))
            cyl.CreateHeightAttr(float(0.05 / mpu))
            UsdGeom.XformCommonAPI(cyl).SetRotate(Gf.Vec3f(0.0, 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
            ring_prim = cyl.GetPrim()

        # green color
        try:
            gprim = UsdGeom.Gprim(ring_prim)
            gprim.CreateDisplayColorAttr([Gf.Vec3f(*map(float, color_rgb))])
        except Exception:
            pass

        # Trigger collider (no rigid body)
        trig_path = goal_root_path + "/Trigger"
        trig = UsdGeom.Cylinder.Define(stage, trig_path)
        trig.CreateRadiusAttr(float((radius_m * trigger_radius_scale) / mpu))
        trig.CreateHeightAttr(float(trigger_height_m / mpu))
        # align vertical (default cylinder axis is +Z)
        # Move trigger so it spans above ground:
        UsdGeom.XformCommonAPI(trig).SetTranslate(Gf.Vec3d(0.0, 0.0, float((0.5 * trigger_height_m) / mpu)))

        trig_prim = trig.GetPrim()
        UsdPhysics.CollisionAPI.Apply(trig_prim)
        UsdGeom.Imageable(trig_prim).MakeInvisible()

        try:
            UsdPhysics.CollisionAPI(trig_prim).CreateCollisionEnabledAttr(False)
        except Exception:
            pass

        # Tag
        goal_prim = stage.GetPrimAtPath(goal_root_path)
        goal_prim.SetCustomDataByKey("is_goal", True)
        goal_prim.SetCustomDataByKey("goal_radius_m", float(radius_m))
        goal_prim.SetCustomDataByKey("goal_center_local_m", (float(cx), float(cy), float(cz)))
        world_center = _get_world_translation_m(stage, goal_root_path)
        if world_center is None:
            world_center = (float(cx), float(cy), float(cz))
        goal_prim.SetCustomDataByKey("goal_center_m", tuple(map(float, world_center)))

        return goal_root_path

    def _spawn_vehicle_trigger(
        self,
        vehicle_root_path: str,
        *,
        offset_m: Tuple[float, float, float],
        size_m: Tuple[float, float, float],
        enable: bool = True,
        script_enable: bool = True,
    ) -> None:
        if not enable:
            return

        stage = self.stage
        vehicle_chassis_path = f"{vehicle_root_path}/Vehicle"
        veh_prim = stage.GetPrimAtPath(vehicle_chassis_path)
        if not veh_prim.IsValid():
            return

        trig_root = f"{vehicle_chassis_path}/VehicleTrigger"
        trig_prim = stage.GetPrimAtPath(trig_root)
        if not trig_prim.IsValid():
            UsdGeom.Xform.Define(stage, trig_root)
            trig_prim = stage.GetPrimAtPath(trig_root)

        cube_path = f"{trig_root}/Cube"
        cube = UsdGeom.Cube.Define(stage, cube_path)
        cube.GetSizeAttr().Set(1.0)

        mpu = _meters_per_unit(stage)
        xform_api = UsdGeom.XformCommonAPI(cube)
        xform_api.SetTranslate(
            Gf.Vec3d(
                float(offset_m[0]) / mpu,
                float(offset_m[1]) / mpu,
                float(offset_m[2]) / mpu,
            )
        )
        xform_api.SetScale(
            Gf.Vec3f(
                float(size_m[0]) / mpu,
                float(size_m[1]) / mpu,
                float(size_m[2]) / mpu,
            )
        )

        UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        if PhysxSchema is not None:
            try:
                PhysxSchema.PhysxTriggerAPI.Apply(cube.GetPrim())
            except Exception:
                pass
            if script_enable:
                try:
                    trigger_script = Path(__file__).resolve().parent / "trigger_road_contact.py"
                    trig_api = PhysxSchema.PhysxTriggerAPI.Apply(cube.GetPrim())
                    trig_api.CreateEnterScriptTypeAttr().Set(PhysxSchema.Tokens.scriptFile)
                    trig_api.CreateOnEnterScriptAttr().Set(str(trigger_script))
                    trig_api.CreateLeaveScriptTypeAttr().Set(PhysxSchema.Tokens.scriptFile)
                    trig_api.CreateOnLeaveScriptAttr().Set(str(trigger_script))
                except Exception:
                    pass

        try:
            cd = veh_prim.GetCustomData()
        except Exception:
            cd = {}
        if isinstance(cd, dict):
            agent_id = cd.get("agent_id", None)
            if agent_id is not None:
                cube.GetPrim().SetCustomDataByKey("vehicle_trigger_owner_id", int(agent_id))
        cube.GetPrim().SetCustomDataByKey("vehicle_trigger", True)

    # -------- agents + goals --------

    def build_agents_with_goals(
        self,
        cfg: Dict[str, Any],
        max_agents: Optional[int] = None,
        spawn_z_m: float = 1.0,
        goal_radius_m: float = 3.0,
        require_goal_in_bounds: bool = True,

        # parked-car knobs:
        parked_if_start_in_goal: bool = True,
        skip_if_start_in_goal: bool = False,
        start_goal_thresh_m: Optional[float] = None,  # if None uses goal_radius_m
        parked_ground_z_m: float = 0.0,
        parked_chassis_size_m: Tuple[float, float, float] = (4.0, 2.0, 1.0),
        parked_wheel_radius_m: float = 0.35,
        parked_wheel_thickness_m: float = 0.15,
        parked_wheel_inset_x_m: float = 0.35,
        parked_wheel_inset_y_m: float = 0.25,
        parked_ground_clearance_m: float = 0.25,

        # goal ring knobs:
        goal_ring_z_m: float = 0.0,  # enforce on ground
        goal_ring_tube_radius_m: float = 0.12,
        goal_trigger_height_m: float = 0.6,

        # vehicle trigger knobs:
        vehicle_trigger_enable: bool = False,
        vehicle_trigger_offset_m: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        vehicle_trigger_size_m: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        vehicle_trigger_script_enable: bool = True,
    ) -> None:
        agents = (cfg.get("agents", {}) or {}).get("items", []) or []
        kept = 0
        skipped = 0

        mgr = _goal_mgr(self.stage)  # ensure singleton

        for idx, a in enumerate(agents):
            if max_agents is not None and kept >= int(max_agents):
                break

            s = a.get("start", {}) or {}
            e = a.get("end", {}) or {}

            sx = _safe_float(s.get("x", None))
            sy = _safe_float(s.get("y", None))
            sz = _safe_float(s.get("z", 0.0), 0.0)
            syaw = _safe_float(s.get("yaw", 0.0), 0.0)  # radians

            ex = _safe_float(e.get("x", None))
            ey = _safe_float(e.get("y", None))
            ez = _safe_float(e.get("z", 0.0), 0.0)

            if sx is None or sy is None or ex is None or ey is None:
                skipped += 1
                continue

            # LOCAL coords
            sx, sy, sz = self._to_local_xyz(sx, sy, sz)
            ex, ey, ez = self._to_local_xyz(ex, ey, ez)

            # bounds gate
            if not self._in_bounds_xy(float(sx), float(sy)):
                skipped += 1
                continue
            if require_goal_in_bounds and (not self._in_bounds_xy(float(ex), float(ey))):
                skipped += 1
                continue

            thresh = float(start_goal_thresh_m) if start_goal_thresh_m is not None else float(goal_radius_m)
            start_in_goal = self._is_start_within_goal(sx, sy, sz, ex, ey, ez, thresh)

            if skip_if_start_in_goal and start_in_goal:
                skipped += 1
                continue

            agent_id = _safe_int(a.get("agent_id", idx), idx)
            agent_path = f"{self.agents_root}/Agent_{kept:04d}_id{agent_id}"
            UsdGeom.Xform.Define(self.stage, agent_path)
            agent_prim = self.stage.GetPrimAtPath(agent_path)
            agent_prim.SetCustomDataByKey("agent_id", int(agent_id))
            agent_prim.SetCustomDataByKey("kept_idx", int(kept))
            agent_prim.SetCustomDataByKey("start_local_m", (float(sx), float(sy), float(sz)))
            agent_prim.SetCustomDataByKey("start_yaw_deg", float(np.degrees(float(syaw))))
            agent_prim.SetCustomDataByKey("goal_local_m", (float(ex), float(ey), float(ez)))
            agent_prim.SetCustomDataByKey("start_in_goal", bool(start_in_goal))

            yaw_deg = float(np.degrees(float(syaw)))

            # start_in_goal => parked car (NO GOAL)
            if parked_if_start_in_goal and start_in_goal:
                parked_path = f"{agent_path}/ParkedCar"
                self._spawn_parked_car_visual(
                    parked_path,
                    position_m=(float(sx), float(sy), float(parked_ground_z_m)),
                    yaw_deg=yaw_deg,
                    chassis_size_m=parked_chassis_size_m,
                    wheel_radius_m=parked_wheel_radius_m,
                    wheel_thickness_m=parked_wheel_thickness_m,
                    wheel_inset_x_m=parked_wheel_inset_x_m,
                    wheel_inset_y_m=parked_wheel_inset_y_m,
                    ground_clearance_m=parked_ground_clearance_m,
                )
                prim_outer = self.stage.GetPrimAtPath(parked_path)
                prim_outer.SetCustomDataByKey("agent_id", int(agent_id))
                prim_outer.SetCustomDataByKey("agent_type", int(_safe_int(a.get("agent_type", -1), -1)))
                prim_outer.SetCustomDataByKey("track_idx", int(_safe_int(a.get("track_idx", kept), kept)))
                prim_outer.SetCustomDataByKey("start_in_goal", True)
                prim_outer.SetCustomDataByKey("controllable", False)
                self._set_vehicle_size_metadata(
                    prim_outer,
                    size_m=(
                        float(parked_chassis_size_m[0]),
                        float(parked_chassis_size_m[1]),
                        float(parked_chassis_size_m[2]),
                    ),
                )

                kept += 1
                continue

            # otherwise: full PhysX vehicle
            veh_parent = f"{agent_path}/Vehicle_Parent"
            veh_outer = self._spawn_vehicle_wizard_under(
                veh_parent,
                position_m=(float(sx), float(sy), float(spawn_z_m)),
                yaw_deg=yaw_deg,
            )
            if veh_outer is None:
                skipped += 1
                continue

            prim_outer = self.stage.GetPrimAtPath(veh_outer)
            prim_outer.SetCustomDataByKey("agent_id", int(agent_id))
            prim_outer.SetCustomDataByKey("agent_type", int(_safe_int(a.get("agent_type", -1), -1)))
            prim_outer.SetCustomDataByKey("track_idx", int(_safe_int(a.get("track_idx", kept), kept)))
            prim_outer.SetCustomDataByKey("start_in_goal", False)
            prim_outer.SetCustomDataByKey("controllable", True)
            self._set_vehicle_size_metadata(
                prim_outer,
                size_m=self._infer_vehicle_size_m(veh_outer),
            )
            prim_outer.SetCustomDataByKey("road_contact_types", Vt.IntArray())
            prim_outer.SetCustomDataByKey("vehicle_contact_ids", Vt.IntArray())
            prim_outer.SetCustomDataByKey("vehicle_collided", False)
            if PhysxSchema is not None:
                try:
                    rb_prim = self.stage.GetPrimAtPath(f"{veh_outer}/Vehicle")
                    if rb_prim.IsValid():
                        _apply_contact_report_to_colliders(rb_prim)
                except Exception:
                    pass

            self._spawn_vehicle_trigger(
                veh_outer,
                offset_m=vehicle_trigger_offset_m,
                size_m=vehicle_trigger_size_m,
                enable=bool(vehicle_trigger_enable),
                script_enable=bool(vehicle_trigger_script_enable),
            )

            # GOAL ring + trigger collider (no RB), enforced z=goal_ring_z_m
            goal_path = f"{self.goals_root}/Goal_{kept:04d}_id{agent_id}"
            self._spawn_goal_ring_with_trigger(
                goal_path,
                center_m=(float(ex), float(ey), float(ez)),
                radius_m=float(goal_radius_m),
                ring_tube_radius_m=float(goal_ring_tube_radius_m),
                trigger_height_m=float(goal_trigger_height_m),
                z_on_ground=float(goal_ring_z_m),
            )

            # register with polling manager (correct car only)
            world_center = _get_world_translation_m(self.stage, goal_path)
            if world_center is None:
                world_center = (float(ex), float(ey), float(goal_ring_z_m))
            mgr.add_goal(
                goal_path,
                center_m=tuple(map(float, world_center)),
                radius_m=float(goal_radius_m),
                car_root_path=str(veh_outer),
                agent_id=int(agent_id),
            )

            kept += 1

    def respawn_agent_with_goal(
        self,
        *,
        kept_idx: int,
        agent_id: int,
        start_local_m: Tuple[float, float, float],
        start_yaw_deg: float,
        goal_local_m: Tuple[float, float, float],
        start_in_goal: bool,
        spawn_z_m: float = 1.0,
        # parked-car knobs:
        parked_ground_z_m: float = 0.0,
        parked_chassis_size_m: Tuple[float, float, float] = (4.0, 2.0, 1.0),
        parked_wheel_radius_m: float = 0.35,
        parked_wheel_thickness_m: float = 0.15,
        parked_wheel_inset_x_m: float = 0.35,
        parked_wheel_inset_y_m: float = 0.25,
        parked_ground_clearance_m: float = 0.25,
        # goal ring knobs:
        goal_radius_m: float = 3.0,
        goal_ring_z_m: float = 0.0,
        goal_ring_tube_radius_m: float = 0.12,
        goal_trigger_height_m: float = 0.6,

        # vehicle trigger knobs:
        vehicle_trigger_enable: bool = False,
        vehicle_trigger_offset_m: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        vehicle_trigger_size_m: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        vehicle_trigger_script_enable: bool = True,
    ) -> None:
        agent_path = f"{self.agents_root}/Agent_{int(kept_idx):04d}_id{int(agent_id)}"
        UsdGeom.Xform.Define(self.stage, agent_path)
        agent_prim = self.stage.GetPrimAtPath(agent_path)
        agent_prim.SetCustomDataByKey("agent_id", int(agent_id))
        agent_prim.SetCustomDataByKey("kept_idx", int(kept_idx))
        agent_prim.SetCustomDataByKey(
            "start_local_m",
            (float(start_local_m[0]), float(start_local_m[1]), float(start_local_m[2])),
        )
        agent_prim.SetCustomDataByKey("start_yaw_deg", float(start_yaw_deg))
        agent_prim.SetCustomDataByKey(
            "goal_local_m",
            (float(goal_local_m[0]), float(goal_local_m[1]), float(goal_local_m[2])),
        )
        agent_prim.SetCustomDataByKey("start_in_goal", bool(start_in_goal))

        if bool(start_in_goal):
            parked_path = f"{agent_path}/ParkedCar"
            self._spawn_parked_car_visual(
                parked_path,
                position_m=(float(start_local_m[0]), float(start_local_m[1]), float(parked_ground_z_m)),
                yaw_deg=float(start_yaw_deg),
                chassis_size_m=parked_chassis_size_m,
                wheel_radius_m=parked_wheel_radius_m,
                wheel_thickness_m=parked_wheel_thickness_m,
                wheel_inset_x_m=parked_wheel_inset_x_m,
                wheel_inset_y_m=parked_wheel_inset_y_m,
                ground_clearance_m=parked_ground_clearance_m,
            )
            prim_outer = self.stage.GetPrimAtPath(parked_path)
            prim_outer.SetCustomDataByKey("agent_id", int(agent_id))
            prim_outer.SetCustomDataByKey("start_in_goal", True)
            prim_outer.SetCustomDataByKey("controllable", False)
            self._set_vehicle_size_metadata(
                prim_outer,
                size_m=(
                    float(parked_chassis_size_m[0]),
                    float(parked_chassis_size_m[1]),
                    float(parked_chassis_size_m[2]),
                ),
            )
            return

        veh_parent = f"{agent_path}/Vehicle_Parent"
        veh_outer = self._spawn_vehicle_wizard_under(
            veh_parent,
            position_m=(float(start_local_m[0]), float(start_local_m[1]), float(spawn_z_m)),
            yaw_deg=float(start_yaw_deg),
        )
        if veh_outer is None:
            return

        prim_outer = self.stage.GetPrimAtPath(veh_outer)
        prim_outer.SetCustomDataByKey("agent_id", int(agent_id))
        prim_outer.SetCustomDataByKey("start_in_goal", False)
        prim_outer.SetCustomDataByKey("controllable", True)
        self._set_vehicle_size_metadata(
            prim_outer,
            size_m=self._infer_vehicle_size_m(veh_outer),
        )
        prim_outer.SetCustomDataByKey("road_contact_types", Vt.IntArray())
        prim_outer.SetCustomDataByKey("vehicle_contact_ids", Vt.IntArray())
        prim_outer.SetCustomDataByKey("vehicle_collided", False)
        if PhysxSchema is not None:
            try:
                rb_prim = self.stage.GetPrimAtPath(f"{veh_outer}/Vehicle")
                if rb_prim.IsValid():
                    _apply_contact_report_to_colliders(rb_prim)
            except Exception:
                pass

        self._spawn_vehicle_trigger(
            veh_outer,
            offset_m=vehicle_trigger_offset_m,
            size_m=vehicle_trigger_size_m,
            enable=bool(vehicle_trigger_enable),
            script_enable=bool(vehicle_trigger_script_enable),
        )

        goal_path = f"{self.goals_root}/Goal_{int(kept_idx):04d}_id{int(agent_id)}"
        self._spawn_goal_ring_with_trigger(
            goal_path,
            center_m=(float(goal_local_m[0]), float(goal_local_m[1]), float(goal_local_m[2])),
            radius_m=float(goal_radius_m),
            ring_tube_radius_m=float(goal_ring_tube_radius_m),
            trigger_height_m=float(goal_trigger_height_m),
            z_on_ground=float(goal_ring_z_m),
        )

        mgr = _goal_mgr(self.stage)
        world_center = _get_world_translation_m(self.stage, goal_path)
        if world_center is None:
            world_center = (float(goal_local_m[0]), float(goal_local_m[1]), float(goal_ring_z_m))
        mgr.add_goal(
            goal_path,
            center_m=tuple(map(float, world_center)),
            radius_m=float(goal_radius_m),
            car_root_path=str(veh_outer),
            agent_id=int(agent_id),
        )

    # -------- entry point --------

    def build_from_json(
        self,
        json_path: Union[str, Path],
        *,
        max_agents: int = 50,

        # road params:
        polyline_reduction_area: float = 0.0,
        min_points_for_reduction: int = 10,
        jump_break_m: float = 3.0,
        seg_width: float = 0.10,
        seg_height: float = 0.10,
        z_lift: float = 0.02,
        flatten_road_z: bool = True,
        road_z_m: float = 0.0,
        enable_segment_collision: bool = False,
        trigger_enable: bool = True,
        trigger_height_m: float = 1.0,
        trigger_width_scale: float = 1.0,
        trigger_offset_z_m: float = 0.5,
        trigger_match_segment: bool = True,
        trigger_script_enable: bool = True,
        allowed_road_types: Optional[Sequence[int]] = None,
        road_render_mode: str = "point_instancer",

        # agent params:
        spawn_z_m: float = 1.0,
        goal_radius_m: float = 3.0,

        # parked car params:
        parked_if_start_in_goal: bool = True,
        skip_if_start_in_goal: bool = False,
        start_goal_thresh_m: Optional[float] = None,
        parked_ground_z_m: float = 0.0,
        parked_chassis_size_m: Tuple[float, float, float] = (4.0, 2.0, 1.0),
        parked_wheel_radius_m: float = 0.35,
        parked_wheel_thickness_m: float = 0.15,
        parked_wheel_inset_x_m: float = 0.35,
        parked_wheel_inset_y_m: float = 0.25,
        parked_ground_clearance_m: float = 0.25,

        # goal ring params:
        goal_ring_z_m: float = 0.0,
        goal_ring_tube_radius_m: float = 0.12,
        goal_trigger_height_m: float = 0.6,

        # vehicle trigger params:
        vehicle_trigger_enable: bool = False,
        vehicle_trigger_offset_m: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        vehicle_trigger_size_m: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        vehicle_trigger_script_enable: bool = True,
    ) -> None:
        p = Path(json_path).expanduser().resolve()
        with p.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        root_prim = self.stage.GetPrimAtPath(self.world_root)
        root_prim.SetCustomDataByKey("scene_json", str(p))
        root_prim.SetCustomDataByKey("origin_mode", str(self.origin_mode))
        root_prim.SetCustomDataByKey("bounds_w_m", float(self.bounds.width_m))
        root_prim.SetCustomDataByKey("bounds_l_m", float(self.bounds.length_m))

        self.build_road_segments_grouped(
            cfg,
            polyline_reduction_area=polyline_reduction_area,
            min_points_for_reduction=min_points_for_reduction,
            jump_break_m=jump_break_m,
            seg_width=seg_width,
            seg_height=seg_height,
            z_lift=z_lift,
            flatten_road_z=flatten_road_z,
            road_z_m=road_z_m,
            enable_segment_collision=enable_segment_collision,
            trigger_enable=trigger_enable,
            trigger_height_m=trigger_height_m,
            trigger_width_scale=trigger_width_scale,
            trigger_offset_z_m=trigger_offset_z_m,
            trigger_match_segment=trigger_match_segment,
            trigger_script_enable=trigger_script_enable,
            allowed_road_types=allowed_road_types,
            road_render_mode=road_render_mode,
        )
        self.build_agents_with_goals(
            cfg,
            max_agents=max_agents,
            spawn_z_m=spawn_z_m,
            goal_radius_m=goal_radius_m,
            require_goal_in_bounds=True,

            parked_if_start_in_goal=parked_if_start_in_goal,
            skip_if_start_in_goal=skip_if_start_in_goal,
            start_goal_thresh_m=start_goal_thresh_m,
            parked_ground_z_m=parked_ground_z_m,
            parked_chassis_size_m=parked_chassis_size_m,
            parked_wheel_radius_m=parked_wheel_radius_m,
            parked_wheel_thickness_m=parked_wheel_thickness_m,
            parked_wheel_inset_x_m=parked_wheel_inset_x_m,
            parked_wheel_inset_y_m=parked_wheel_inset_y_m,
            parked_ground_clearance_m=parked_ground_clearance_m,

            goal_ring_z_m=goal_ring_z_m,
            goal_ring_tube_radius_m=goal_ring_tube_radius_m,
            goal_trigger_height_m=goal_trigger_height_m,

            vehicle_trigger_enable=vehicle_trigger_enable,
            vehicle_trigger_offset_m=vehicle_trigger_offset_m,
            vehicle_trigger_size_m=vehicle_trigger_size_m,
            vehicle_trigger_script_enable=vehicle_trigger_script_enable,
        )


# ----------------------------
# Chocolate-bar multi-world constructor (GRID of mini-worlds)
# ----------------------------

@dataclass
class GridLayout:
    world_size_m: Tuple[float, float] = (200.0, 200.0)
    padding_m: float = 0.0
    grid_cols: int = 5
    base_z_m: float = 0.0


class ChocolateBarConstructor:
    """
    Places N mini-world roots in a grid and spawns each from a JSON.
    IMPORTANT: Per-world filtering is LOCAL, grid translation is on the root prim only.
    """

    def __init__(
        self,
        stage: Optional[Usd.Stage] = None,
        root_container: str = "/World/MiniWorlds",
        layout: GridLayout = GridLayout(),
        origin_mode: str = "center",
        origin_center_mode: str = "mean",
    ):
        self.stage = stage or omni.usd.get_context().get_stage()
        self.root_container = root_container
        self.layout = layout
        self.origin_mode = origin_mode
        self.origin_center_mode = str(origin_center_mode).strip().lower()

        UsdGeom.Xform.Define(self.stage, self.root_container)

    def _world_root_path(self, i: int) -> str:
        return f"{self.root_container}/world_{i:03d}"

    def _world_translation(self, i: int) -> Tuple[float, float, float]:
        cols = int(self.layout.grid_cols)
        sx, sy = self.layout.world_size_m
        pitch_x = float(sx + self.layout.padding_m)
        pitch_y = float(sy + self.layout.padding_m)
        r = i // cols
        c = i % cols
        return (float(c * pitch_x), float(r * pitch_y), float(self.layout.base_z_m))

    def _set_root_translation(self, prim_path: str, txyz: Tuple[float, float, float]) -> None:
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            UsdGeom.Xform.Define(self.stage, prim_path)
            prim = self.stage.GetPrimAtPath(prim_path)
        xform = UsdGeom.XformCommonAPI(prim)
        xform.SetTranslate(Gf.Vec3d(*txyz))

    def clear_all(self) -> None:
        self.stage.RemovePrim(Sdf.Path(self.root_container))
        UsdGeom.Xform.Define(self.stage, self.root_container)
        
    def build_global_boundary(
        self,
        world_count: int,
        *,
        add_floor: bool = False,
        thickness_m: float = 0.2,
        z_m: float = 0.0,
        color_srgb: Tuple[float, float, float] = (0.15, 0.15, 0.18),
        emissive_strength: float = 0.05,
        add_perimeter_walls: bool = True,
        wall_height_m: float = 3.0,
        wall_thickness_m: float = 0.4,
        add_grid_lines: bool = True,
        line_thickness_m: float = 0.15,
        line_height_m: float = 0.05,
        line_color_srgb: Tuple[float, float, float] = (0.35, 0.35, 0.4),
    ) -> str:
        stage = self.stage
        mpu = _meters_per_unit(stage)
    
        cols = int(self.layout.grid_cols)
        pitch_x = float(self.layout.world_size_m[0] + self.layout.padding_m)
        pitch_y = float(self.layout.world_size_m[1] + self.layout.padding_m)
        sx = float(self.layout.world_size_m[0])
        sy = float(self.layout.world_size_m[1])
    
        rows = int(math.ceil(float(world_count) / float(cols))) if world_count > 0 else 1
    
        # Worlds are centered at their root translation; local bounds are [-sx/2, +sx/2] etc.
        min_x = -0.5 * sx
        max_x = (cols - 1) * pitch_x + 0.5 * sx
        min_y = -0.5 * sy
        max_y = (rows - 1) * pitch_y + 0.5 * sy
    
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
    
        root = f"{self.root_container}/__GlobalBoundary"
        UsdGeom.Xform.Define(stage, root)

        floor_w = (max_x - min_x)
        floor_l = (max_y - min_y)
        floor_h = float(thickness_m)

        mat = _get_or_create_preview_material(
            stage,
            root + "/Materials/FloorMat",
            rgb_srgb=color_srgb,
            emissive_strength=emissive_strength,
        )

        if add_floor:
            floor_path = root + "/Floor"
            floor = UsdGeom.Cube.Define(stage, floor_path)
            floor.GetSizeAttr().Set(1.0)

            fapi = UsdGeom.XformCommonAPI(floor)
            fapi.SetTranslate(Gf.Vec3d(cx / mpu, cy / mpu, (z_m - 0.5 * floor_h) / mpu))
            fapi.SetScale(Gf.Vec3f(floor_w / mpu, floor_l / mpu, floor_h / mpu))

            UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
            _bind_material(floor.GetPrim(), mat)
    
        # --- perimeter walls ---
        if add_perimeter_walls:
            walls_root = root + "/Walls"
            UsdGeom.Xform.Define(stage, walls_root)
    
            def _wall(name: str, x: float, y: float, w: float, l: float):
                p = walls_root + f"/{name}"
                cube = UsdGeom.Cube.Define(stage, p)
                cube.GetSizeAttr().Set(1.0)
                api = UsdGeom.XformCommonAPI(cube)
                api.SetTranslate(Gf.Vec3d(x / mpu, y / mpu, (z_m + 0.5 * wall_height_m) / mpu))
                api.SetScale(Gf.Vec3f(w / mpu, l / mpu, wall_height_m / mpu))
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                _bind_material(cube.GetPrim(), mat)  # same mat or make a wall mat
    
            # left/right walls (thin in X, long in Y)
            _wall("WallLeft",  min_x - 0.5 * wall_thickness_m, cy, wall_thickness_m, floor_l + 2 * wall_thickness_m)
            _wall("WallRight", max_x + 0.5 * wall_thickness_m, cy, wall_thickness_m, floor_l + 2 * wall_thickness_m)
            # bottom/top walls (thin in Y, long in X)
            _wall("WallBottom", cx, min_y - 0.5 * wall_thickness_m, floor_w + 2 * wall_thickness_m, wall_thickness_m)
            _wall("WallTop",    cx, max_y + 0.5 * wall_thickness_m, floor_w + 2 * wall_thickness_m, wall_thickness_m)
    
        # --- grid lines at world boundaries (cheap: boxes) ---
        if add_grid_lines:
            lines_root = root + "/GridLines"
            UsdGeom.Xform.Define(stage, lines_root)
            line_mat = _get_or_create_preview_material(stage, root + "/Materials/GridLineMat",
                                                       rgb_srgb=line_color_srgb,
                                                       emissive_strength=0.08)
    
            # vertical lines between columns
            for c in range(1, cols):
                x = (c * pitch_x) - 0.5 * sx  # boundary between worlds in X
                p = lines_root + f"/V_{c:02d}"
                ln = UsdGeom.Cube.Define(stage, p)
                ln.GetSizeAttr().Set(1.0)
                api = UsdGeom.XformCommonAPI(ln)
                api.SetTranslate(Gf.Vec3d(x / mpu, cy / mpu, (z_m + 0.5 * line_height_m) / mpu))
                api.SetScale(Gf.Vec3f(line_thickness_m / mpu, floor_l / mpu, line_height_m / mpu))
                UsdPhysics.CollisionAPI.Apply(ln.GetPrim())
                _bind_material(ln.GetPrim(), line_mat)
    
            # horizontal lines between rows
            for r in range(1, rows):
                y = (r * pitch_y) - 0.5 * sy
                p = lines_root + f"/H_{r:02d}"
                ln = UsdGeom.Cube.Define(stage, p)
                ln.GetSizeAttr().Set(1.0)
                api = UsdGeom.XformCommonAPI(ln)
                api.SetTranslate(Gf.Vec3d(cx / mpu, y / mpu, (z_m + 0.5 * line_height_m) / mpu))
                api.SetScale(Gf.Vec3f(floor_w / mpu, line_thickness_m / mpu, line_height_m / mpu))
                UsdPhysics.CollisionAPI.Apply(ln.GetPrim())
                _bind_material(ln.GetPrim(), line_mat)
    
        return root

    def build(
        self,
        json_paths: Sequence[Union[str, Path]],
        world_count: int,
        *,
        bounds_size_m: float = 200.0,
        max_agents_per_world: int = 50,

        # road params:
        polyline_reduction_area: float = 0.0,
        min_points_for_reduction: int = 10,
        jump_break_m: float = 3.0,
        seg_width: float = 0.10,
        seg_height: float = 0.10,
        z_lift: float = 0.02,
        flatten_road_z: bool = True,
        road_z_m: float = 0.0,
        enable_segment_collision: bool = False,
        trigger_enable: bool = False,
        trigger_height_m: float = 1.0,
        trigger_width_scale: float = 1.0,
        trigger_offset_z_m: float = 0.5,
        trigger_match_segment: bool = True,
        trigger_script_enable: bool = True,

        # agent params:
        spawn_z_m: float = 1.0,
        goal_radius_m: float = 3.0,

        # parked car params:
        parked_if_start_in_goal: bool = True,
        skip_if_start_in_goal: bool = False,
        start_goal_thresh_m: Optional[float] = None,
        parked_ground_z_m: float = 0.0,
        parked_chassis_size_m: Tuple[float, float, float] = (4.0, 2.0, 1.0),
        parked_wheel_radius_m: float = 0.35,
        parked_wheel_thickness_m: float = 0.15,
        parked_wheel_inset_x_m: float = 0.35,
        parked_wheel_inset_y_m: float = 0.25,
        parked_ground_clearance_m: float = 0.25,

        # goal ring params:
        goal_ring_z_m: float = 0.0,
        goal_ring_tube_radius_m: float = 0.12,
        goal_trigger_height_m: float = 0.6,

        # vehicle trigger params:
        vehicle_trigger_enable: bool = False,
        vehicle_trigger_offset_m: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        vehicle_trigger_size_m: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        vehicle_trigger_script_enable: bool = True,
        allowed_road_types: Optional[Sequence[int]] = None,
        road_render_mode: str = "point_instancer",
    ) -> None:
        json_list = [str(Path(p).expanduser().resolve()) for p in json_paths]
        if len(json_list) == 0:
            raise ValueError("json_paths is empty")

        import time as _time
        _n = int(world_count)
        _t0 = _time.time()
        _milestones = set(max(0, int(_n * p / 10) - 1) for p in range(1, 11))
        _milestones.add(0)
        _milestones.add(_n - 1)
        print(f"[SceneFactory] Building {_n} worlds...")
        for i in range(_n):
            root = self._world_root_path(i)
            UsdGeom.Xform.Define(self.stage, root)

            # place root in grid (chocolate layout)
            txyz = self._world_translation(i)
            self._set_root_translation(root, txyz)

            bounds = LocalBounds(
                width_m=float(bounds_size_m),
                length_m=float(bounds_size_m),
                origin_xy=(0.0, 0.0),
            )

            builder = WaymoJsonMiniWorldBuilder(
                stage=self.stage,
                world_root=root,
                bounds=bounds,
                origin_mode=self.origin_mode,
                origin_center_mode=self.origin_center_mode,
            )

            if i in _milestones:
                _pct = int(100 * (i + 1) / max(_n, 1))
                _elapsed = _time.time() - _t0
                print(f"[SceneFactory]   {i + 1}/{_n} worlds built ({_pct}%)  {_elapsed:.0f}s elapsed")
            json_path = json_list[i % len(json_list)]
            builder.build_from_json(
                json_path,
                max_agents=max_agents_per_world,

                # road
                polyline_reduction_area=polyline_reduction_area,
                min_points_for_reduction=min_points_for_reduction,
                jump_break_m=jump_break_m,
                seg_width=seg_width,
                seg_height=seg_height,
                z_lift=z_lift,
                flatten_road_z=flatten_road_z,
                road_z_m=road_z_m,
                enable_segment_collision=enable_segment_collision,
                trigger_enable=trigger_enable,
                trigger_height_m=trigger_height_m,
                trigger_width_scale=trigger_width_scale,
                trigger_offset_z_m=trigger_offset_z_m,
                trigger_match_segment=trigger_match_segment,
                trigger_script_enable=trigger_script_enable,
                allowed_road_types=allowed_road_types,
                road_render_mode=road_render_mode,

                # agents
                spawn_z_m=spawn_z_m,
                goal_radius_m=goal_radius_m,

                # parked cars
                parked_if_start_in_goal=parked_if_start_in_goal,
                skip_if_start_in_goal=skip_if_start_in_goal,
                start_goal_thresh_m=start_goal_thresh_m,
                parked_ground_z_m=parked_ground_z_m,
                parked_chassis_size_m=parked_chassis_size_m,
                parked_wheel_radius_m=parked_wheel_radius_m,
                parked_wheel_thickness_m=parked_wheel_thickness_m,
                parked_wheel_inset_x_m=parked_wheel_inset_x_m,
                parked_wheel_inset_y_m=parked_wheel_inset_y_m,
                parked_ground_clearance_m=parked_ground_clearance_m,

                # goal ring
                goal_ring_z_m=goal_ring_z_m,
                goal_ring_tube_radius_m=goal_ring_tube_radius_m,
                goal_trigger_height_m=goal_trigger_height_m,

                # vehicle trigger
                vehicle_trigger_enable=vehicle_trigger_enable,
                vehicle_trigger_offset_m=vehicle_trigger_offset_m,
                vehicle_trigger_size_m=vehicle_trigger_size_m,
                vehicle_trigger_script_enable=vehicle_trigger_script_enable,
            )
        _elapsed = _time.time() - _t0
        print(f"[SceneFactory] Done. Built {_n} worlds in {_elapsed:.1f}s")
        self.build_global_boundary(
            world_count=int(world_count),
            add_floor=False,
            thickness_m=0.2,
            z_m=float(self.layout.base_z_m),
            add_perimeter_walls=True,
            add_grid_lines=True,
        )
