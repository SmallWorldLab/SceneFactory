from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tomllib
import traceback
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ACCEL_MIN = 0.0
ACCEL_MAX = 1.0
STEER_MIN = -1.0
STEER_MAX = 1.0
BRAKE_MIN = 0.0
BRAKE_MAX = 1.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _yaw_from_basis_vectors(forward_world: Sequence[float]) -> float:
    fx = float(forward_world[0])
    fy = float(forward_world[1])
    return math.atan2(fy, fx)


@dataclass(frozen=True)
class VehicleCommand:
    accelerator: float
    steering: float
    brake: float

    def clamped(self) -> "VehicleCommand":
        return VehicleCommand(
            accelerator=_clamp(self.accelerator, ACCEL_MIN, ACCEL_MAX),
            steering=_clamp(self.steering, STEER_MIN, STEER_MAX),
            brake=_clamp(self.brake, BRAKE_MIN, BRAKE_MAX),
        )

    def to_dict(self) -> Dict[str, float]:
        cmd = self.clamped()
        return {
            "accelerator": float(cmd.accelerator),
            "steering": float(cmd.steering),
            "brake": float(cmd.brake),
        }


@dataclass(frozen=True)
class CommandSegment:
    duration_s: float
    command: VehicleCommand
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_s": float(self.duration_s),
            "label": str(self.label),
            "command": self.command.clamped().to_dict(),
        }


@dataclass(frozen=True)
class SurfacePatch:
    name: str
    x_center_m: float
    y_center_m: float
    length_m: float
    width_m: float
    static_friction: float
    dynamic_friction: float
    tire_friction: float
    color_srgb: Tuple[float, float, float]

    def contains_xy(self, x_m: float, y_m: float) -> bool:
        half_length = 0.5 * float(self.length_m)
        half_width = 0.5 * float(self.width_m)
        return (
            float(self.x_center_m) - half_length <= float(x_m) <= float(self.x_center_m) + half_length
            and float(self.y_center_m) - half_width <= float(y_m) <= float(self.y_center_m) + half_width
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": str(self.name),
            "x_center_m": float(self.x_center_m),
            "y_center_m": float(self.y_center_m),
            "length_m": float(self.length_m),
            "width_m": float(self.width_m),
            "static_friction": float(self.static_friction),
            "dynamic_friction": float(self.dynamic_friction),
            "tire_friction": float(self.tire_friction),
            "color_srgb": [float(v) for v in self.color_srgb],
        }


class CommandProgram:
    def __init__(self, segments: Iterable[CommandSegment]):
        self._segments: List[CommandSegment] = list(segments)
        if not self._segments:
            raise ValueError("CommandProgram requires at least one segment.")
        for segment in self._segments:
            if float(segment.duration_s) <= 0.0:
                raise ValueError(f"Segment duration must be positive: {segment}")

    @property
    def total_duration_s(self) -> float:
        return float(sum(float(segment.duration_s) for segment in self._segments))

    def command_at(self, time_s: float) -> VehicleCommand:
        t = max(0.0, float(time_s))
        remaining = t
        for segment in self._segments:
            duration_s = float(segment.duration_s)
            if remaining < duration_s:
                return segment.command.clamped()
            remaining -= duration_s
        return self._segments[-1].command.clamped()

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [segment.to_dict() for segment in self._segments]


def _command_from_mapping(payload: Dict[str, Any]) -> VehicleCommand:
    if not isinstance(payload, dict):
        raise ValueError(f"Command payload must be a dict, got {type(payload).__name__}")
    return VehicleCommand(
        accelerator=float(payload.get("accelerator", 0.0)),
        steering=float(payload.get("steering", 0.0)),
        brake=float(payload.get("brake", 0.0)),
    ).clamped()


def command_program_from_payload(payload: Any) -> CommandProgram:
    if isinstance(payload, dict):
        segments_payload = payload.get("segments", None)
    else:
        segments_payload = payload

    if not isinstance(segments_payload, list):
        raise ValueError("Command program payload must be a list or a dict with a 'segments' list.")

    segments: List[CommandSegment] = []
    for index, segment_payload in enumerate(segments_payload):
        if not isinstance(segment_payload, dict):
            raise ValueError(f"Segment {index} must be a dict, got {type(segment_payload).__name__}")
        duration_s = float(segment_payload.get("duration_s", 0.0))
        label = str(segment_payload.get("label", ""))
        command_payload = segment_payload.get("command", segment_payload)
        command = _command_from_mapping(command_payload)
        segments.append(CommandSegment(duration_s=duration_s, command=command, label=label))

    return CommandProgram(segments)


def load_command_program(path: str | Path) -> CommandProgram:
    source_path = Path(path).expanduser().resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    return command_program_from_payload(payload)


def build_default_program() -> CommandProgram:
    return CommandProgram(
        [
            CommandSegment(1.0, VehicleCommand(0.20, 0.00, 0.00), "launch"),
            CommandSegment(3.5, VehicleCommand(0.65, 0.00, 0.00), "cross_dry"),
            CommandSegment(2.0, VehicleCommand(0.55, 0.24, 0.00), "wet_patch_left"),
            CommandSegment(2.0, VehicleCommand(0.55, -0.24, 0.00), "gravel_patch_right"),
            CommandSegment(1.5, VehicleCommand(0.00, 0.00, 0.60), "brake"),
        ]
    )


def build_default_surface_patches(
    *,
    patch_length_m: float,
    track_width_m: float,
) -> List[SurfacePatch]:
    length_m = float(patch_length_m)
    width_m = float(track_width_m)
    return [
        SurfacePatch(
            name="dry_asphalt",
            x_center_m=-1.0 * length_m,
            y_center_m=0.0,
            length_m=length_m,
            width_m=width_m,
            static_friction=1.00,
            dynamic_friction=0.95,
            tire_friction=1.00,
            color_srgb=(0.28, 0.28, 0.30),
        ),
        SurfacePatch(
            name="wet_asphalt",
            x_center_m=0.0,
            y_center_m=0.0,
            length_m=length_m,
            width_m=width_m,
            static_friction=0.75,
            dynamic_friction=0.70,
            tire_friction=0.72,
            color_srgb=(0.22, 0.31, 0.40),
        ),
        SurfacePatch(
            name="gravel",
            x_center_m=1.0 * length_m,
            y_center_m=0.0,
            length_m=length_m,
            width_m=width_m,
            static_friction=0.60,
            dynamic_friction=0.55,
            tire_friction=0.58,
            color_srgb=(0.55, 0.48, 0.36),
        ),
    ]


def select_surface_patch(
    patches: Sequence[SurfacePatch],
    x_m: float,
    y_m: float,
) -> Optional[SurfacePatch]:
    for patch in patches:
        if patch.contains_xy(x_m, y_m):
            return patch
    return None


class _VehicleController:
    def __init__(self, ctrl_prim: Any):
        self._ctrl_prim = ctrl_prim
        self._accelerator_attr = ctrl_prim.GetAttribute("physxVehicleController:accelerator")
        self._steer_attr = ctrl_prim.GetAttribute("physxVehicleController:steer")
        self._brake_attr = ctrl_prim.GetAttribute("physxVehicleController:brake0")
        if not self._accelerator_attr.IsValid():
            raise RuntimeError(f"Vehicle controller prim is missing accelerator attr: {ctrl_prim.GetPath().pathString}")
        if not self._steer_attr.IsValid():
            raise RuntimeError(f"Vehicle controller prim is missing steer attr: {ctrl_prim.GetPath().pathString}")
        if not self._brake_attr.IsValid():
            raise RuntimeError(f"Vehicle controller prim is missing brake0 attr: {ctrl_prim.GetPath().pathString}")

    @property
    def prim_path(self) -> str:
        return self._ctrl_prim.GetPath().pathString

    def apply(self, command: VehicleCommand) -> None:
        cmd = command.clamped()
        self._accelerator_attr.Set(float(cmd.accelerator))
        self._steer_attr.Set(float(cmd.steering))
        self._brake_attr.Set(float(cmd.brake))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a standalone PhysX DriveBasic patch-track teacher vehicle and record rollout telemetry."
    )
    parser.add_argument("--headless", action="store_true", help="Run without UI.")
    parser.add_argument("--output-dir", type=str, default="artifacts/physx_teacher_patch_track/default")
    parser.add_argument(
        "--command-program",
        type=str,
        default="",
        help="Optional JSON file describing the teacher command segments. If omitted, the built-in demo program is used.",
    )
    parser.add_argument("--dt", type=float, default=1.0 / 60.0, help="Simulation step.")
    parser.add_argument("--track-width-m", type=float, default=10.0)
    parser.add_argument("--patch-length-m", type=float, default=12.0)
    parser.add_argument("--spawn-height-m", type=float, default=1.2)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--settle-steps", type=int, default=60, help="Physics steps to settle the vehicle on the ground before logging.")
    parser.add_argument("--max-steps", type=int, default=0, help="0 uses the full built-in command program.")
    parser.add_argument("--save-stage-usd", type=str, default="", help="Optional path to save the built USD stage.")
    return parser.parse_args()


def _enable_extensions(ext_names: Sequence[str]) -> None:
    import omni.kit.app

    app = omni.kit.app.get_app()
    manager = app.get_extension_manager()
    available_ids = {
        str(ext_info.get("id"))
        for ext_info in manager.get_extensions()
        if ext_info.get("id") is not None
    }
    for ext_name in ext_names:
        if str(ext_name) not in available_ids:
            continue
        try:
            manager.set_extension_enabled_immediate(ext_name, True)
        except Exception:
            continue
    for _ in range(5):
        app.update()


_EXTENSION_ROOT_BY_MODULE: Dict[str, Path] = {}
_EXTENSION_MODULE_MAP_READY = False


def _get_extscache_root() -> Optional[Path]:
    try:
        import isaacsim
    except ModuleNotFoundError:
        return None

    return Path(isaacsim.__file__).resolve().parent / "extscache"


def _load_extcache_module_map() -> None:
    global _EXTENSION_MODULE_MAP_READY
    if _EXTENSION_MODULE_MAP_READY:
        return
    extscache_root = _get_extscache_root()
    if extscache_root is None or not extscache_root.exists():
        _EXTENSION_MODULE_MAP_READY = True
        return

    for candidate in sorted(extscache_root.iterdir()):
        manifest_path = candidate / "config" / "extension.toml"
        if not manifest_path.exists():
            continue
        try:
            manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        modules = manifest.get("python", {}).get("module", [])
        if not isinstance(modules, list):
            continue
        for module_entry in modules:
            if not isinstance(module_entry, dict):
                continue
            declared_name = str(module_entry.get("name", "")).strip()
            if declared_name and declared_name not in _EXTENSION_ROOT_BY_MODULE:
                _EXTENSION_ROOT_BY_MODULE[declared_name] = candidate

    _EXTENSION_MODULE_MAP_READY = True


def _resolve_extension_module_name(module_name: str) -> Optional[str]:
    _load_extcache_module_map()
    if module_name in _EXTENSION_ROOT_BY_MODULE:
        return module_name

    best_match: Optional[str] = None
    for declared_name in _EXTENSION_ROOT_BY_MODULE:
        if module_name.startswith(declared_name + "."):
            if best_match is None or len(declared_name) > len(best_match):
                best_match = declared_name
    return best_match


def _locate_extension_root_for_python_module(module_name: str) -> Optional[Path]:
    declared_name = _resolve_extension_module_name(module_name)
    if declared_name is None:
        return None
    return _EXTENSION_ROOT_BY_MODULE.get(declared_name)


def _extend_omni_namespace_with_extension_root(extension_root: Path) -> None:
    extension_root_str = str(extension_root)
    omni_pkg_root = str(extension_root / "omni")
    if extension_root_str not in sys.path:
        sys.path.insert(0, extension_root_str)

    try:
        import omni
    except ModuleNotFoundError:
        return

    omni_paths = list(getattr(omni, "__path__", []))
    if omni_pkg_root not in omni_paths:
        try:
            omni.__path__.append(omni_pkg_root)
        except Exception:
            omni.__path__ = omni_paths + [omni_pkg_root]


def _ensure_extension_python_module_available(module_name: str) -> None:
    try:
        importlib.import_module(module_name)
        return
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", "")
        missing_declared_name = _resolve_extension_module_name(missing_name) if missing_name else None
        module_declared_name = _resolve_extension_module_name(module_name) or module_name
        if missing_declared_name and missing_declared_name != module_declared_name:
            raise

    extension_root = _locate_extension_root_for_python_module(module_name)
    if extension_root is None:
        raise ModuleNotFoundError(
            f"Could not locate the {module_name} Python package in isaacsim/extscache."
        )

    _extend_omni_namespace_with_extension_root(extension_root)
    if importlib.util.find_spec(module_name) is None:
        raise ModuleNotFoundError(
            f"Could not locate the {module_name} Python package in isaacsim/extscache."
        )

    importlib.import_module(module_name)


def _ensure_world_default_prim(stage: Any) -> None:
    from pxr import UsdGeom

    world = stage.GetPrimAtPath("/World")
    if not world.IsValid():
        world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    if not stage.GetDefaultPrim().IsValid():
        stage.SetDefaultPrim(world)


def _set_stage_units(stage: Any) -> None:
    from pxr import UsdGeom, UsdPhysics

    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)


def _ensure_physics_scene(stage: Any, *, scene_path: str = "/World/PhysicsScene") -> str:
    from pxr import Gf, PhysxSchema, UsdPhysics

    scene_prim = stage.GetPrimAtPath(scene_path)
    if not scene_prim.IsValid():
        scene = UsdPhysics.Scene.Define(stage, scene_path)
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        scene_prim = scene.GetPrim()

    try:
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
        physx_scene.CreateEnableCCDAttr().Set(True)
    except Exception:
        pass
    return scene_path


def _srgb_to_linear(channel: float) -> float:
    c = float(channel)
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _get_or_create_preview_material(
    stage: Any,
    mat_path: str,
    *,
    rgb_srgb: Tuple[float, float, float],
) -> Any:
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Get(stage, mat_path)
    if not material:
        material = UsdShade.Material.Define(stage, mat_path)

    shader_path = f"{mat_path}/PreviewSurface"
    shader = UsdShade.Shader.Get(stage, shader_path)
    if not shader:
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")

    rgb_linear = Gf.Vec3f(*[_srgb_to_linear(v) for v in rgb_srgb])
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(rgb_linear)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.92)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind_material(prim: Any, material: Any) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI(prim).Bind(material)


def _create_physics_material(
    stage: Any,
    *,
    mat_path: str,
    patch: SurfacePatch,
) -> Any:
    from pxr import PhysxSchema, UsdPhysics

    material = _get_or_create_preview_material(stage, mat_path, rgb_srgb=patch.color_srgb)
    prim = material.GetPrim()
    physics_material = UsdPhysics.MaterialAPI.Apply(prim)
    physics_material.CreateStaticFrictionAttr().Set(float(patch.static_friction))
    physics_material.CreateDynamicFrictionAttr().Set(float(patch.dynamic_friction))
    physics_material.CreateRestitutionAttr().Set(0.0)
    PhysxSchema.PhysxMaterialAPI.Apply(prim)
    prim.SetCustomDataByKey("surface_name", str(patch.name))
    prim.SetCustomDataByKey("tire_friction", float(patch.tire_friction))
    return material


def _create_surface_patch_geom(
    stage: Any,
    *,
    prim_path: str,
    material: Any,
    patch: SurfacePatch,
    thickness_m: float = 0.25,
    z_m: float = 0.0,
) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.GetSizeAttr().Set(1.0)
    xform_api = UsdGeom.XformCommonAPI(cube)
    xform_api.SetTranslate(
        Gf.Vec3d(
            float(patch.x_center_m),
            float(patch.y_center_m),
            float(z_m - 0.5 * thickness_m),
        )
    )
    xform_api.SetScale(
        Gf.Vec3f(
            float(patch.length_m),
            float(patch.width_m),
            float(thickness_m),
        )
    )
    prim = cube.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    _bind_material(prim, material)
    prim.SetCustomDataByKey("surface_name", str(patch.name))
    prim.SetCustomDataByKey("surface_patch", patch.to_dict())


def _upsert_friction_table_entry(
    stage: Any,
    *,
    table_path: str,
    material_path: str,
    friction_value: float,
) -> None:
    from pxr import PhysxSchema, Sdf

    table = PhysxSchema.PhysxVehicleTireFrictionTable.Get(stage, table_path)
    if not table:
        table = PhysxSchema.PhysxVehicleTireFrictionTable.Define(stage, table_path)

    relationship = table.CreateGroundMaterialsRel()
    targets = list(relationship.GetTargets())
    values_attr = table.CreateFrictionValuesAttr()
    values = list(values_attr.Get() or [])
    material_target = Sdf.Path(material_path)

    if material_target in targets:
        index = targets.index(material_target)
        while len(values) <= index:
            values.append(float(friction_value))
        values[index] = float(friction_value)
    else:
        relationship.AddTarget(material_target)
        targets = list(relationship.GetTargets())
        while len(values) < len(targets) - 1:
            values.append(float(friction_value))
        values.append(float(friction_value))

    values_attr.Set(values)


def _ensure_physx_vehicle_python_available() -> None:
    resolved_dependency_roots: set[str] = set()
    while True:
        try:
            _ensure_extension_python_module_available("omni.physxvehicle")
            return
        except ModuleNotFoundError as exc:
            missing_root_name = _resolve_extension_module_name(getattr(exc, "name", ""))
            if (
                not missing_root_name
                or missing_root_name == "omni.physxvehicle"
                or missing_root_name in resolved_dependency_roots
            ):
                raise
            _ensure_extension_python_module_available(missing_root_name)
            resolved_dependency_roots.add(missing_root_name)


def _wait_for_physx_vehicle_python(app: Any, *, max_updates: int = 120) -> None:
    last_exc: Optional[BaseException] = None
    for _ in range(max_updates):
        try:
            _ensure_physx_vehicle_python_available()
            return
        except ModuleNotFoundError as exc:
            last_exc = exc
            missing_root_name = _resolve_extension_module_name(getattr(exc, "name", ""))
            if missing_root_name and not missing_root_name.startswith("omni."):
                raise
        except ImportError as exc:
            last_exc = exc
        app.update()

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Timed out waiting for omni.physxvehicle to become importable.")


def _get_unit_scale(stage: Any) -> Any:
    _ensure_physx_vehicle_python_available()
    from omni.physxvehicle.scripts.helpers.UnitScale import UnitScale
    from pxr import UsdGeom, UsdPhysics

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    kilograms_per_unit = float(UsdPhysics.GetStageKilogramsPerUnit(stage) or 1.0)
    return UnitScale(1.0 / meters_per_unit, 1.0 / kilograms_per_unit)


def _spawn_drive_basic_teacher(
    stage: Any,
    *,
    parent_path: str,
    position_m: Tuple[float, float, float],
    yaw_deg: float,
) -> str:
    import omni.usd
    _ensure_physx_vehicle_python_available()
    from omni.physxvehicle.scripts.commands import PhysXVehicleWizardCreateCommand
    from omni.physxvehicle.scripts.wizards import physxVehicleWizard as VehicleWizard
    from pxr import Gf, UsdGeom

    _ensure_world_default_prim(stage)

    parent_xform = UsdGeom.Xform.Define(stage, parent_path)
    xform_api = UsdGeom.XformCommonAPI(parent_xform)
    xform_api.SetTranslate(Gf.Vec3d(*[float(v) for v in position_m]))
    xform_api.SetRotate(Gf.Vec3f(0.0, 0.0, float(yaw_deg)), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    unit_scale = _get_unit_scale(stage)
    vehicle_data = VehicleWizard.VehicleData(
        unit_scale,
        VehicleWizard.VehicleData.AXIS_Z,
        VehicleWizard.VehicleData.AXIS_X,
    )
    vehicle_data.rootVehiclePath = f"{parent_path}/Vehicle"
    vehicle_data.rootSharedPath = "/World/VehicleShared"
    vehicle_data.set_drive_type(VehicleWizard.DRIVE_TYPE_BASIC)
    vehicle_data.maxSteerAngle[0] = 32.0
    vehicle_data.driven[0] = True
    vehicle_data.driven[1] = False

    result = PhysXVehicleWizardCreateCommand.execute(vehicle_data)
    success = bool(result[0]) if isinstance(result, (list, tuple)) and result else False
    if not success:
        raise RuntimeError(f"PhysX vehicle wizard failed: {result}")
    return str(vehicle_data.rootVehiclePath)


def _resolve_controller_prim(stage: Any, vehicle_root_path: str) -> Any:
    candidates = [f"{vehicle_root_path}/Vehicle", vehicle_root_path]
    for path in candidates:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        if prim.GetAttribute("physxVehicleController:accelerator").IsValid():
            return prim
    raise RuntimeError(f"Could not resolve vehicle controller prim under {vehicle_root_path}")


def _discover_wheel_attachments(stage: Any, vehicle_root_path: str) -> List[Dict[str, Any]]:
    from pxr import PhysxSchema, Usd

    root_prim = stage.GetPrimAtPath(vehicle_root_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Vehicle root does not exist: {vehicle_root_path}")

    wheel_info: List[Tuple[int, str]] = []
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsValid():
            continue
        try:
            has_api = prim.HasAPI(PhysxSchema.PhysxVehicleWheelAttachmentAPI)
        except Exception:
            has_api = False
        if not has_api:
            continue
        api = PhysxSchema.PhysxVehicleWheelAttachmentAPI(prim)
        try:
            index = int(api.GetIndexAttr().Get())
        except Exception:
            index = len(wheel_info)
        wheel_info.append((index, prim.GetPath().pathString))

    wheel_info.sort(key=lambda item: item[0])
    labels_by_index = {
        0: "front_left",
        1: "front_right",
        2: "rear_left",
        3: "rear_right",
    }
    return [
        {
            "index": int(index),
            "label": labels_by_index.get(index, f"wheel_{index}"),
            "path": str(path),
        }
        for index, path in wheel_info
    ]


def _vec3_to_list(value: Any) -> List[float]:
    return [float(value[0]), float(value[1]), float(value[2])]


def _quat_to_list(value: Any) -> List[float]:
    imag = value.GetImaginary()
    return [float(value.GetReal()), float(imag[0]), float(imag[1]), float(imag[2])]


def _serialize_drive_state(state: Dict[Any, Any], state_indices: Dict[str, int]) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for name, index in state_indices.items():
        if index not in state:
            continue
        value = state[index]
        if isinstance(value, (tuple, list)):
            data[name] = [float(v) for v in value]
        elif isinstance(value, bool):
            data[name] = bool(value)
        elif isinstance(value, int):
            data[name] = int(value)
        else:
            data[name] = float(value)
    return data


def _collect_rollout_frame(
    *,
    step_idx: int,
    sim_time_s: float,
    command: VehicleCommand,
    stage: Any,
    vehicle_body_path: str,
    controller_path: str,
    wheel_attachments: Sequence[Dict[str, Any]],
    patches: Sequence[SurfacePatch],
    material_to_patch: Dict[str, SurfacePatch],
    physx_interface: Any,
    wheel_state_indices: Dict[str, int],
    drive_state_indices: Dict[str, int],
) -> Dict[str, Any]:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    body_prim = stage.GetPrimAtPath(vehicle_body_path)
    if not body_prim.IsValid():
        raise RuntimeError(f"Vehicle body prim missing: {vehicle_body_path}")

    transform = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    position = transform.ExtractTranslation()
    forward_world = transform.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
    rotation = transform.ExtractRotationQuat()

    rigid_body = UsdPhysics.RigidBodyAPI(body_prim)
    linear_velocity = rigid_body.GetVelocityAttr().Get()
    angular_velocity = rigid_body.GetAngularVelocityAttr().Get()

    drive_state_raw = physx_interface.get_vehicle_drive_state(controller_path) or {}
    drive_state = _serialize_drive_state(drive_state_raw, drive_state_indices)

    wheels: List[Dict[str, Any]] = []
    for wheel in wheel_attachments:
        wheel_state = physx_interface.get_wheel_state(wheel["path"]) or {}
        is_on_ground = bool(wheel_state.get(wheel_state_indices["is_on_ground"], 0))
        ground_material_path = wheel_state.get(wheel_state_indices["ground_material"], "") or ""
        if is_on_ground:
            hit_position = wheel_state.get(wheel_state_indices["ground_hit_position"], None)
            hit_position_list = [float(v) for v in hit_position] if hit_position is not None else None
        else:
            hit_position_list = None

        patch = material_to_patch.get(str(ground_material_path))
        if patch is None and is_on_ground and hit_position_list is not None:
            patch = select_surface_patch(patches, hit_position_list[0], hit_position_list[1])

        wheels.append(
            {
                "index": int(wheel["index"]),
                "label": str(wheel["label"]),
                "path": str(wheel["path"]),
                "is_on_ground": bool(is_on_ground),
                "ground_material_path": str(ground_material_path),
                "ground_hit_position_m": hit_position_list,
                "surface_name": None if patch is None else str(patch.name),
                "rotation_speed_rad_s": float(wheel_state.get(wheel_state_indices["rotation_speed"], 0.0)),
                "rotation_angle_rad": float(wheel_state.get(wheel_state_indices["rotation_angle"], 0.0)),
                "steer_angle_rad": float(wheel_state.get(wheel_state_indices["steer_angle"], 0.0)),
                "suspension_jounce": float(wheel_state.get(wheel_state_indices["suspension_jounce"], 0.0)),
                "tire_friction": float(wheel_state.get(wheel_state_indices["tire_friction"], 0.0)),
                "tire_longitudinal_slip": float(wheel_state.get(wheel_state_indices["tire_longitudinal_slip"], 0.0)),
                "tire_lateral_slip": float(wheel_state.get(wheel_state_indices["tire_lateral_slip"], 0.0)),
            }
        )

    return {
        "step": int(step_idx),
        "sim_time_s": float(sim_time_s),
        "command": command.clamped().to_dict(),
        "vehicle": {
            "position_m": _vec3_to_list(position),
            "orientation_wxyz": _quat_to_list(rotation),
            "yaw_rad": float(_yaw_from_basis_vectors(forward_world)),
            "linear_velocity_mps": _vec3_to_list(linear_velocity),
            "angular_velocity_rad_s": _vec3_to_list(angular_velocity),
        },
        "drive_state": drive_state,
        "wheels": wheels,
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _reset_status_log(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "_runner_status.log").write_text("", encoding="utf-8")


def _append_status(output_dir: Path, message: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "_runner_status.log").open("a", encoding="utf-8") as stream:
        stream.write(str(message).rstrip() + "\n")


def _build_track_and_materials(
    stage: Any,
    *,
    root_path: str,
    patches: Sequence[SurfacePatch],
) -> Tuple[Dict[str, SurfacePatch], Dict[str, str]]:
    from pxr import UsdGeom

    UsdGeom.Xform.Define(stage, root_path)
    UsdGeom.Xform.Define(stage, f"{root_path}/Materials")
    UsdGeom.Xform.Define(stage, f"{root_path}/Ground")

    material_to_patch: Dict[str, SurfacePatch] = {}
    patch_to_material_path: Dict[str, str] = {}
    for patch in patches:
        material_path = f"{root_path}/Materials/{patch.name}"
        ground_path = f"{root_path}/Ground/{patch.name}"
        material = _create_physics_material(stage, mat_path=material_path, patch=patch)
        _create_surface_patch_geom(stage, prim_path=ground_path, material=material, patch=patch)
        material_to_patch[str(material.GetPath())] = patch
        patch_to_material_path[patch.name] = str(material.GetPath())
    return material_to_patch, patch_to_material_path


def _patch_teacher_tire_table(
    stage: Any,
    *,
    patches: Sequence[SurfacePatch],
    patch_to_material_path: Dict[str, str],
) -> None:
    summer_table_path = "/World/VehicleShared/SummerTireFrictionTable"
    for patch in patches:
        material_path = patch_to_material_path.get(patch.name)
        if material_path is None:
            continue
        _upsert_friction_table_entry(
            stage,
            table_path=summer_table_path,
            material_path=material_path,
            friction_value=float(patch.tire_friction),
        )


def _run() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _reset_status_log(output_dir)
    _append_status(output_dir, "parsed_args")

    from isaacsim import SimulationApp

    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]
    simulation_app = SimulationApp(
        {
            "headless": bool(args.headless),
            "renderer": "None" if bool(args.headless) else "RayTracedLighting",
        }
    )
    sys.argv = original_argv
    _append_status(output_dir, "simulation_app_started")

    try:
        import omni.kit.app
        import omni.physx
        import omni.timeline
        import omni.usd
        from omni.physx.bindings._physx import (
            VEHICLE_DRIVE_STATE_ACCELERATOR,
            VEHICLE_DRIVE_STATE_BRAKE0,
            VEHICLE_DRIVE_STATE_STEER,
            VEHICLE_DRIVE_STATE_TARGET_GEAR,
            VEHICLE_WHEEL_STATE_GROUND_HIT_POSITION,
            VEHICLE_WHEEL_STATE_GROUND_MATERIAL,
            VEHICLE_WHEEL_STATE_IS_ON_GROUND,
            VEHICLE_WHEEL_STATE_ROTATION_ANGLE,
            VEHICLE_WHEEL_STATE_ROTATION_SPEED,
            VEHICLE_WHEEL_STATE_STEER_ANGLE,
            VEHICLE_WHEEL_STATE_SUSPENSION_JOUNCE,
            VEHICLE_WHEEL_STATE_TIRE_FRICTION,
            VEHICLE_WHEEL_STATE_TIRE_LATERAL_SLIP,
            VEHICLE_WHEEL_STATE_TIRE_LONGITUDINAL_SLIP,
        )
        _append_status(output_dir, "runtime_imports_ready")

        _enable_extensions(["omni.physxvehicle", "omni.physx.vehicle"])
        _append_status(output_dir, "extensions_enabled")
        app = omni.kit.app.get_app()
        _wait_for_physx_vehicle_python(app)
        _append_status(output_dir, "physx_vehicle_python_ready")

        usd_context = omni.usd.get_context()
        usd_context.new_stage()
        stage = usd_context.get_stage()
        _ensure_world_default_prim(stage)
        _set_stage_units(stage)
        _ensure_physics_scene(stage)
        _append_status(output_dir, "stage_ready")

        timeline = omni.timeline.get_timeline_interface()
        physx_interface = omni.physx.get_physx_interface()

        patches = build_default_surface_patches(
            patch_length_m=float(args.patch_length_m),
            track_width_m=float(args.track_width_m),
        )
        material_to_patch, patch_to_material_path = _build_track_and_materials(
            stage,
            root_path="/World/PatchTrack",
            patches=patches,
        )

        total_track_length = float(sum(patch.length_m for patch in patches))
        vehicle_root_path = _spawn_drive_basic_teacher(
            stage,
            parent_path="/World/PatchTrack/Teacher",
            position_m=(
                -0.5 * total_track_length + 0.35 * float(args.patch_length_m),
                0.0,
                float(args.spawn_height_m),
            ),
            yaw_deg=0.0,
        )
        _patch_teacher_tire_table(
            stage,
            patches=patches,
            patch_to_material_path=patch_to_material_path,
        )

        controller_prim = _resolve_controller_prim(stage, vehicle_root_path)
        controller = _VehicleController(controller_prim)
        vehicle_body_path = f"{vehicle_root_path}/Vehicle"
        wheel_attachments = _discover_wheel_attachments(stage, vehicle_root_path)
        if not wheel_attachments:
            raise RuntimeError(f"No wheel attachments found under {vehicle_root_path}")
        _append_status(output_dir, "vehicle_ready")

        if args.command_program:
            program = load_command_program(args.command_program)
            program_source = str(Path(args.command_program).expanduser().resolve())
        else:
            program = build_default_program()
            program_source = "builtin_default"
        _append_status(output_dir, f"program_ready:{program_source}")
        dt_s = float(args.dt)
        max_steps = int(args.max_steps) if int(args.max_steps) > 0 else int(math.ceil(program.total_duration_s / dt_s))

        metadata = {
            "dt_s": dt_s,
            "max_steps": int(max_steps),
            "settle_steps": int(args.settle_steps),
            "command_program_source": str(program_source),
            "vehicle_root_path": str(vehicle_root_path),
            "controller_path": str(controller.prim_path),
            "vehicle_body_path": str(vehicle_body_path),
            "wheel_attachments": wheel_attachments,
            "surface_patches": [patch.to_dict() for patch in patches],
            "material_paths": patch_to_material_path,
            "command_program": program.to_dict_list(),
        }
        _write_json(output_dir / "rollout_meta.json", metadata)
        _append_status(output_dir, "metadata_written")

        for _ in range(int(args.warmup_steps)):
            app.update()

        timeline.play()
        controller.apply(VehicleCommand(0.0, 0.0, 0.0))
        for _ in range(int(args.settle_steps)):
            app.update()

        wheel_state_indices = {
            "ground_hit_position": int(VEHICLE_WHEEL_STATE_GROUND_HIT_POSITION),
            "ground_material": int(VEHICLE_WHEEL_STATE_GROUND_MATERIAL),
            "is_on_ground": int(VEHICLE_WHEEL_STATE_IS_ON_GROUND),
            "rotation_angle": int(VEHICLE_WHEEL_STATE_ROTATION_ANGLE),
            "rotation_speed": int(VEHICLE_WHEEL_STATE_ROTATION_SPEED),
            "steer_angle": int(VEHICLE_WHEEL_STATE_STEER_ANGLE),
            "suspension_jounce": int(VEHICLE_WHEEL_STATE_SUSPENSION_JOUNCE),
            "tire_friction": int(VEHICLE_WHEEL_STATE_TIRE_FRICTION),
            "tire_longitudinal_slip": int(VEHICLE_WHEEL_STATE_TIRE_LONGITUDINAL_SLIP),
            "tire_lateral_slip": int(VEHICLE_WHEEL_STATE_TIRE_LATERAL_SLIP),
        }
        drive_state_indices = {
            "accelerator": int(VEHICLE_DRIVE_STATE_ACCELERATOR),
            "brake0": int(VEHICLE_DRIVE_STATE_BRAKE0),
            "steer": int(VEHICLE_DRIVE_STATE_STEER),
            "target_gear": int(VEHICLE_DRIVE_STATE_TARGET_GEAR),
        }

        frames_path = output_dir / "rollout_frames.jsonl"
        with frames_path.open("w", encoding="utf-8") as stream:
            for step_idx in range(max_steps):
                sim_time_s = float(step_idx * dt_s)
                command = program.command_at(sim_time_s)
                controller.apply(command)
                app.update()
                frame = _collect_rollout_frame(
                    step_idx=step_idx,
                    sim_time_s=sim_time_s + dt_s,
                    command=command,
                    stage=stage,
                    vehicle_body_path=vehicle_body_path,
                    controller_path=vehicle_body_path,
                    wheel_attachments=wheel_attachments,
                    patches=patches,
                    material_to_patch=material_to_patch,
                    physx_interface=physx_interface,
                    wheel_state_indices=wheel_state_indices,
                    drive_state_indices=drive_state_indices,
                )
                stream.write(json.dumps(frame) + "\n")
        _append_status(output_dir, "frames_written")

        controller.apply(VehicleCommand(0.0, 0.0, 1.0))
        for _ in range(3):
            app.update()

        if args.save_stage_usd:
            save_path = Path(args.save_stage_usd).expanduser().resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            usd_context.save_as_stage(str(save_path))

        timeline.stop()
        _append_status(output_dir, "run_completed")
        print(f"[physx_teacher_patch_track] wrote rollout to {output_dir}")
    except BaseException:
        _append_status(output_dir, traceback.format_exc())
        raise
    finally:
        simulation_app.close()


def main() -> None:
    _run()


if __name__ == "__main__":
    main()
