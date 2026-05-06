from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from xml.etree import ElementTree as ET


# Match the PhysX vehicle wizard reference car used by the teacher path:
# chassis = 4.0 x 2.0 x 1.0 m, tire radius = 0.35 m, tire width = 0.15 m.
# For the 2-axle layout, the wizard places wheel centers at +/- (0.5 * chassis_length - 2 * wheel_radius)
# longitudinally and at +/- 0.5 * chassis_width laterally.
PHYSX_WIZARD_DEFAULT_CHASSIS_LENGTH_M = 4.0
PHYSX_WIZARD_DEFAULT_CHASSIS_WIDTH_M = 2.0
PHYSX_WIZARD_DEFAULT_CHASSIS_HEIGHT_M = 1.0
PHYSX_WIZARD_DEFAULT_CHASSIS_MASS_KG = 1800.0
PHYSX_WIZARD_DEFAULT_WHEEL_RADIUS_M = 0.35
PHYSX_WIZARD_DEFAULT_WHEEL_WIDTH_M = 0.15
PHYSX_WIZARD_DEFAULT_WHEEL_MASS_KG = 20.0
PHYSX_WIZARD_DEFAULT_CHASSIS_RGBA = (0.2784314, 0.64705884, 1.0, 1.0)
PHYSX_WIZARD_DEFAULT_WHEELBASE_M = (
    PHYSX_WIZARD_DEFAULT_CHASSIS_LENGTH_M - 4.0 * PHYSX_WIZARD_DEFAULT_WHEEL_RADIUS_M
)
PHYSX_WIZARD_DEFAULT_TRACK_WIDTH_M = PHYSX_WIZARD_DEFAULT_CHASSIS_WIDTH_M
PHYSX_WIZARD_DEFAULT_GROUND_CLEARANCE_M = PHYSX_WIZARD_DEFAULT_WHEEL_RADIUS_M
PHYSX_WIZARD_DEFAULT_SUSPENSION_REST_LENGTH_M = 0.5 * PHYSX_WIZARD_DEFAULT_WHEEL_RADIUS_M
PHYSX_WIZARD_DEFAULT_SUSPENSION_TRAVEL_M = PHYSX_WIZARD_DEFAULT_WHEEL_RADIUS_M


@dataclass(frozen=True)
class StudentVehicleSpec:
    name: str = "student_fwd_vehicle"
    chassis_length_m: float = PHYSX_WIZARD_DEFAULT_CHASSIS_LENGTH_M
    chassis_width_m: float = PHYSX_WIZARD_DEFAULT_CHASSIS_WIDTH_M
    chassis_height_m: float = PHYSX_WIZARD_DEFAULT_CHASSIS_HEIGHT_M
    chassis_mass_kg: float = PHYSX_WIZARD_DEFAULT_CHASSIS_MASS_KG
    wheelbase_m: float = PHYSX_WIZARD_DEFAULT_WHEELBASE_M
    track_width_m: float = PHYSX_WIZARD_DEFAULT_TRACK_WIDTH_M
    wheel_radius_m: float = PHYSX_WIZARD_DEFAULT_WHEEL_RADIUS_M
    wheel_width_m: float = PHYSX_WIZARD_DEFAULT_WHEEL_WIDTH_M
    wheel_mass_kg: float = PHYSX_WIZARD_DEFAULT_WHEEL_MASS_KG
    ground_clearance_m: float = PHYSX_WIZARD_DEFAULT_GROUND_CLEARANCE_M
    suspension_mount_from_bottom_m: float = PHYSX_WIZARD_DEFAULT_SUSPENSION_REST_LENGTH_M
    suspension_travel_m: float = PHYSX_WIZARD_DEFAULT_SUSPENSION_TRAVEL_M
    suspension_link_mass_kg: float = 6.0
    steering_knuckle_mass_kg: float = 3.0
    steering_limit_deg: float = 32.0
    steering_damping: float = 25.0
    steering_friction: float = 1.0
    suspension_damping: float = 2200.0
    suspension_friction: float = 120.0
    wheel_damping: float = 0.25
    suspension_effort_limit_n: float = 15000.0
    suspension_velocity_limit_mps: float = 3.0
    steering_effort_limit_nm: float = 1200.0
    steering_velocity_limit_radps: float = 4.0
    wheel_effort_limit_nm: float = 4000.0
    wheel_velocity_limit_radps: float = 400.0


@dataclass(frozen=True)
class CornerSpec:
    name: str
    x_m: float
    y_m: float
    is_front: bool


def build_default_student_vehicle_spec() -> StudentVehicleSpec:
    return StudentVehicleSpec()


def load_student_vehicle_spec(path: str | Path) -> StudentVehicleSpec:
    source_path = Path(path).expanduser().resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Student vehicle spec must be a JSON object, got {type(payload).__name__}")
    default = asdict(build_default_student_vehicle_spec())
    unknown = sorted(set(payload.keys()) - set(default.keys()))
    if unknown:
        raise ValueError(f"Unknown student vehicle spec fields: {', '.join(unknown)}")
    return StudentVehicleSpec(**{**default, **payload})


def write_student_vehicle_spec(path: str | Path, spec: StudentVehicleSpec) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(spec), indent=2) + "\n", encoding="utf-8")
    return output_path


def build_corner_specs(spec: StudentVehicleSpec) -> List[CornerSpec]:
    half_wheelbase = 0.5 * float(spec.wheelbase_m)
    half_track = 0.5 * float(spec.track_width_m)
    return [
        CornerSpec(name="front_left", x_m=half_wheelbase, y_m=half_track, is_front=True),
        CornerSpec(name="front_right", x_m=half_wheelbase, y_m=-half_track, is_front=True),
        CornerSpec(name="rear_left", x_m=-half_wheelbase, y_m=half_track, is_front=False),
        CornerSpec(name="rear_right", x_m=-half_wheelbase, y_m=-half_track, is_front=False),
    ]


def nominal_wheel_center_z_m(spec: StudentVehicleSpec) -> float:
    return float(spec.wheel_radius_m) - float(spec.ground_clearance_m) - 0.5 * float(spec.chassis_height_m)


def suspension_mount_z_m(spec: StudentVehicleSpec) -> float:
    return -0.5 * float(spec.chassis_height_m) + float(spec.suspension_mount_from_bottom_m)


def suspension_rest_length_m(spec: StudentVehicleSpec) -> float:
    return suspension_mount_z_m(spec) - nominal_wheel_center_z_m(spec)


def nominal_root_height_m(spec: StudentVehicleSpec, *, ground_margin_m: float = 0.03) -> float:
    return float(spec.wheel_radius_m) - nominal_wheel_center_z_m(spec) + float(ground_margin_m)


def validate_student_vehicle_spec(spec: StudentVehicleSpec) -> None:
    scalar_fields = [
        "chassis_length_m",
        "chassis_width_m",
        "chassis_height_m",
        "chassis_mass_kg",
        "wheelbase_m",
        "track_width_m",
        "wheel_radius_m",
        "wheel_width_m",
        "wheel_mass_kg",
        "ground_clearance_m",
        "suspension_mount_from_bottom_m",
        "suspension_travel_m",
        "suspension_link_mass_kg",
        "steering_knuckle_mass_kg",
        "suspension_effort_limit_n",
        "suspension_velocity_limit_mps",
        "steering_effort_limit_nm",
        "steering_velocity_limit_radps",
        "wheel_effort_limit_nm",
        "wheel_velocity_limit_radps",
    ]
    for field_name in scalar_fields:
        value = float(getattr(spec, field_name))
        if value <= 0.0:
            raise ValueError(f"{field_name} must be positive, got {value}")

    if float(spec.track_width_m) <= float(spec.wheel_width_m):
        raise ValueError("track_width_m must exceed wheel_width_m")
    if float(spec.wheelbase_m) <= float(spec.wheel_radius_m):
        raise ValueError("wheelbase_m is unrealistically short for the wheel radius")
    if not 0.0 < float(spec.suspension_mount_from_bottom_m) < float(spec.chassis_height_m):
        raise ValueError("suspension_mount_from_bottom_m must be inside the chassis height")
    if not 0.0 < float(spec.steering_limit_deg) < 89.0:
        raise ValueError("steering_limit_deg must be in (0, 89)")
    if float(spec.ground_clearance_m) > float(spec.wheel_radius_m):
        raise ValueError("ground_clearance_m must not exceed wheel_radius_m")
    if suspension_rest_length_m(spec) <= 0.0:
        raise ValueError("Suspension rest length must be positive; adjust chassis or suspension geometry")


def _box_inertia(mass_kg: float, size_xyz_m: Sequence[float]) -> Tuple[float, float, float]:
    sx, sy, sz = [float(v) for v in size_xyz_m]
    mass = float(mass_kg)
    return (
        mass * (sy * sy + sz * sz) / 12.0,
        mass * (sx * sx + sz * sz) / 12.0,
        mass * (sx * sx + sy * sy) / 12.0,
    )


def _cylinder_inertia_y_axis(mass_kg: float, radius_m: float, length_m: float) -> Tuple[float, float, float]:
    mass = float(mass_kg)
    radius = float(radius_m)
    length = float(length_m)
    return (
        mass * (3.0 * radius * radius + length * length) / 12.0,
        0.5 * mass * radius * radius,
        mass * (3.0 * radius * radius + length * length) / 12.0,
    )


def _indent_xml(element: ET.Element, level: int = 0) -> None:
    indent = "\n" + ("  " * level)
    if len(element):
        if not element.text or not element.text.strip():
            element.text = indent + "  "
        for child in element:
            _indent_xml(child, level + 1)
        if not element[-1].tail or not element[-1].tail.strip():
            element[-1].tail = indent
    elif level and (not element.tail or not element.tail.strip()):
        element.tail = indent


def _format_triplet(values: Sequence[float]) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def _add_material(robot: ET.Element, *, name: str, rgba: Sequence[float]) -> None:
    material = ET.SubElement(robot, "material", {"name": str(name)})
    ET.SubElement(material, "color", {"rgba": _format_triplet(rgba)})


def _add_inertial(link: ET.Element, *, mass_kg: float, diagonal_inertia: Sequence[float]) -> None:
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": f"{float(mass_kg):.9g}"})
    ixx, iyy, izz = [float(v) for v in diagonal_inertia]
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": f"{ixx:.9g}",
            "ixy": "0",
            "ixz": "0",
            "iyy": f"{iyy:.9g}",
            "iyz": "0",
            "izz": f"{izz:.9g}",
        },
    )


def _add_box_geometry(
    parent: ET.Element,
    *,
    size_xyz_m: Sequence[float],
    material_name: str | None = None,
    include_visual: bool = True,
    include_collision: bool = True,
) -> None:
    if include_visual:
        visual = ET.SubElement(parent, "visual")
        ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "box", {"size": _format_triplet(size_xyz_m)})
        if material_name is not None:
            ET.SubElement(visual, "material", {"name": str(material_name)})

    if include_collision:
        collision = ET.SubElement(parent, "collision")
        ET.SubElement(collision, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "box", {"size": _format_triplet(size_xyz_m)})


def _add_sphere_geometry(
    parent: ET.Element,
    *,
    radius_m: float,
    material_name: str | None = None,
    include_visual: bool = True,
    include_collision: bool = True,
) -> None:
    if include_visual:
        visual = ET.SubElement(parent, "visual")
        ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "sphere", {"radius": f"{float(radius_m):.9g}"})
        if material_name is not None:
            ET.SubElement(visual, "material", {"name": str(material_name)})

    if include_collision:
        collision = ET.SubElement(parent, "collision")
        ET.SubElement(collision, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "sphere", {"radius": f"{float(radius_m):.9g}"})


def _add_cylinder_y_geometry(
    parent: ET.Element,
    *,
    radius_m: float,
    length_m: float,
    material_name: str | None = None,
    include_visual: bool = True,
    include_collision: bool = True,
) -> None:
    rpy = f"{0.5 * math.pi:.9g} 0 0"
    if include_visual:
        visual = ET.SubElement(parent, "visual")
        ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": rpy})
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "cylinder", {"radius": f"{float(radius_m):.9g}", "length": f"{float(length_m):.9g}"})
        if material_name is not None:
            ET.SubElement(visual, "material", {"name": str(material_name)})

    if include_collision:
        collision = ET.SubElement(parent, "collision")
        ET.SubElement(collision, "origin", {"xyz": "0 0 0", "rpy": rpy})
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "cylinder", {"radius": f"{float(radius_m):.9g}", "length": f"{float(length_m):.9g}"})


def _add_joint(
    robot: ET.Element,
    *,
    name: str,
    joint_type: str,
    parent_link: str,
    child_link: str,
    xyz: Sequence[float],
    rpy: Sequence[float] = (0.0, 0.0, 0.0),
    axis_xyz: Sequence[float] | None = None,
    limit: Dict[str, float] | None = None,
    dynamics: Dict[str, float] | None = None,
) -> None:
    joint = ET.SubElement(robot, "joint", {"name": str(name), "type": str(joint_type)})
    ET.SubElement(joint, "origin", {"xyz": _format_triplet(xyz), "rpy": _format_triplet(rpy)})
    ET.SubElement(joint, "parent", {"link": str(parent_link)})
    ET.SubElement(joint, "child", {"link": str(child_link)})
    if axis_xyz is not None:
        ET.SubElement(joint, "axis", {"xyz": _format_triplet(axis_xyz)})
    if limit is not None:
        ET.SubElement(
            joint,
            "limit",
            {key: f"{float(value):.9g}" for key, value in limit.items()},
        )
    if dynamics is not None:
        ET.SubElement(
            joint,
            "dynamics",
            {key: f"{float(value):.9g}" for key, value in dynamics.items()},
        )


def build_student_vehicle_urdf(spec: StudentVehicleSpec) -> str:
    validate_student_vehicle_spec(spec)

    suspension_rest_m = suspension_rest_length_m(spec)
    suspension_link_height_m = max(0.08, suspension_rest_m)
    suspension_link_size = (
        0.08,
        0.05,
        suspension_link_height_m,
    )
    steering_knuckle_radius_m = 0.045
    chassis_size = (
        float(spec.chassis_length_m),
        float(spec.chassis_width_m),
        float(spec.chassis_height_m),
    )
    wheel_inertia = _cylinder_inertia_y_axis(
        float(spec.wheel_mass_kg),
        float(spec.wheel_radius_m),
        float(spec.wheel_width_m),
    )

    robot = ET.Element("robot", {"name": str(spec.name)})
    _add_material(robot, name="wizard_chassis_blue", rgba=PHYSX_WIZARD_DEFAULT_CHASSIS_RGBA)
    _add_material(robot, name="hidden_helper", rgba=(0.0, 0.0, 0.0, 0.0))
    _add_material(robot, name="wheel_black", rgba=(0.06, 0.06, 0.06, 1.0))

    chassis = ET.SubElement(robot, "link", {"name": "base_link"})
    _add_box_geometry(chassis, size_xyz_m=chassis_size, material_name="wizard_chassis_blue")
    _add_inertial(chassis, mass_kg=float(spec.chassis_mass_kg), diagonal_inertia=_box_inertia(float(spec.chassis_mass_kg), chassis_size))

    for corner in build_corner_specs(spec):
        suspension_link_name = f"{corner.name}_suspension_link"
        suspension_link = ET.SubElement(robot, "link", {"name": suspension_link_name})
        _add_box_geometry(
            suspension_link,
            size_xyz_m=suspension_link_size,
            material_name="hidden_helper",
            include_collision=False,
        )
        _add_inertial(
            suspension_link,
            mass_kg=float(spec.suspension_link_mass_kg),
            diagonal_inertia=_box_inertia(float(spec.suspension_link_mass_kg), suspension_link_size),
        )

        _add_joint(
            robot,
            name=f"{corner.name}_suspension_joint",
            joint_type="prismatic",
            parent_link="base_link",
            child_link=suspension_link_name,
            xyz=(corner.x_m, corner.y_m, suspension_mount_z_m(spec)),
            axis_xyz=(0.0, 0.0, 1.0),
            limit={
                "lower": -0.5 * float(spec.suspension_travel_m),
                "upper": 0.5 * float(spec.suspension_travel_m),
                "effort": float(spec.suspension_effort_limit_n),
                "velocity": float(spec.suspension_velocity_limit_mps),
            },
            dynamics={
                "damping": float(spec.suspension_damping),
                "friction": float(spec.suspension_friction),
            },
        )

        wheel_parent_link = suspension_link_name
        wheel_joint_z_m = -suspension_rest_m + 0.5 * suspension_link_height_m
        if corner.is_front:
            steer_link_name = f"{corner.name}_steer_link"
            steer_link = ET.SubElement(robot, "link", {"name": steer_link_name})
            _add_sphere_geometry(
                steer_link,
                radius_m=steering_knuckle_radius_m,
                material_name="hidden_helper",
                include_collision=False,
            )
            _add_inertial(
                steer_link,
                mass_kg=float(spec.steering_knuckle_mass_kg),
                diagonal_inertia=_box_inertia(
                    float(spec.steering_knuckle_mass_kg),
                    (2.0 * steering_knuckle_radius_m, 2.0 * steering_knuckle_radius_m, 2.0 * steering_knuckle_radius_m),
                ),
            )
            _add_joint(
                robot,
                name=f"{corner.name}_steer_joint",
                joint_type="revolute",
                parent_link=suspension_link_name,
                child_link=steer_link_name,
                xyz=(0.0, 0.0, -0.5 * suspension_link_height_m),
                axis_xyz=(0.0, 0.0, 1.0),
                limit={
                    "lower": -math.radians(float(spec.steering_limit_deg)),
                    "upper": math.radians(float(spec.steering_limit_deg)),
                    "effort": float(spec.steering_effort_limit_nm),
                    "velocity": float(spec.steering_velocity_limit_radps),
                },
                dynamics={
                    "damping": float(spec.steering_damping),
                    "friction": float(spec.steering_friction),
                },
            )
            wheel_parent_link = steer_link_name
            # Keep front and rear wheel centers at the same nominal height.
            wheel_joint_z_m = -suspension_rest_m + suspension_link_height_m

        wheel_link_name = f"{corner.name}_wheel_link"
        wheel_link = ET.SubElement(robot, "link", {"name": wheel_link_name})
        _add_cylinder_y_geometry(
            wheel_link,
            radius_m=float(spec.wheel_radius_m),
            length_m=float(spec.wheel_width_m),
            material_name="wheel_black",
        )
        _add_inertial(wheel_link, mass_kg=float(spec.wheel_mass_kg), diagonal_inertia=wheel_inertia)

        _add_joint(
            robot,
            name=f"{corner.name}_wheel_joint",
            joint_type="continuous",
            parent_link=wheel_parent_link,
            child_link=wheel_link_name,
            xyz=(0.0, 0.0, wheel_joint_z_m),
            axis_xyz=(0.0, 1.0, 0.0),
            limit={
                "effort": float(spec.wheel_effort_limit_nm),
                "velocity": float(spec.wheel_velocity_limit_radps),
            },
            dynamics={"damping": float(spec.wheel_damping), "friction": 0.0},
        )

    _indent_xml(robot)
    return ET.tostring(robot, encoding="unicode")


def write_student_vehicle_urdf(path: str | Path, spec: StudentVehicleSpec) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urdf_text = "<?xml version=\"1.0\"?>\n" + build_student_vehicle_urdf(spec) + "\n"
    output_path.write_text(urdf_text, encoding="utf-8")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a procedural FWD student vehicle URDF from a parametric spec.")
    parser.add_argument("--output-urdf", type=str, default="artifacts/student_vehicle_assets/default/student_fwd_vehicle.urdf")
    parser.add_argument("--spec-json", type=str, default="", help="Optional JSON file overriding StudentVehicleSpec fields.")
    parser.add_argument("--output-spec-json", type=str, default="", help="Optional path to write the resolved spec JSON.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    spec = build_default_student_vehicle_spec()
    if str(args.spec_json):
        spec = load_student_vehicle_spec(args.spec_json)
    urdf_path = write_student_vehicle_urdf(args.output_urdf, spec)
    if str(args.output_spec_json):
        write_student_vehicle_spec(args.output_spec_json, spec)
    print(f"[procedural_student_vehicle] wrote URDF to {urdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
