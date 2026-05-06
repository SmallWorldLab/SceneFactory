from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import traceback

from src.physx_teacher_patch_track import _enable_extensions, _ensure_physics_scene, _ensure_world_default_prim, _set_stage_units
from src.procedural_student_vehicle import (
    StudentVehicleSpec,
    build_default_student_vehicle_spec,
    load_student_vehicle_spec,
    nominal_root_height_m,
    write_student_vehicle_spec,
    write_student_vehicle_urdf,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and import a procedural student vehicle into Isaac Sim.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-dir", type=str, default="artifacts/student_vehicle_assets/default")
    parser.add_argument("--spec-json", type=str, default="", help="Optional JSON file overriding StudentVehicleSpec fields.")
    parser.add_argument("--urdf-path", type=str, default="", help="Optional existing URDF to import instead of generating one.")
    parser.add_argument(
        "--save-vehicle-usd",
        type=str,
        default="",
        help="Optional explicit USD export path for the vehicle-only asset. Defaults to <output-dir>/<vehicle-name>.usd.",
    )
    parser.add_argument(
        "--save-stage-usd",
        type=str,
        default="",
        help="Optional explicit USD export path for the full debug stage. If omitted, only the vehicle asset is saved.",
    )
    parser.add_argument("--sim-steps", type=int, default=180)
    parser.add_argument("--spawn-x-m", type=float, default=0.0)
    parser.add_argument("--spawn-y-m", type=float, default=0.0)
    parser.add_argument("--spawn-yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--spawn-clearance-m",
        type=float,
        default=0.25,
        help="Extra vertical clearance above the nominal wheel-contact pose so settling under gravity is visible.",
    )
    return parser.parse_args()


def _build_spec(args: argparse.Namespace) -> StudentVehicleSpec:
    if str(args.spec_json):
        return load_student_vehicle_spec(args.spec_json)
    return build_default_student_vehicle_spec()


def _write_artifacts(output_dir: str | Path, spec: StudentVehicleSpec, urdf_path_arg: str) -> tuple[Path, Path]:
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    spec_path = output_root / "student_vehicle_spec.json"
    write_student_vehicle_spec(spec_path, spec)
    if str(urdf_path_arg):
        return Path(urdf_path_arg).expanduser().resolve(), spec_path
    urdf_path = output_root / "student_fwd_vehicle.urdf"
    write_student_vehicle_urdf(urdf_path, spec)
    return urdf_path, spec_path


def _spawn_height_m(spec: StudentVehicleSpec, spawn_clearance_m: float) -> float:
    return float(nominal_root_height_m(spec)) + max(0.0, float(spawn_clearance_m))


def _vehicle_asset_root_path(robot_root_path: str) -> str:
    root_name = str(robot_root_path).rstrip("/").split("/")[-1]
    if not root_name:
        raise ValueError(f"Unable to derive a vehicle asset root from robot path: {robot_root_path!r}")
    return f"/{root_name}"


def _default_vehicle_usd_path(output_dir: str | Path, spec: StudentVehicleSpec) -> Path:
    output_root = Path(output_dir).expanduser().resolve()
    return output_root / f"{spec.name}.usd"


def _vehicle_physics_material_root_path(asset_root_path: str) -> str:
    return f"{str(asset_root_path).rstrip('/')}/PhysicsMaterials"


def _vehicle_physics_material_paths(asset_root_path: str) -> dict[str, str]:
    root_path = _vehicle_physics_material_root_path(asset_root_path)
    return {
        "wheel": f"{root_path}/wheel_contact_material",
        "chassis": f"{root_path}/chassis_contact_material",
    }


def _vehicle_collision_material_bind_targets(asset_root_path: str) -> dict[str, list[str]]:
    root_path = str(asset_root_path).rstrip("/")
    return {
        "wheel": [
            f"{root_path}/front_left_wheel_link",
            f"{root_path}/front_right_wheel_link",
            f"{root_path}/rear_left_wheel_link",
            f"{root_path}/rear_right_wheel_link",
        ],
        "chassis": [
            f"{root_path}/base_link",
        ],
    }


def _vehicle_subtree_should_be_deinstanced(*, prim_name: str, has_references: bool, is_instanceable: bool) -> bool:
    return bool(is_instanceable and has_references and prim_name in {"visuals", "collisions"})


def _rewrite_layer_material_bindings(
    layer,
    *,
    source_vehicle_path,
    asset_root_path,
) -> int:
    from pxr import Sdf

    source_looks_prefix = Sdf.Path(f"{str(source_vehicle_path)}/Looks")
    asset_looks_prefix = Sdf.Path(f"{str(asset_root_path)}/Looks")
    rewritten_count = 0

    def _visit_prim_spec(prim_spec) -> None:
        nonlocal rewritten_count
        for property_name in list(prim_spec.properties.keys()):
            if not property_name.startswith("material:binding"):
                continue
            relationship_spec = prim_spec.relationships[property_name]
            targets = list(relationship_spec.targetPathList.explicitItems)
            if not targets:
                continue
            updated_targets = []
            changed = False
            for target in targets:
                updated_target = target
                if target.HasPrefix(source_looks_prefix):
                    updated_target = target.ReplacePrefix(source_looks_prefix, asset_looks_prefix)
                if updated_target != target:
                    changed = True
                updated_targets.append(updated_target)
            if changed:
                relationship_spec.targetPathList.explicitItems = updated_targets
                rewritten_count += 1
        for child_spec in prim_spec.nameChildren.values():
            _visit_prim_spec(child_spec)

    for root_spec in layer.rootPrims:
        _visit_prim_spec(root_spec)
    return rewritten_count


def _author_vehicle_physics_materials(vehicle_stage, *, asset_root_path: str) -> dict[str, str]:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    material_paths = _vehicle_physics_material_paths(asset_root_path)
    bind_targets = _vehicle_collision_material_bind_targets(asset_root_path)

    def _create_material(material_path: str, *, static_friction: float, dynamic_friction: float) -> UsdShade.Material:
        material = UsdShade.Material.Define(vehicle_stage, material_path)
        prim = material.GetPrim()

        physics_material_api = UsdPhysics.MaterialAPI(prim)
        if not physics_material_api:
            physics_material_api = UsdPhysics.MaterialAPI.Apply(prim)
        physics_material_api.CreateStaticFrictionAttr().Set(float(static_friction))
        physics_material_api.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
        physics_material_api.CreateRestitutionAttr().Set(0.0)

        physx_material_api = PhysxSchema.PhysxMaterialAPI(prim)
        if not physx_material_api:
            physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(prim)
        physx_material_api.CreateFrictionCombineModeAttr().Set("min")
        physx_material_api.CreateRestitutionCombineModeAttr().Set("min")
        return material

    wheel_material = _create_material(material_paths["wheel"], static_friction=1.0, dynamic_friction=1.0)
    chassis_material = _create_material(material_paths["chassis"], static_friction=1.0, dynamic_friction=1.0)

    for link_path in bind_targets["wheel"]:
        link_prim = vehicle_stage.GetPrimAtPath(link_path)
        if not link_prim.IsValid():
            raise RuntimeError(f"Wheel link is missing from exported vehicle asset: {link_path}")
        UsdShade.MaterialBindingAPI(link_prim).Bind(wheel_material)

    for link_path in bind_targets["chassis"]:
        link_prim = vehicle_stage.GetPrimAtPath(link_path)
        if not link_prim.IsValid():
            raise RuntimeError(f"Chassis link is missing from exported vehicle asset: {link_path}")
        UsdShade.MaterialBindingAPI(link_prim).Bind(chassis_material)

    return material_paths


def _deinstance_vehicle_reference_subtrees(vehicle_stage, *, asset_root_path: str) -> int:
    from pxr import Usd

    asset_root_prim = vehicle_stage.GetPrimAtPath(str(asset_root_path))
    if not asset_root_prim.IsValid():
        raise RuntimeError(f"Vehicle asset root does not exist in exported stage: {asset_root_path}")

    changed_count = 0
    for prim in Usd.PrimRange(asset_root_prim):
        if not prim.IsValid():
            continue
        if _vehicle_subtree_should_be_deinstanced(
            prim_name=prim.GetName(),
            has_references=prim.HasAuthoredReferences(),
            is_instanceable=prim.IsInstanceable(),
        ):
            prim.SetInstanceable(False)
            changed_count += 1
    return changed_count


def _inline_vehicle_reference_subtrees(vehicle_stage, *, asset_root_path: str) -> int:
    from pxr import Sdf, Usd

    layer = vehicle_stage.GetRootLayer()
    asset_root_prim = vehicle_stage.GetPrimAtPath(str(asset_root_path))
    if not asset_root_prim.IsValid():
        raise RuntimeError(f"Vehicle asset root does not exist in exported stage: {asset_root_path}")

    changed_count = 0

    def _reference_prim_path(prim_spec) -> Sdf.Path | None:
        list_op = prim_spec.referenceList
        for attr_name in ("explicitItems", "prependedItems", "appendedItems", "addedItems", "orderedItems"):
            ref_items = list(getattr(list_op, attr_name, []) or [])
            if ref_items:
                ref_path = ref_items[0].primPath
                if ref_path and not ref_path.isEmpty:
                    return ref_path
        return None

    for prim in Usd.PrimRange(asset_root_prim):
        if not prim.IsValid():
            continue
        prim_path = prim.GetPath()
        prim_spec = layer.GetPrimAtPath(prim_path)
        if prim_spec is None:
            continue
        if not _vehicle_subtree_should_be_deinstanced(
            prim_name=prim.GetName(),
            has_references=prim.HasAuthoredReferences(),
            is_instanceable=prim.IsInstanceable(),
        ):
            continue

        source_ref_path = _reference_prim_path(prim_spec)
        if source_ref_path is None:
            prim.SetInstanceable(False)
            changed_count += 1
            continue

        source_ref_prim = vehicle_stage.GetPrimAtPath(source_ref_path)
        source_ref_spec = layer.GetPrimAtPath(source_ref_path)
        if not source_ref_prim.IsValid() and source_ref_spec is None:
            raise RuntimeError(f"Vehicle subtree reference target is missing: {source_ref_path}")

        copied_any_child = False
        if source_ref_spec is not None:
            for child_spec in source_ref_spec.nameChildren.values():
                dest_child_path = prim_path.AppendChild(child_spec.name)
                if layer.GetPrimAtPath(dest_child_path) is None:
                    if not Sdf.CopySpec(layer, child_spec.path, layer, dest_child_path):
                        raise RuntimeError(f"Failed to inline {child_spec.path} into {dest_child_path}")
                copied_any_child = True

        if not copied_any_child:
            # Some references point directly at a leaf prim rather than a container prim.
            # In that case we inline that leaf under visuals/collisions as <leaf-name>.
            dest_name = source_ref_path.name
            dest_child_path = prim_path.AppendChild(dest_name)
            if layer.GetPrimAtPath(dest_child_path) is None:
                if not Sdf.CopySpec(layer, source_ref_path, layer, dest_child_path):
                    raise RuntimeError(f"Failed to inline {source_ref_path} into {dest_child_path}")

        prim.GetReferences().ClearReferences()
        prim.SetInstanceable(False)
        changed_count += 1

    return changed_count


def _validate_vehicle_subtrees_populated(vehicle_stage, *, asset_root_path: str) -> None:
    from pxr import Usd

    asset_root_prim = vehicle_stage.GetPrimAtPath(str(asset_root_path))
    if not asset_root_prim.IsValid():
        raise RuntimeError(f"Vehicle asset root does not exist in exported stage: {asset_root_path}")

    empty_subtrees: list[str] = []
    for prim in Usd.PrimRange(asset_root_prim):
        if not prim.IsValid():
            continue
        prim_name = prim.GetName()
        if prim_name not in {"visuals", "collisions"}:
            continue
        parent_name = prim.GetParent().GetName() if prim.GetParent().IsValid() else ""
        if prim_name == "collisions" and parent_name not in {
            "base_link",
            "front_left_wheel_link",
            "front_right_wheel_link",
            "rear_left_wheel_link",
            "rear_right_wheel_link",
        }:
            continue
        if prim.HasAuthoredReferences():
            continue
        if any(True for _ in prim.GetChildren()):
            continue
        empty_subtrees.append(str(prim.GetPath()))

    if empty_subtrees:
        joined_paths = ", ".join(empty_subtrees[:8])
        if len(empty_subtrees) > 8:
            joined_paths += ", ..."
        raise RuntimeError(
            "Exported vehicle asset contains empty visuals/collisions subtrees after reference rewrite: "
            f"{joined_paths}"
        )


def _remove_flattened_prototypes(vehicle_stage) -> int:
    removed_count = 0
    root_children = list(vehicle_stage.GetPseudoRoot().GetChildren())
    for prim in root_children:
        if not prim.IsValid():
            continue
        if prim.GetName().startswith("Flattened_Prototype_"):
            vehicle_stage.RemovePrim(prim.GetPath())
            removed_count += 1
    return removed_count


def main() -> int:
    args = _parse_args()
    output_root = Path(args.output_dir).expanduser().resolve()
    spec = _build_spec(args)
    urdf_path, spec_path = _write_artifacts(output_root, spec, args.urdf_path)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": bool(args.headless)})
    try:
        _enable_extensions(["isaacsim.asset.importer.urdf"])

        import omni.kit.app
        import omni.kit.commands
        import omni.timeline
        import omni.usd
        from pxr import Gf, PhysicsSchemaTools, Sdf, Usd, UsdGeom, UsdLux

        app = omni.kit.app.get_app()
        for _ in range(10):
            app.update()

        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        _ensure_world_default_prim(stage)
        _set_stage_units(stage)
        _ensure_physics_scene(stage)

        PhysicsSchemaTools.addGroundPlane(stage, "/World/GroundPlane", "Z", 50.0, Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(0.5))
        light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
        light.CreateIntensityAttr(700.0)

        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        if not status:
            raise RuntimeError("URDFCreateImportConfig failed")
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.import_inertia_tensor = True
        import_config.fix_base = False
        import_config.collision_from_visuals = False
        import_config.parse_mimic = True
        try:
            from isaacsim.asset.importer.urdf._urdf import UrdfJointTargetType

            import_config.default_drive_type = UrdfJointTargetType.JOINT_DRIVE_NONE
        except Exception:
            pass

        import_result = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=str(urdf_path),
            import_config=import_config,
        )
        if isinstance(import_result, tuple) and len(import_result) == 2 and isinstance(import_result[0], bool):
            import_status, robot_root_path = import_result
            if not import_status:
                raise RuntimeError(f"URDFParseAndImportFile failed: {import_result!r}")
        else:
            robot_root_path = import_result
        if not isinstance(robot_root_path, str) or not robot_root_path:
            raise RuntimeError(f"URDFParseAndImportFile returned an invalid robot path: {import_result!r}")

        for _ in range(5):
            app.update()

        root_prim = stage.GetPrimAtPath(robot_root_path)
        if not root_prim.IsValid():
            raise RuntimeError(f"Imported robot root does not exist: {robot_root_path}")

        xformable = UsdGeom.Xformable(root_prim)
        translate_op = None
        rotate_op = None
        for op in xformable.GetOrderedXformOps():
            if op.IsInverseOp():
                continue
            if translate_op is None and op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
            if rotate_op is None and op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                rotate_op = op
        if translate_op is None:
            translate_op = xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble, opSuffix="studentSpawn")
        if rotate_op is None:
            rotate_op = xformable.AddRotateXYZOp(precision=UsdGeom.XformOp.PrecisionFloat, opSuffix="studentSpawn")

        spawn_z_m = _spawn_height_m(spec, args.spawn_clearance_m)
        translate_op.Set(Gf.Vec3d(float(args.spawn_x_m), float(args.spawn_y_m), float(spawn_z_m)))
        rotate_op.Set(Gf.Vec3f(0.0, 0.0, float(args.spawn_yaw_deg)))

        for _ in range(5):
            simulation_app.update()

        base_link_prim = stage.GetPrimAtPath(f"{robot_root_path}/base_link")
        base_link_world_position_before_sim_m = None
        if base_link_prim.IsValid():
            before_translation = omni.usd.get_world_transform_matrix(base_link_prim).ExtractTranslation()
            base_link_world_position_before_sim_m = [
                float(before_translation[0]),
                float(before_translation[1]),
                float(before_translation[2]),
            ]

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(max(1, int(args.sim_steps))):
            simulation_app.update()
        timeline.stop()

        base_link_world_position_after_sim_m = None
        if base_link_prim.IsValid():
            after_translation = omni.usd.get_world_transform_matrix(base_link_prim).ExtractTranslation()
            base_link_world_position_after_sim_m = [
                float(after_translation[0]),
                float(after_translation[1]),
                float(after_translation[2]),
            ]

        def _export_vehicle_asset_usd(source_stage, source_robot_root_path: str, output_path: Path) -> tuple[Path, str]:
            flattened_layer = source_stage.Flatten()
            source_vehicle_path = Sdf.Path(str(source_robot_root_path))
            if not flattened_layer.GetPrimAtPath(source_vehicle_path):
                raise RuntimeError(f"Flattened stage is missing vehicle prim at {source_vehicle_path}")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            vehicle_layer = Sdf.Layer.CreateNew(str(output_path))
            if vehicle_layer is None:
                raise RuntimeError(f"Failed to create vehicle asset layer: {output_path}")

            asset_root_path = Sdf.Path(_vehicle_asset_root_path(source_robot_root_path))
            if not Sdf.CopySpec(flattened_layer, source_vehicle_path, vehicle_layer, asset_root_path):
                raise RuntimeError(f"Failed to copy vehicle prim from {source_vehicle_path} to {asset_root_path}")

            for root_spec in flattened_layer.rootPrims:
                root_spec_path = root_spec.path
                if root_spec_path.name.startswith("Flattened_Prototype_"):
                    if not Sdf.CopySpec(flattened_layer, root_spec_path, vehicle_layer, root_spec_path):
                        raise RuntimeError(f"Failed to copy prototype prim {root_spec_path}")

            _rewrite_layer_material_bindings(
                vehicle_layer,
                source_vehicle_path=source_vehicle_path,
                asset_root_path=asset_root_path,
            )

            vehicle_stage = Usd.Stage.Open(vehicle_layer)
            if vehicle_stage is None:
                raise RuntimeError(f"Failed to open vehicle asset stage: {output_path}")
            UsdGeom.SetStageUpAxis(vehicle_stage, UsdGeom.GetStageUpAxis(source_stage))
            UsdGeom.SetStageMetersPerUnit(vehicle_stage, UsdGeom.GetStageMetersPerUnit(source_stage))

            vehicle_root_prim = vehicle_stage.GetPrimAtPath(asset_root_path)
            if not vehicle_root_prim.IsValid():
                raise RuntimeError(f"Vehicle asset root does not exist in exported stage: {asset_root_path}")
            _author_vehicle_physics_materials(vehicle_stage, asset_root_path=str(asset_root_path))
            _inline_vehicle_reference_subtrees(vehicle_stage, asset_root_path=str(asset_root_path))
            _deinstance_vehicle_reference_subtrees(vehicle_stage, asset_root_path=str(asset_root_path))
            _validate_vehicle_subtrees_populated(vehicle_stage, asset_root_path=str(asset_root_path))
            _remove_flattened_prototypes(vehicle_stage)
            vehicle_stage.SetDefaultPrim(vehicle_root_prim)
            vehicle_stage.GetRootLayer().Save()
            return output_path, str(asset_root_path)

        vehicle_usd_path = (
            Path(args.save_vehicle_usd).expanduser().resolve()
            if str(args.save_vehicle_usd)
            else _default_vehicle_usd_path(output_root, spec)
        )
        vehicle_usd_path, vehicle_root_path = _export_vehicle_asset_usd(stage, robot_root_path, vehicle_usd_path)

        stage_usd_path = ""
        if str(args.save_stage_usd):
            debug_stage_usd_path = Path(args.save_stage_usd).expanduser().resolve()
            debug_stage_usd_path.parent.mkdir(parents=True, exist_ok=True)
            stage.Export(str(debug_stage_usd_path))
            stage_usd_path = str(debug_stage_usd_path)

        meta = {
            "robot_root_path": str(robot_root_path),
            "vehicle_root_path": str(vehicle_root_path),
            "urdf_path": str(urdf_path),
            "spec_path": str(spec_path),
            "vehicle_usd_path": str(vehicle_usd_path),
            "sim_steps": int(args.sim_steps),
            "spawn": {
                "x_m": float(args.spawn_x_m),
                "y_m": float(args.spawn_y_m),
                "yaw_deg": float(args.spawn_yaw_deg),
                "z_m": float(spawn_z_m),
                "clearance_m": float(args.spawn_clearance_m),
            },
            "base_link_world_position_before_sim_m": base_link_world_position_before_sim_m,
            "base_link_world_position_after_sim_m": base_link_world_position_after_sim_m,
            "spec": asdict(spec),
        }
        if stage_usd_path:
            meta["stage_usd_path"] = stage_usd_path
        meta_path = output_root / "student_vehicle_import_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"[procedural_student_vehicle_import] imported student vehicle to {robot_root_path}", flush=True)
        print(f"[procedural_student_vehicle_import] wrote vehicle asset to {vehicle_usd_path}", flush=True)
        if stage_usd_path:
            print(f"[procedural_student_vehicle_import] wrote debug stage to {stage_usd_path}", flush=True)
        print(f"[procedural_student_vehicle_import] wrote metadata to {meta_path}", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
