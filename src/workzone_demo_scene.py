"""workzone_demo_scene.py
SceneFactory workzone demo: multiple worlds each with a straight road,
fixed workzone box, and randomized taper-zone traffic cones.

Launch:
    bash run_workzone_demo.sh
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="SceneFactory workzone taper-zone demo.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/scene_factory/workzone_demo.yaml",
    )
    parser.add_argument("--num_worlds", type=int, default=-1,
                        help="Override world count from config.")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Override random seed.")
    parser.add_argument("--save_stage_usd", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output_dir", type=str, default="artifacts/scene_factory/workzone_demo")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# USD cone builder
# ---------------------------------------------------------------------------

def _spawn_traffic_cone(
    stage: Any,
    prim_path: str,
    x: float,
    y: float,
    z: float = 0.0,
    cone_height_m: float = 0.72,
    cone_base_radius_m: float = 0.20,
) -> None:
    """Spawn a single traffic cone (orange cone + white band) at world position (x, y, z)."""
    from pxr import Gf, UsdGeom, UsdShade, Sdf, Vt

    xform = UsdGeom.Xform.Define(stage, prim_path)
    UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(x, y, z))

    # Body: cone shape approximated as a tapered cylinder using UsdGeom.Cone
    cone_path = f"{prim_path}/Body"
    cone = UsdGeom.Cone.Define(stage, cone_path)
    cone.CreateRadiusAttr(cone_base_radius_m)
    cone.CreateHeightAttr(cone_height_m)
    cone.CreateAxisAttr("Z")
    # translate cone center up by half-height so base sits at z=0
    body_xform = UsdGeom.XformCommonAPI(cone)
    body_xform.SetTranslate(Gf.Vec3d(0.0, 0.0, cone_height_m * 0.5))

    # Orange material
    mat_path = f"{prim_path}/OrangeMat"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader_path = f"{mat_path}/Shader"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.95, 0.35, 0.02))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(cone.GetPrim()).Bind(mat)

    # White reflective band (thin torus-like ring — approximated with cylinder)
    band_path = f"{prim_path}/Band"
    band = UsdGeom.Cylinder.Define(stage, band_path)
    band.CreateRadiusAttr(cone_base_radius_m * 0.65)
    band.CreateHeightAttr(0.06)
    band.CreateAxisAttr("Z")
    UsdGeom.XformCommonAPI(band).SetTranslate(Gf.Vec3d(0.0, 0.0, cone_height_m * 0.55))

    band_mat_path = f"{prim_path}/WhiteMat"
    band_mat = UsdShade.Material.Define(stage, band_mat_path)
    band_shader = UsdShade.Shader.Define(stage, f"{band_mat_path}/Shader")
    band_shader.CreateIdAttr("UsdPreviewSurface")
    band_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.95, 0.95, 0.95))
    band_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.2)
    band_mat.CreateSurfaceOutput().ConnectToSource(band_shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(band.GetPrim()).Bind(band_mat)


def _spawn_workzone_box(
    stage: Any,
    prim_path: str,
    center_x: float,
    center_y: float,
    length_m: float,
    width_m: float,
    z: float = 0.01,
    height_m: float = 0.05,
) -> None:
    """Spawn a flat box marking the workzone footprint (red, semi-transparent look)."""
    from pxr import Gf, UsdGeom, UsdShade, Sdf

    xform = UsdGeom.Xform.Define(stage, prim_path)
    UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(center_x, center_y, z + height_m * 0.5))

    cube = UsdGeom.Cube.Define(stage, f"{prim_path}/Box")
    # USD Cube is a unit cube; scale it
    from pxr import UsdGeom as UG
    ops = cube.AddXformOp(UG.XformOp.TypeScale)
    from pxr import Vt, Gf as _Gf
    ops.Set(_Gf.Vec3f(length_m * 0.5, width_m * 0.5, height_m * 0.5))

    mat_path = f"{prim_path}/RedMat"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(_Gf.Vec3f(0.85, 0.08, 0.08))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(mat)


def _spawn_ground_plane(
    stage: Any,
    prim_path: str,
    size_m: float = 300.0,
    z: float = -0.005,
) -> None:
    """Thin grey ground plane."""
    from pxr import Gf, UsdGeom, UsdShade, Sdf

    xform = UsdGeom.Xform.Define(stage, prim_path)
    UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(0.0, 0.0, z))

    cube = UsdGeom.Cube.Define(stage, f"{prim_path}/Plane")
    ops = cube.AddXformOp(UsdGeom.XformOp.TypeScale)
    ops.Set(Gf.Vec3f(size_m * 0.5, size_m * 0.5, 0.005))

    mat_path = f"{prim_path}/GreyMat"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.32, 0.32, 0.32))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(mat)


# ---------------------------------------------------------------------------
# World builder
# ---------------------------------------------------------------------------

def build_workzone_world(
    stage: Any,
    world_root: str,
    scene: dict[str, Any],
    *,
    world_offset_x: float = 0.0,
    world_offset_y: float = 0.0,
    road_cfg: dict[str, Any],
) -> None:
    """Build one mini-world at the given XY offset."""
    from pxr import Gf, UsdGeom

    # World xform — all children are in local coords
    world_xform = UsdGeom.Xform.Define(stage, world_root)
    UsdGeom.XformCommonAPI(world_xform).SetTranslate(Gf.Vec3d(world_offset_x, world_offset_y, 0.0))

    road_length = float(scene["meta"].get("road_length_m", 200.0))
    road_half_w = float(scene["meta"].get("road_half_width_m", 7.4))

    # Ground plane
    _spawn_ground_plane(stage, f"{world_root}/Ground", size_m=road_length + 40.0)

    # Road surface (dark grey road-coloured box)
    from pxr import Gf as _Gf, UsdGeom as _UG, UsdShade, Sdf
    road_xform = UsdGeom.Xform.Define(stage, f"{world_root}/Road")
    UsdGeom.XformCommonAPI(road_xform).SetTranslate(_Gf.Vec3d(0.0, 0.0, 0.001))
    road_cube = _UG.Cube.Define(stage, f"{world_root}/Road/Surface")
    ops = road_cube.AddXformOp(_UG.XformOp.TypeScale)
    ops.Set(_Gf.Vec3f((road_length * 0.5), road_half_w, 0.003))
    road_mat = UsdShade.Material.Define(stage, f"{world_root}/Road/RoadMat")
    road_shader = UsdShade.Shader.Define(stage, f"{world_root}/Road/RoadMat/Shader")
    road_shader.CreateIdAttr("UsdPreviewSurface")
    road_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(_Gf.Vec3f(0.18, 0.18, 0.18))
    road_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    road_mat.CreateSurfaceOutput().ConnectToSource(road_shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(road_cube.GetPrim()).Bind(road_mat)

    # Road edge lines (white boxes)
    for sign, name in [(-1, "LeftEdge"), (1, "RightEdge")]:
        edge_y = sign * road_half_w
        edge_xform = UsdGeom.Xform.Define(stage, f"{world_root}/Road/{name}")
        UsdGeom.XformCommonAPI(edge_xform).SetTranslate(_Gf.Vec3d(0.0, edge_y, 0.005))
        ec = _UG.Cube.Define(stage, f"{world_root}/Road/{name}/Line")
        ops2 = ec.AddXformOp(_UG.XformOp.TypeScale)
        ops2.Set(_Gf.Vec3f(road_length * 0.5, 0.08, 0.004))
        em = UsdShade.Material.Define(stage, f"{world_root}/Road/{name}/Mat")
        es = UsdShade.Shader.Define(stage, f"{world_root}/Road/{name}/Mat/S")
        es.CreateIdAttr("UsdPreviewSurface")
        es.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(_Gf.Vec3f(1.0, 1.0, 1.0))
        em.CreateSurfaceOutput().ConnectToSource(es.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(ec.GetPrim()).Bind(em)

    # Center broken divider (yellow dashes — simplified as a continuous line for now)
    ctr_xform = UsdGeom.Xform.Define(stage, f"{world_root}/Road/CenterLine")
    UsdGeom.XformCommonAPI(ctr_xform).SetTranslate(_Gf.Vec3d(0.0, 0.0, 0.005))
    cc = _UG.Cube.Define(stage, f"{world_root}/Road/CenterLine/Line")
    ops3 = cc.AddXformOp(_UG.XformOp.TypeScale)
    ops3.Set(_Gf.Vec3f(road_length * 0.5, 0.05, 0.004))
    ym = UsdShade.Material.Define(stage, f"{world_root}/Road/CenterLine/Mat")
    ys = UsdShade.Shader.Define(stage, f"{world_root}/Road/CenterLine/Mat/S")
    ys.CreateIdAttr("UsdPreviewSurface")
    ys.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(_Gf.Vec3f(0.95, 0.85, 0.0))
    ym.CreateSurfaceOutput().ConnectToSource(ys.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(cc.GetPrim()).Bind(ym)

    # Workzone box
    workzone = scene.get("workzone", {})
    cone_cfg = workzone.get("cone_cfg", {})
    wz_len = float(workzone.get("workzone_length_m", 30.0))  # fixed across all worlds
    lane_w = float(cone_cfg.get("lane_width_m", 3.7))
    # Rightmost lane center: road has 2 lanes per direction, so rightmost lane
    # spans (lane_w → 2*lane_w) with center at lane_w * 1.5
    wz_center_y = lane_w * 1.5
    _spawn_workzone_box(
        stage, f"{world_root}/Workzone",
        center_x=0.0, center_y=wz_center_y,
        length_m=wz_len, width_m=lane_w,
    )

    # Traffic cones
    cones_root = f"{world_root}/TrafficCones"
    UsdGeom.Xform.Define(stage, cones_root)
    for ci, cone in enumerate(workzone.get("cones", [])):
        _spawn_traffic_cone(
            stage,
            f"{cones_root}/cone_{ci:03d}",
            x=float(cone["x"]),
            y=float(cone["y"]),
            z=float(cone.get("z", 0.05)),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import os
    os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp_cache")

    parser = _build_parser()
    args = parser.parse_args()

    from isaaclab.app import AppLauncher
    launcher = AppLauncher(args)
    sim_app = launcher.app

    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext

    cfg_dict = _load_yaml(args.config)
    world_cfg  = cfg_dict.get("world", {})
    viewer_cfg = cfg_dict.get("viewer", {})

    num_worlds = args.num_worlds if args.num_worlds > 0 else int(world_cfg.get("num_worlds", 6))
    seed = args.seed if args.seed >= 0 else int(world_cfg.get("seed", 42))
    road_length_m = float(world_cfg.get("road_length_m", 200.0))
    grid_cols = int(world_cfg.get("grid_cols", 3))
    world_spacing_x = float(world_cfg.get("world_spacing_x_m", 250.0))
    world_spacing_y = float(world_cfg.get("world_spacing_y_m", 30.0))

    # --- Generate scene JSONs ---
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.workzone_scene_generator import write_workzone_scenes
    scene_dir = Path(args.output_dir) / "scenes"
    workzone_length_m = float(world_cfg.get("workzone_length_m", 30.0))
    print(f"\n[workzone_demo] Generating {num_worlds} workzone scenes …")
    print(f"  Fixed workzone: {workzone_length_m:.0f} m  |  Taper design randomized per world")
    scene_paths = write_workzone_scenes(scene_dir, num_worlds, seed, road_length_m, workzone_length_m)

    # --- Build USD stage ---
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(
        eye=tuple(viewer_cfg.get("eye", [0.0, -200.0, 120.0])),
        target=tuple(viewer_cfg.get("lookat", [0.0, 0.0, 0.0])),
    )

    import omni.usd
    stage = omni.usd.get_context().get_stage()

    # Sky/ambient light
    from pxr import UsdLux
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(float(viewer_cfg.get("light_intensity", 2000.0)))

    print(f"[workzone_demo] Building {num_worlds} worlds on USD stage …")
    for i, scene_path in enumerate(scene_paths):
        scene = json.loads(scene_path.read_text())
        col = i % grid_cols
        row = i // grid_cols
        ox = col * world_spacing_x - (grid_cols - 1) * world_spacing_x / 2.0
        oy = row * world_spacing_y
        world_root = f"/World/WorkzoneWorlds/world_{i:03d}"
        build_workzone_world(
            stage, world_root, scene,
            world_offset_x=ox,
            world_offset_y=oy,
            road_cfg=cfg_dict.get("road", {}),
        )
        print(f"  world {i:02d}: offset=({ox:.0f}, {oy:.0f})  "
              f"cones={len(scene['workzone']['cones'])}")

    # Save stage
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_stage_usd:
        usd_path = str(out_dir / "workzone_demo.usda")
        stage.Export(usd_path)
        print(f"[workzone_demo] Stage exported → {usd_path}")

    # Run simulation / viewer
    sim_steps = int(world_cfg.get("sim_steps", 0))
    print(f"[workzone_demo] Simulation running (steps={sim_steps or 'until closed'}) …")
    sim.reset()
    step = 0
    while sim_app.is_running():
        sim.step()
        step += 1
        if sim_steps > 0 and step >= sim_steps:
            break

    sim_app.close()


if __name__ == "__main__":
    main()
