# launch Isaac Sim before any other imports
# default first two lines in any standalone application
import argparse
import json
from pathlib import Path
from typing import Literal, Union

import numpy as np
from isaacsim import SimulationApp

app = SimulationApp({"headless": False})  # we can also run as headless.

import omni.kit.actions.core
from isaacsim.core.api import World
from isaacsim.core.utils import extensions
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import Sdf, Usd, UsdGeom


def switch_lighting(mode: Literal["camera", "stage"] = "camera"):
    # switch lighting
    action_registry = omni.kit.actions.core.get_action_registry()
    action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_" + mode)
    action.execute()


def get_visibility_attribute(stage: Usd.Stage, prim_path: str) -> Union[Usd.Attribute, None]:
    """Return the visibility attribute of a prim"""
    path = Sdf.Path(prim_path)
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return None
    visibility_attribute = prim.GetAttribute("visibility")
    return visibility_attribute


def hide_prim(stage: Usd.Stage, prim_path: str):
    """Hide a prim

    Args:
        stage (Usd.Stage, required): The USD Stage
        prim_path (str, required): The prim path of the prim to hide
    """
    visibility_attribute = get_visibility_attribute(stage, prim_path)
    if visibility_attribute is None:
        return
    visibility_attribute.Set("invisible")


def show_prim(stage: Usd.Stage, prim_path: str):
    """Show a prim

    Args:
        stage (Usd.Stage, required): The USD Stage
        prim_path (str, required): The prim path of the prim to show
    """
    visibility_attribute = get_visibility_attribute(stage, prim_path)
    if visibility_attribute is None:
        return
    visibility_attribute.Set("inherited")


def dump_state(
    time: float,
    position: tuple[float, 3],
    orientation: tuple[float, 4],
    linear_velocity: tuple[float, 3],
):
    data = {
        "time": time,
        "position": {"x": position[0], "y": position[1], "z": position[2]},
        "orientation": {
            "w": orientation[0],
            "x": orientation[1],
            "y": orientation[2],
            "z": orientation[3],
        },
        "linear_velocity": {
            "vx": linear_velocity[0],
            "vy": linear_velocity[1],
            "vz": linear_velocity[2],
        },
    }
    print(json.dumps(data))


def parse_assets(raw_assets):
    assets = []
    for name, x, y, z, theta in raw_assets or []:
        assets.append((name, float(x), float(y), float(z), float(theta)))
    return assets


def main(simulation_app):
    parser = argparse.ArgumentParser(description="Run standalone simulation with optional scene selection.")
    parser.add_argument(
        "--scene",
        type=Path,
        help="Path to the USD scene file to load.",
        default=None,
    )
    parser.add_argument(
        "--lighting",
        type=str,
        choices=["camera", "stage"],
        default="stage",
        help="Lighting mode to use.",
    )
    parser.add_argument(
        "--asset",
        nargs=5,
        action="append",
        metavar=("NAME", "X", "Y", "Z", "THETA"),
        help="Asset definition: name x y z theta (in deg) (can be provided multiple times)",
        default=[],
    )
    args = parser.parse_args()
    args.asset = parse_assets(args.asset)

    extensions.enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()

    root_prim = "/map"

    world = World()
    ground_plane = world.scene.add_ground_plane(prim_path=root_prim + "/defaultGroundPlane", z_position=0.05)
    if args.scene is not None:
        hide_prim(world.stage, ground_plane.prim_path)
        switch_lighting(mode=args.lighting)
        _scene = add_reference_to_stage(usd_path=str(args.scene), prim_path=root_prim)
    else:
        switch_lighting(mode="camera")

    # load robot
    stretch_asset_path = "/home/benni/repos/stretch_isaac/importable_stretch.usd"
    prim_stretch = add_reference_to_stage(usd_path=stretch_asset_path, prim_path=root_prim)

    for id, asset in enumerate(args.asset):
        asset_usd_path, x, y, z, theta = asset
        name = Path(asset_usd_path).stem
        print(
            f"Adding asset '{name}' at position ({x}, {y}, {z}) with rotation {theta} and asset path '{asset_usd_path}'"
        )
        prim_asset = add_reference_to_stage(usd_path=str(asset_usd_path), prim_path=f"{root_prim}/{name}_{id}")
        prim_asset.GetAttribute("xformOp:translate").Set((x, y, z))
        xform = UsdGeom.Xformable(prim_asset)
        xform.AddRotateZOp().Set(float(theta))

    world.reset()

    stretch = Articulation(prim_path=str(prim_stretch.GetPath()) + "/stretch")
    stretch.initialize()

    print_pose_interval: int = 33
    try:
        step_count = 0
        while simulation_app.is_running():
            world.step(render=True)  # execute one physics step and one rendering step
            step_count += 1
            if step_count == print_pose_interval:
                step_count = 0
                position: np.ndarray
                orientation: np.ndarray
                position, orientation = stretch.get_world_pose()
                linear_velocity: np.ndarray = stretch.get_linear_velocity()
                dump_state(
                    float(world.current_time),
                    position.tolist(),
                    orientation.tolist(),
                    linear_velocity.tolist(),
                )
    except KeyboardInterrupt:
        print("Exiting simulation...")

    simulation_app.close()


if __name__ == "__main__":
    main(app)
