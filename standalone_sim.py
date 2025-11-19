# launch Isaac Sim before any other imports
# default first two lines in any standalone application
from typing import Literal, Union
from isaacsim import SimulationApp

app = SimulationApp({"headless": False})  # we can also run as headless.

from isaacsim.core.api import World
from isaacsim.core.utils import extensions
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.kit.actions.core
from pxr import Sdf, Usd


def switch_lighting(mode: Literal["camera", "stage"] = "camera"):
    # switch lighting
    action_registry = omni.kit.actions.core.get_action_registry()
    action = action_registry.get_action(
        "omni.kit.viewport.menubar.lighting", "set_lighting_mode_" + mode
    )
    action.execute()


def get_visibility_attribute(
    stage: Usd.Stage, prim_path: str
) -> Union[Usd.Attribute, None]:
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


def main(simulation_app):
    extensions.enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()

    root_prim = "/map"

    world = World()
    ground_plane = world.scene.add_ground_plane(
        prim_path=root_prim + "/defaultGroundPlane", z_position=0.02
    )
    hide_prim(world.stage, ground_plane.prim_path)

    # load robot
    stretch_asset_path = "/home/benni/repos/stretch_isaac/importable_stretch.usd"
    stretch = add_reference_to_stage(usd_path=stretch_asset_path, prim_path=root_prim)

    use_matterport3d: bool = True

    if use_matterport3d:
        scene_asset_path = "/home/benni/datasets/hm3d-minival-glb-v0.2/00800-TEEsavR23oF/TEEsavR23oF_collision.usd"
        switch_lighting("camera")
    else:
        scene_asset_path = (
            "/home/benni/datasets/InteriorAgent/kujiale_0003/kujiale_0003.usda"
        )
        switch_lighting("stage")
    scene = add_reference_to_stage(usd_path=scene_asset_path, prim_path=root_prim)

    world.reset()

    try:
        while simulation_app.is_running():
            world.step(render=True)  # execute one physics step and one rendering step
    except KeyboardInterrupt:
        print("Exiting simulation...")

    simulation_app.close()


if __name__ == "__main__":
    main(app)
