# launch Isaac Sim before any other imports
# default first two lines in any standalone application
from typing import Literal
from isaacsim import SimulationApp


def switch_lighting(mode: Literal["camera", "stage"] = "camera"):
    # switch lighting
    action_registry = omni.kit.actions.core.get_action_registry()
    action = action_registry.get_action(
        "omni.kit.viewport.menubar.lighting", "set_lighting_mode_" + mode
    )
    action.execute()


def main(simulation_app):
    extensions.enable_extension("isaacsim.ros2.bridge")

    simulation_app.update()

    world = World()
    world.scene.add_default_ground_plane()

    # load robot
    stretch_asset_path = "/home/benni/repos/stretch_isaac/importable_stretch.usd"
    stretch = add_reference_to_stage(usd_path=stretch_asset_path, prim_path="/map")

    use_matterport3d: bool = True

    if use_matterport3d:
        scene_asset_path = "/home/benni/datasets/hm3d-minival-glb-v0.2/00800-TEEsavR23oF/TEEsavR23oF_collision.usd"
        switch_lighting("camera")
    else:
        scene_asset_path = (
            "/home/benni/repos/InteriorAgent/kujiale_0003/kujiale_0003.usda"
        )
        switch_lighting("stage")
    scene = add_reference_to_stage(usd_path=scene_asset_path, prim_path="/map")

    world.reset()

    for i in range(5000):
        world.step(render=True)  # execute one physics step and one rendering step

    simulation_app.close()


if __name__ == "__main__":
    app = SimulationApp({"headless": False})  # we can also run as headless.

    from isaacsim.core.api import World
    from isaacsim.core.utils import extensions
    from omni.isaac.core.utils.stage import add_reference_to_stage
    import omni.kit.actions.core

    main(app)
