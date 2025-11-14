# launch Isaac Sim before any other imports
# default first two lines in any standalone application
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})  # we can also run as headless.

from isaacsim.core.api import World
from isaacsim.core.utils import extensions
from omni.isaac.core.utils.stage import add_reference_to_stage

extensions.enable_extension("isaacsim.ros2.bridge")

simulation_app.update()

world = World()
world.scene.add_default_ground_plane()

# load robot
stretch_asset_path = "/home/benni/repos/benni_stretch_issac/importable_stretch.usd"
stretch = add_reference_to_stage(usd_path=stretch_asset_path, prim_path="/map")


world.reset()

for i in range(500):
    world.step(render=True)  # execute one physics step and one rendering step

simulation_app.close()
