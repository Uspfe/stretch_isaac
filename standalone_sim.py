import argparse
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standalone simulation with optional scene selection. Given 'goal_asset', the shortest path to the goal asset is computed and its distance printed periodically. Additional assets can be spawned and existing assets removed based on substrings."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the simulation in headless mode.",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        help="Path to the USD scene file to load (If not provided, a simple ground plane is loaded. Else the ground plane is loaded but visually hidden to create a flat floor collider.).",
        default=None,
    )
    parser.add_argument(
        "--robot-start",
        nargs=4,
        type=float,
        metavar=("X", "Y", "Z", "THETA"),
        help="Starting position and orientation of the robot (in deg).",
        default=(0.0, 0.0, 0.0, 0.0),
    )
    parser.add_argument(
        "--lighting",
        type=str,
        choices=["camera", "stage"],
        default="stage",
        help="Lighting mode to use (default: stage).",
    )
    parser.add_argument(
        "--asset",
        nargs=5,
        action="append",
        metavar=("PATH", "X", "Y", "Z", "THETA"),
        help="Assets to spawn additionally. Definition: path (path to USD file) x y z theta (in deg) (can be provided multiple times)",
        default=[],
    )
    parser.add_argument(
        "--rasset",
        type=str,
        help="Substring of assets to remove from the scene.",
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "--rasset-exclude",
        type=str,
        help="Substring of assets to exclude from removal even if they match rasset.",
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "--gasset",
        type=str,
        help="Goal assets to broadcast their position.",
    )
    parser.add_argument(
        "--gasset-exclude",
        type=str,
        help="Substring of goal assets to exclude from broadcasting even if they match gasset.",
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "--disable-scene-collider",
        action="store_true",
        help="Disable the scene collider.",
    )
    parser.add_argument(
        "--no-ground-plane",
        action="store_true",
        help="Disable the ground plane.",
    )
    parser.add_argument(
        "--visualize-shortest-path",
        action="store_true",
        help="Visualize the shortest path to the goal asset with matplotlib.",
    )

    args = parser.parse_args()

    if args.scene is not None and not args.scene.exists():
        raise FileNotFoundError(f"Scene file {args.scene} does not exist.")

    return args


args = parse_args()
app = SimulationApp({"headless": args.headless})

# issacsim imports only become availbele after `SimulationApp` is created
import omni  # pyright: ignore[reportMissingImports] # noqa: E402
import omni.kit.actions.core  # pyright: ignore[reportMissingImports] # noqa: E402
from isaacsim.core.api import World  # pyright: ignore[reportMissingImports] # noqa: E402
from isaacsim.core.utils import extensions  # pyright: ignore[reportMissingImports] # noqa: E402
from omni.isaac.core.articulations import Articulation  # pyright: ignore[reportMissingImports] # noqa: E402
from omni.isaac.core.utils.stage import add_reference_to_stage  # pyright: ignore[reportMissingImports] # noqa: E402

from utils.isaac_utils import (  # noqa: E402
    disable_collision,
    dump_prim_position,
    dump_state,
    get_shortest_path_to_prims,
    get_toplevel_prims_substring,
    hide_prim,
    set_prim_pose,
    switch_lighting,
)


def parse_assets(raw_assets):
    assets = []
    for name, x, y, z, theta in raw_assets or []:
        assets.append((name, float(x), float(y), float(z), float(theta)))
    return assets


def main(simulation_app, args: argparse.Namespace):
    args.asset = parse_assets(args.asset)
    args.scene = args.scene.resolve().absolute() if args.scene is not None else None

    extensions.enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()

    root_prim = "/map"

    goal_assets = []
    shortest_goal_distance = None
    if args.scene is not None:
        print(f"Loading scene from {args.scene}")

        # load scene
        omni.usd.get_context().open_stage(str(args.scene))
        loaded_scene_root = "/Root"
        world = World()
        _scene = world.stage.GetPrimAtPath(loaded_scene_root)

        # hide specified assets
        hide_assets = get_toplevel_prims_substring(_scene, args.rasset, True)
        for prim in hide_assets:
            if any(exclude in prim.GetName() for exclude in args.rasset_exclude):
                print(f"Excluding prim {prim.GetPath()} from hiding")
                continue
            print(f"Hiding prim {prim.GetPath()}")
            hide_prim(world.stage, str(prim.GetPath()))

        # find goal assets
        print(f"Searching for goal assets with substring: {args.gasset}")
        goal_assets = get_toplevel_prims_substring(_scene, [args.gasset]) if args.gasset is not None else []
        goal_assets = [
            prim for prim in goal_assets if not any(exclude in prim.GetName() for exclude in args.gasset_exclude)
        ]

        world.reset()

        # disable collision of hidden assets before computing the shortest distances to the goal assets
        for prim in hide_assets:
            disable_collision(prim)

        # compute shortest path to goal assets
        print("Computing shortest path to goals...")
        if len(goal_assets) > 0:
            shortest_goal_distance, goal_positions, shortest_path = get_shortest_path_to_prims(
                goal_assets,
                start_position=np.array(args.robot_start[0:2]),
                root_prim_path=loaded_scene_root,
                visualize=args.visualize_shortest_path,
            )
            if shortest_goal_distance is not None:
                print(f"Shortest distance to goal assets (with offset): {round(shortest_goal_distance, 2)}")

        # disable scene collider if requested
        if args.disable_scene_collider:
            print(f"Disabling collision for scene {_scene.GetPath()}")
            disable_collision(_scene)
    else:
        world = World()
        world.reset()

    # add ground plane
    ground_plane = world.scene.add_default_ground_plane(prim_path=root_prim + "/defaultGroundPlane", z_position=0.05)
    if args.scene is not None:
        hide_prim(world.stage, ground_plane.prim_path)

    # set lighting mode
    print(f"Setting lighting mode to {args.lighting}")
    switch_lighting(mode=args.lighting)

    # load robot
    stretch_asset_path = Path("./robot_usd/stretch3.usd").absolute().as_posix()
    prim_stretch = add_reference_to_stage(usd_path=stretch_asset_path, prim_path=root_prim)

    # load additional assets
    for id, asset in enumerate(args.asset):
        asset_usd_path, x, y, z, theta = asset
        name = Path(asset_usd_path).stem
        print(
            f"Adding asset '{name}' at position ({x}, {y}, {z}) with rotation {theta} and asset path '{asset_usd_path}'"
        )
        prim_asset = add_reference_to_stage(usd_path=str(asset_usd_path), prim_path=f"{root_prim}/{name}_{id}")
        set_prim_pose(prim_asset, (x, y, z), theta)
    world.reset()

    # initialize robot articulation
    stretch = Articulation(prim_path=str(prim_stretch.GetPath()) + "/stretch")
    stretch.set_world_pose(
        np.array([args.robot_start[0], args.robot_start[1], args.robot_start[2]]),
        np.array([np.cos(np.deg2rad(args.robot_start[3]) / 2), 0, 0, np.sin(np.deg2rad(args.robot_start[3]) / 2)]),
    )
    stretch.initialize()

    # main simulation loop
    print_pose_interval: int = 33
    print_goal_interval: int = 110
    try:
        step_count = 0
        while simulation_app.is_running():
            world.step(render=True)  # execute one physics step and one rendering step
            step_count += 1

            # periodically print robot pose
            if step_count % print_pose_interval == 0:
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

            # periodically print goal asset positions
            if step_count % print_goal_interval == 0:
                dump_prim_position(goal_assets, shortest_goal_distance)

            # reset counter periodically to avoid overflow
            if step_count > 1e4:
                step_count = 0
    except KeyboardInterrupt:
        print("Exiting simulation...")

    simulation_app.close()


if __name__ == "__main__":
    main(app, args)
