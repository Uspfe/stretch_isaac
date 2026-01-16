# launch Isaac Sim before any other imports
# default first two lines in any standalone application
import argparse
import csv
import json
from pathlib import Path
import time
from typing import Literal, Optional, Union

from matplotlib import pyplot as plt
import numpy as np
from isaacsim import SimulationApp
from utils.multi_dijkstra import MultiDijkstra

app = SimulationApp({"headless": True})  # we can also run as headless.

import omni.kit.actions.core
from isaacsim.core.api import World
from isaacsim.core.utils import extensions
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import Sdf, Usd, UsdGeom, Gf, PhysxSchema
import omni
import carb


def read_colors(csv_path: Path) -> dict:
    colors = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            colors[row["object"]] = [float(row["r"]), float(row["g"]), float(row["b"])]
    return colors

def compute_occupancy_map(
    root_prim_path: str,
    resolution: float,
    width_m: float,
    height_m: float,
    z_min: float,
    z_max: float,
    return_color: bool = False,
    prim_colors: dict = {},
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes a 2D occupancy map under the given root prim.
    Uses fixed grid size (width, height) and cell resolution.
    Returns a numpy bool array.
    """
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_prim_path)

    if not root.IsValid():
        raise ValueError(f"Invalid prim path: {root_prim_path}")

    # Root AABB (only to determine map center)
    bbox = UsdGeom.Boundable(root).ComputeWorldBound(0, "default").GetRange()
    # bbox_size = bbox.GetMax() - bbox.GetMin()  # Gf.Vec3f
    width = int(np.ceil(width_m / resolution))
    height = int(np.ceil(height_m / resolution))

    center = (bbox.GetMin() + bbox.GetMax()) * 0.5

    xs = center[0] + (np.arange(width) * resolution - width * resolution * 0.5)
    ys = center[1] + (np.arange(height) * resolution - height * resolution * 0.5)

    occ = np.zeros((width, height))
    color = np.zeros((width, height, 3))  # RGB map

    physx = omni.physx.get_physx_scene_query_interface()

    cell_offsets = [
        (0, 0),  # center
        (-0.5, -0.5),
        (-0.5, 0.5),
        (0.5, -0.5),
        (0.5, 0.5),
    ]

    def on_hit(hit):
        return True

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            num_hits = physx.overlap_box(
                carb.Float3(resolution * 0.5, resolution * 0.5, (z_max - z_min) / 2),
                carb.Float3(x, y, (z_min + z_max) / 2),
                carb.Float4(1.0, 0.0, 0.0, 0.0),
                on_hit,
                True,
            )
            if num_hits > 0:
                occ[ix, iy] = 1

            if not return_color:
                continue

            if num_hits == 0:
                color[ix, iy] = (1, 1, 1)  # white
                continue

            # cast multiple rays to get top prim and base the color on that
            prim_hit_counts = {}
            for dx_frac, dy_frac in cell_offsets:
                rx = x + dx_frac * resolution
                ry = y + dy_frac * resolution
                start = carb.Float3(rx, ry, z_max)
                direction = carb.Float3(0, 0, -1)
                hit = physx.raycast_closest(start, direction, z_max - z_min, False)
                if hit["hit"]:
                    prim_path = hit["collision"]
                    prim_hit_counts[prim_path] = prim_hit_counts.get(prim_path, 0) + 1

            if prim_hit_counts:
                # choose prim with most hits
                top_prim = max(prim_hit_counts, key=lambda k: prim_hit_counts[k])
                
                for key in prim_colors:
                    if key in top_prim:  # substring match
                        color[ix, iy] = prim_colors[key]
                        break
                else:
                    # fallback: assign random color
                    prim_colors[top_prim] = np.random.rand(3)
                    color[ix, iy] = prim_colors[top_prim]
            else:
                # fallback black
                color[ix, iy] = (0, 0, 0)

    return occ, xs, ys, color


def get_shortest_path_to_prims(
    prims: list[Usd.Prim],
    start_position: np.ndarray = np.zeros(2),
    resolution: float = 0.1,
    map_width: float = 20.0,
    map_height: float = 20.0,
    z_min: float = 0.2,
    z_max: float = 1.8,
    visualize: bool = False,
) -> Optional[tuple[float, np.ndarray]]:
    if len(prims) == 0:
        return None
    goal_positions = dump_prim_position(prims, print_output=False)
    occupancy_map, x, y, _ = compute_occupancy_map(
        root_prim_path="/Root", resolution=resolution, width_m=map_width, height_m=map_height, z_min=z_min, z_max=z_max
    )
    multi_dijkstra = MultiDijkstra(
        occupancy_map.T, resolution=resolution, origin=np.array([x[0], y[0]]), approx_downsample_resolution=None
    )
    _costs, dists, paths = multi_dijkstra.get_min_distance_to_goals(start_position, goal_positions[:, :2])
    i = np.argmin(dists)
    dist = dists[i]
    path = paths[i]
    print(f"Computed shorted path with distance {dist}")

    if visualize:
        X, Y = np.meshgrid(x, y, indexing="ij")
        plt.pcolormesh(X, Y, 1 - occupancy_map, cmap="gray", shading="auto")
        plt.scatter(start_position[0], start_position[1], c="green", marker="o", label="Start")
        plt.scatter(goal_positions[:, 0], goal_positions[:, 1], c="red", marker="x", label="Goals")
        plt.plot(path[:, 0], path[:, 1], c="blue", linewidth=2, label="Path")
        plt.gca().set_aspect("equal")
        plt.title("Occupancy Map")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    return dist, goal_positions, path


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
    print("<robot>" + json.dumps(data) + "</robot>")


def dump_prim_position(
    prims: list[Usd.Prim], shortest_distance: Optional[float] = None, print_output: bool = True
) -> np.ndarray:
    positions = np.ndarray((len(prims), 3))
    data = {}
    for i, prim in enumerate(prims):
        xformable = UsdGeom.Xformable(prim)
        world_matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pos = world_matrix.ExtractTranslation()
        data[prim.GetName()] = {"x": round(pos[0], 2), "y": round(pos[1], 2), "z": round(pos[2], 2)}
        positions[i, :] = [pos[0], pos[1], pos[2]]
    if shortest_distance is not None:
        data["shortest_distance"] = round(shortest_distance, 2)
    if print_output:
        print("<goals>" + json.dumps(data) + "</goals>")
    return positions


def parse_assets(raw_assets):
    assets = []
    for name, x, y, z, theta in raw_assets or []:
        assets.append((name, float(x), float(y), float(z), float(theta)))
    return assets


def get_toplevel_prims_substring(
    search_root: Usd.Prim, prim_substring: list[str], references_only: bool = False
) -> list[Usd.Prim]:
    matched_prims = []
    for prim in Usd.PrimRange(search_root):
        prim_name = prim.GetName()

        has_payload = prim.HasPayload()
        has_reference = prim.HasAuthoredReferences()
        valid = (not references_only) or (has_reference or has_payload)
        if has_payload or has_reference:
            print(f"{prim_name}: payload={has_payload}, reference={has_reference}")

        if any(
            (valid and substring in prim_name and substring not in str(prim.GetPath().GetParentPath()))
            for substring in prim_substring
        ):
            matched_prims.append(prim)
    return matched_prims


def set_prim_pose(prim, pos, theta):
    xform = UsdGeom.Xformable(prim)

    # translate
    ops = xform.GetOrderedXformOps()
    t_op = next((op for op in ops if op.GetOpName() == "xformOp:translate"), None)
    if t_op is None:
        t_op = xform.AddTranslateOp()
    t_op.Set(Gf.Vec3d(*pos))

    # rotate Z
    r_op = xform.AddRotateZOp()
    r_op.Set(float(theta))


def disable_collision(root_prim: Usd.Prim):
    for prim in Usd.PrimRange(root_prim):
        collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        attr = collision_api.GetPrim().GetAttribute("physics:collisionEnabled")
        if attr:
            attr.Set(False)
        # attr.Set(False)
        # if not attr:
        #     # Create it if missing
        #     attr = collision_api.GetPrim().CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool)


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
        "--generate-map",
        type=Path,
        help="Path to save the generated occupancy map.",
        default=None,
    )
    parser.add_argument(
        "--generate-map-objs",
        type=str,
        help="Objects whose positions to store additionally",
        nargs="*",
        default=[],
    )

    args = parser.parse_args()
    args.asset = parse_assets(args.asset)

    if args.generate_map is not None and args.generate_map.suffix != ".npz":
        raise ValueError("generate-map path must end with .npz")

    extensions.enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()

    root_prim = "/map"

    goal_assets = []
    shortest_goal_distance = None
    if args.scene is not None:
        print(f"Loading scene from {args.scene}")
        omni.usd.get_context().open_stage(str(args.scene))
        world = World()
        _scene = world.stage.GetPrimAtPath("/Root")
        hide_assets = get_toplevel_prims_substring(_scene, args.rasset, True)
        for prim in hide_assets:
            if any(exclude in prim.GetName() for exclude in args.rasset_exclude):
                print(f"Excluding prim {prim.GetPath()} from hiding")
                continue
            print(f"Hiding prim {prim.GetPath()}")
            hide_prim(world.stage, str(prim.GetPath()))

        print(f"Searching for goal assets with substring: {args.gasset}")
        goal_assets = get_toplevel_prims_substring(_scene, [args.gasset]) if args.gasset is not None else []
        goal_assets = [prim for prim in goal_assets if not any(exclude in prim.GetName() for exclude in args.gasset_exclude)]

        world.reset()

        # disable collision of hidden assests before computing the map
        for prim in hide_assets:
            disable_collision(prim)

        print("Computing shortest path to goals...")
        if len(goal_assets) > 0:
            shortest_goal_distance, goal_positions, shortest_path = get_shortest_path_to_prims(goal_assets)
            if shortest_goal_distance is not None:
                shortest_goal_distance -= 1.5  # viewing distance offset
                print(f"Shortest distance to goal assets (with offset): {round(shortest_goal_distance, 2)}")
        else:
            goal_positions = np.zeros((0, 3))
            shortest_path = np.zeros((0, 2))

        if args.generate_map is not None:
            prim_colors = read_colors(Path("./interior_agent_objects.csv"))
            _occupancy_map, x, y, colored_map = compute_occupancy_map(
                root_prim_path="/Root", resolution=0.1, width_m=20, height_m=20, z_min=0.2, z_max=1.8, return_color=True, prim_colors=prim_colors
            )

            if len(args.generate_map_objs) > 0:
                additional_assets = get_toplevel_prims_substring(_scene, args.generate_map_objs)
                print(f"Found additional assets for map generation: {[prim.GetPath() for prim in additional_assets]} for substrings {args.generate_map_objs}")
                additional_positions = dump_prim_position(additional_assets, print_output=False)
            else:
                additional_positions = np.ndarray((0, 3))

            np.savez_compressed(args.generate_map, colored_map=colored_map, x=x, y=y, goal_positions=goal_positions, shortest_path=shortest_path, additional_positions=additional_positions)
            # print state once, so parent process know isaac sim started successfully
            dump_state(
                0.0,
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            )
            time.sleep(1.0)  # ensure file is written
            print(f"Saved map to {args.generate_map}")
            exit(0)

        print(f"Disabling collision for scene {_scene.GetPath()}")
        disable_collision(_scene)

    ground_plane = world.scene.add_ground_plane(prim_path=root_prim + "/defaultGroundPlane", z_position=0.05)
    if args.scene is not None:
        hide_prim(world.stage, ground_plane.prim_path)

    # print(f"Setting lighting mode to {args.lighting}")
    # switch_lighting(mode=args.lighting)

    # load robot
    stretch_asset_path = "/home/benni/repos/stretch_isaac/importable_stretch_no_arm_collider.usd"
    prim_stretch = add_reference_to_stage(usd_path=stretch_asset_path, prim_path=root_prim)

    for id, asset in enumerate(args.asset):
        asset_usd_path, x, y, z, theta = asset
        name = Path(asset_usd_path).stem
        print(
            f"Adding asset '{name}' at position ({x}, {y}, {z}) with rotation {theta} and asset path '{asset_usd_path}'"
        )
        prim_asset = add_reference_to_stage(usd_path=str(asset_usd_path), prim_path=f"{root_prim}/{name}_{id}")
        set_prim_pose(prim_asset, (x, y, z), theta)
    world.reset()

    stretch = Articulation(prim_path=str(prim_stretch.GetPath()) + "/stretch")
    stretch.initialize()

    print_pose_interval: int = 33
    print_goal_interval: int = 110
    try:
        step_count = 0
        while simulation_app.is_running():
            world.step(render=True)  # execute one physics step and one rendering step
            step_count += 1
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
            if step_count % print_goal_interval == 0:
                dump_prim_position(goal_assets, shortest_goal_distance)
            if step_count > 1e4:
                step_count = 0
    except KeyboardInterrupt:
        print("Exiting simulation...")

    simulation_app.close()


if __name__ == "__main__":
    main(app)
