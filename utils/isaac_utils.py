import json
from typing import Literal, Optional, Union

import carb  # pyright: ignore[reportMissingImports] # noqa: E402
import numpy as np
import omni  # pyright: ignore[reportMissingImports] # noqa: E402
import omni.kit.actions.core  # pyright: ignore[reportMissingImports] # noqa: E402
from matplotlib import pyplot as plt
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom  # pyright: ignore[reportMissingImports] # noqa: E402

from utils.multi_dijkstra import MultiDijkstra


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
    root_prim_path: str = "/Root",
) -> Optional[tuple[float, np.ndarray]]:
    if len(prims) == 0:
        return None
    goal_positions = dump_prim_position(prims, print_output=False)
    occupancy_map, x, y, _ = compute_occupancy_map(
        root_prim_path=root_prim_path,
        resolution=resolution,
        width_m=map_width,
        height_m=map_height,
        z_min=z_min,
        z_max=z_max,
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
    """Switch the lighting mode of the stage."""
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
    """Print the provided robot state as a JSON string wrapped in <robot> tags."""
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
    """Print the positions of the provided prims as a JSON string wrapped in <goals> tags. Also includes the shortest distance if provided."""
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


def get_toplevel_prims_substring(
    search_root: Usd.Prim, prim_substring: list[str], references_only: bool = False
) -> list[Usd.Prim]:
    """Get all toplevel prims under the given root prim whose name contains any of the provided substrings.

    Chilren whose parent also matches are not considered toplevel prims and thus not returned.
    If references_only is True, only prims with payloads or references are considered.
    """
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


def set_prim_pose(prim: Usd.Prim, pos: tuple[3, float], theta: float):
    """Set the position and orientation (around Z axis) of a prim."""
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
    """Disable collision for the given prim and all its children."""
    for prim in Usd.PrimRange(root_prim):
        collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        attr = collision_api.GetPrim().GetAttribute("physics:collisionEnabled")
        if attr:
            attr.Set(False)
