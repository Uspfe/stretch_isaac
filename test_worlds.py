"""The worlds of the Nav2 tests: obstacle courses and house routes, the maps Nav2 gets of them, and the geometric
checks of the robot's footprint against them.

The maps are assumed correct unless a course says otherwise: on a course the map is exact (every box is aligned to its
cells), in the house it is the scene's ground-truth occupancy grid. A course's `map_errors` are cells occupied in the map
but not in the world, like a person mapped next to the robot; its `unmapped` boxes are in the world but not in the map,
like a person who was not there when the map was made. Collisions are judged against the world on a course and against
the map in the house, with the base's outline as Nav2 knows it (its footprint without padding).
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import shapely
import yaml
from shapely import affinity

STRETCH_ISAAC = Path(__file__).resolve().parent
NAVIGATION_PARAMETERS = STRETCH_ISAAC.parent / "stretch_mpc" / "stretch_mpc_ros" / "parameters" / "nav2_params.yaml"
HOUSE_SCENE = STRETCH_ISAAC / "example_stage" / "hm3d_example.usd"

Box = tuple[float, float, float, float]  # min x, min y, max x, max y in the map frame
Pose = tuple[float, float, float]  # x, y, yaw in the map frame
Point = tuple[float, float]

MAP_RESOLUTION = 0.05
GOAL_CLEARANCE = 0.35  # a goal is feasible when the robot can turn on it: the back reaches 0.28 m from base_link
CLEARANCE_TOLERANCE = 1e-6
UNKNOWN, FREE, OCCUPIED = -1, 0, 100


@dataclass
class OccupancyMap:
    """An occupancy grid in the format of stretch_mpc's static_map_publisher; cells[row, column], row along y."""

    cells: np.ndarray
    resolution: float
    origin: Point

    @classmethod
    def from_boxes(cls, boxes: list[Box], resolution: float = MAP_RESOLUTION) -> "OccupancyMap":
        min_x, min_y = min(box[0] for box in boxes), min(box[1] for box in boxes)
        max_x, max_y = max(box[2] for box in boxes), max(box[3] for box in boxes)
        width, height = round((max_x - min_x) / resolution), round((max_y - min_y) / resolution)
        centres_x = min_x + (np.arange(width) + 0.5) * resolution
        centres_y = min_y + (np.arange(height) + 0.5) * resolution
        cells = np.full((height, width), FREE, dtype=np.int8)
        for box_min_x, box_min_y, box_max_x, box_max_y in boxes:
            rows = (centres_y > box_min_y) & (centres_y < box_max_y)
            columns = (centres_x > box_min_x) & (centres_x < box_max_x)
            cells[np.ix_(rows, columns)] = OCCUPIED
        return cls(cells, resolution, (min_x, min_y))

    @classmethod
    def load(cls, path: Path) -> "OccupancyMap":
        data = np.load(path)
        cells = data["data"].reshape(int(data["height"]), int(data["width"]))
        origin = (float(data["origin_position_x"]), float(data["origin_position_y"]))
        return cls(cells, float(data["resolution"]), origin)

    def save(self, path: Path) -> None:
        height, width = self.cells.shape
        np.savez_compressed(
            path,
            data=self.cells.flatten(),
            frame_id=np.asarray("map"),
            stamp_sec=np.int64(0),
            stamp_nanosec=np.uint32(0),
            map_load_time_sec=np.int64(0),
            map_load_time_nanosec=np.uint32(0),
            resolution=np.float32(self.resolution),
            width=np.uint32(width),
            height=np.uint32(height),
            origin_position_x=np.float64(self.origin[0]),
            origin_position_y=np.float64(self.origin[1]),
            origin_position_z=np.float64(0.0),
            origin_orientation_x=np.float64(0.0),
            origin_orientation_y=np.float64(0.0),
            origin_orientation_z=np.float64(0.0),
            origin_orientation_w=np.float64(1.0),
        )

    def _cell_boxes(self, mask: np.ndarray) -> shapely.Geometry:
        rows, columns = np.nonzero(mask)
        min_x = self.origin[0] + columns * self.resolution
        min_y = self.origin[1] + rows * self.resolution
        return shapely.union_all(shapely.box(min_x, min_y, min_x + self.resolution, min_y + self.resolution))

    def obstacles(self) -> shapely.Geometry:
        """The occupied cells, which a collision is judged against."""
        return self._cell_boxes(self.cells == OCCUPIED)

    def clearance(self, point: Point) -> float:
        """Distance from `point` to the nearest cell that is not free (occupied, or unknown like outside a building)."""
        return shapely.Point(point).distance(self._cell_boxes(self.cells != FREE))


@dataclass
class Course:
    """Boxes on an empty floor, with a start pose and a goal pose.

    `reference_path` lists the corners of a path from the start to the goal that the robot fits along, which the
    controller tests give to the controller. A course whose goal the robot cannot reach (inside an obstacle, like on a
    table, or cut off) has no such path but a `closest_approach`: the robot must still head for the goal and end within
    that distance of it (the closest reachable pose plus the controller's 0.25 m goal tolerance). A course that fails
    for a reason accepted for now names it in `known_issue`; `run all` leaves it out. `map_errors` are boxes occupied
    in the map only, `unmapped` boxes are in the world only; the planner and controller tests, which rely on the map,
    skip a course with unmapped boxes. `elevated` boxes are in the map and in the world but too high for the lidar, like
    a table top, so the simulator leaves them out. The planner test also skips a course whose robot starts in occupied
    map cells or inside the planners' margin (`start_occupied`): no planner starts there; the escape node, which needs
    the simulator, first moves the robot out.
    """

    boxes: list[Box]
    start: Pose
    goal: Point
    goal_yaw: float = 0.0
    reference_path: list[Point] = field(default_factory=list)
    closest_approach: Optional[float] = None
    known_issue: Optional[str] = None
    map_errors: list[Box] = field(default_factory=list)
    unmapped: list[Box] = field(default_factory=list)
    elevated: list[Box] = field(default_factory=list)
    start_occupied: bool = False

    def reference_poses(self) -> list[Pose]:
        points = densify([self.start[:2], *self.reference_path, self.goal])
        return [(x, y, heading) for (x, y), heading in zip(points, path_headings(points))]

    def occupancy_map(self) -> OccupancyMap:
        return OccupancyMap.from_boxes(self.boxes + self.map_errors + self.elevated)

    def obstacles(self) -> shapely.Geometry:
        """The world's obstacles, which collisions are judged against."""
        return shapely.union_all([shapely.box(*box) for box in self.boxes + self.unmapped + self.elevated])

    def write_obstacles(self, path: Path) -> None:
        """In the format of standalone_sim.py --obstacles."""
        obstacles = [
            {"center": [(min_x + max_x) / 2, (min_y + max_y) / 2], "size": [max_x - min_x, max_y - min_y]}
            for min_x, min_y, max_x, max_y in self.boxes + self.unmapped
        ]
        path.write_text(json.dumps(obstacles, indent=1))


WALL = 0.1


def arena(length: float, width: float) -> list[Box]:
    """Walls around the floor 0 <= x <= length, -width / 2 <= y <= width / 2."""
    half = width / 2
    return [
        (-WALL, -half - WALL, length + WALL, -half),
        (-WALL, half, length + WALL, half + WALL),
        (-WALL, -half, 0.0, half),
        (length, -half, length + WALL, half),
    ]


def cross_wall(x: float, width: float, gaps: list[tuple[float, float]], depth: float = WALL) -> list[Box]:
    """A wall across the arena from x to x + depth, open at each gap (centre y, gap width)."""
    edges = [-width / 2]
    for centre, gap in sorted(gaps):
        edges += [centre - gap / 2, centre + gap / 2]
    edges.append(width / 2)
    return [(x, low, x + depth, high) for low, high in zip(edges[::2], edges[1::2])]


GAP = 0.6  # the narrowest passage the robot (0.34 m wide) must get through
NARROW_GAP = 0.3  # narrower than the robot
COURSES = {
    "doorway": Course(
        arena(6.0, 4.0) + cross_wall(3.0, 4.0, [(1.0, GAP)]),
        start=(0.6, -0.5, 0.0),
        goal=(5.4, -1.0),
        reference_path=[(2.4, 1.0), (3.7, 1.0)],
    ),
    "corridor": Course(
        arena(7.0, 4.0) + cross_wall(2.5, 4.0, [(0.0, GAP)], depth=2.0),
        start=(0.6, 1.0, 0.0),
        goal=(6.4, -1.0),
        reference_path=[(1.8, 0.0), (5.2, 0.0)],
    ),
    "slalom": Course(
        arena(8.0, 3.0)
        + cross_wall(2.0, 3.0, [(0.6, GAP)])
        + cross_wall(4.0, 3.0, [(-0.6, GAP)])
        + cross_wall(6.0, 3.0, [(0.6, GAP)]),
        start=(0.6, 0.0, 0.0),
        goal=(7.4, 0.0),
        reference_path=[(1.3, 0.6), (2.7, 0.6), (3.3, -0.6), (4.7, -0.6), (5.3, 0.6), (6.7, 0.6)],
    ),
    # Straight ahead is a gap too narrow for the robot; the way round is a feasible one near the side wall.
    "narrow_gap_detour": Course(
        arena(6.0, 4.0) + cross_wall(3.0, 4.0, [(0.0, NARROW_GAP), (-1.4, GAP)]),
        start=(0.6, 0.0, 0.0),
        goal=(5.4, 0.0),
        reference_path=[(2.3, -1.4), (3.8, -1.4)],
    ),
    # The robot starts at the closed end of a dead-end corridor, facing the wall; the goal is behind it. It has 2 cm on
    # either side to turn round, but its planners leave a 5 cm margin, and it is not in occupied cells to escape from.
    "dead_end": Course(
        arena(5.0, 3.0) + cross_wall(3.0, 3.0, [(0.0, GAP)], depth=2.0),
        start=(4.6, 0.0, 0.0),
        goal=(1.0, 0.0),
        known_issue="no planner turns within the margin, the escape node does nothing outside occupied cells, and the "
        "Approach planner returns the start, so the robot stays",
    ),
    # A 0.7 m wide corridor turning by 90 degrees. Where the path turns, base_link's position near the front lets the
    # back swing out towards the outer walls; the house showed planners clipping such corners.
    "corner": Course(
        [
            (-WALL, -0.35 - WALL, 3.0 + WALL, -0.35),
            (3.0, -0.35, 3.0 + WALL, 3.0 + WALL),
            (2.3, 3.0, 3.0, 3.0 + WALL),
            (-WALL, -0.35, 0.0, 0.35),
            (-WALL, 0.35, 2.3, 3.0 + WALL),
        ],
        start=(0.6, 0.0, 0.0),
        goal=(2.65, 2.4),
        goal_yaw=math.pi / 2,  # along the corridor: there is no room to turn on the spot with the planner's margin
        # round the corner on a quarter circle about its inner edge, as the planner does: turning on the spot in the
        # middle of a path is avoided by the planner (rotation_penalty) and beyond what the controller does
        reference_path=[
            (2.3 + 0.35 * math.sin(angle), 0.35 - 0.35 * math.cos(angle))
            for angle in [math.radians(degrees) for degrees in range(0, 91, 10)]
        ],
    ),
    # Map errors. The robot must not halt where the map wrongly says it touches an obstacle, but get away from it.
    # Occupied cells under the robot's back and beside it, like a person mapped there; the robot starts facing away from
    # the goal, so turning sweeps its back through them.
    "phantom_under": Course(
        arena(4.0, 3.0),
        start=(1.5, 0.0, math.pi),
        goal=(3.4, 0.0),
        reference_path=[(1.5, 0.6), (2.6, 0.6)],
        map_errors=[(1.55, -0.1, 1.75, 0.1), (1.55, 0.15, 1.65, 0.25), (1.55, -0.25, 1.65, -0.15)],
        known_issue="freeing the footprint from the wrong cells takes more than the escape's 0.2 m",
    ),
    # Along a 0.8 m corridor the map's wall is 0.25 m too far in, overlapping the side of the robot that starts there.
    "phantom_side": Course(
        arena(5.0, 3.0) + cross_wall(0.5, 3.0, [(0.0, 0.8)], depth=3.0),
        start=(1.2, 0.0, 0.0),
        goal=(4.4, 0.0),
        map_errors=[(1.0, 0.15, 2.5, 0.4)],
        start_occupied=True,
        known_issue="the wrong wall narrows the corridor to 0.55 m: the escape frees the footprint by turning, but no "
        "planner can turn back within the margin there (the robot is boxed in by the map error)",
    ),
    # The map is right, but the robot overshot: it starts 3 cm from the wall of a 0.6 m corridor, inside the margin that
    # the planner keeps, so its start counts as occupied there.
    "margin": Course(
        arena(6.0, 3.0) + cross_wall(1.5, 3.0, [(0.0, GAP)], depth=2.0),
        start=(2.0, 0.1, 0.0),
        goal=(5.0, 0.0),
        reference_path=[(3.0, 0.0)],
        start_occupied=True,
    ),
    # The same overshoot next to a single wall in the open, where there is room to move away from it.
    "margin_open": Course(
        arena(6.0, 3.0) + [(1.0, 0.3, 3.5, 0.4)],
        start=(2.0, 0.1, 0.0),
        goal=(5.0, 0.0),
        reference_path=[(3.0, 0.0)],
        start_occupied=True,
    ),
    # A box on the straight way to the goal that the map does not know; the robot has to see it with the lidar and
    # go round it.
    "unmapped_obstacle": Course(
        arena(6.0, 3.0), start=(0.6, 0.0, 0.0), goal=(5.4, 0.0), unmapped=[(2.7, -0.3, 3.3, 0.3)]
    ),
    # The only way to the goal is a gap narrower than the robot; the closest it gets is facing the wall, base_link 2.5 m
    # from the goal.
    "blocked": Course(
        arena(6.0, 4.0) + cross_wall(3.0, 4.0, [(0.0, NARROW_GAP)]),
        start=(0.6, 0.0, 0.0),
        goal=(5.4, 0.0),
        closest_approach=2.9,
    ),
    # A table top in the way that the lidar cannot see (it passes under it): only /map has it.
    "elevated_table": Course(
        arena(6.0, 3.0),
        start=(0.6, 0.0, 0.0),
        goal=(5.4, 0.0),
        reference_path=[(2.2, 1.05), (3.8, 1.05)],
        elevated=[(2.6, -0.6, 3.4, 0.6)],
    ),
    # The goal is in the middle of a 0.8 m table, as when asked to go to an object on it; the closest the robot gets is
    # facing the table's edge, base_link 0.5 m from the goal.
    "goal_on_table": Course(
        arena(5.0, 3.0) + [(3.0, -0.4, 3.8, 0.4)],
        start=(0.6, 0.0, 0.0),
        goal=(3.4, 0.0),
        closest_approach=0.9,
    ),
    # The goal is in a closed 1 m box; the closest the robot gets is facing a wall of the box, base_link 0.7 m away.
    "goal_enclosed": Course(
        arena(6.0, 3.0)
        + cross_wall(3.4, 1.2, [])
        + cross_wall(4.5, 1.2, [])
        + [(3.5, -0.6, 4.5, -0.5), (3.5, 0.5, 4.5, 0.6)],
        start=(0.6, 0.0, 0.0),
        goal=(4.0, 0.0),
        closest_approach=1.1,
    ),
}


@dataclass
class Route:
    """Waypoints through the house scene, driven to one after another from the start."""

    start: Pose
    waypoints: list[Point]

    def legs(self) -> list[tuple[Point, Point]]:
        stops = [self.start[:2], *self.waypoints]
        return list(zip(stops, stops[1:]))


# A realism check on a scanned home (HM3D), next to the courses: through the 0.8 m corridor and round the corner into
# the north room, where the planner's paths once clipped a wall, and back down to the south.
HOUSE_START = (-1.5, 3.0, 0.0)
ROUTES = {"house": Route(HOUSE_START, [(0.0, 6.4), (-1.0, 0.0)])}


def robot_footprint() -> shapely.Polygon:
    """The outline of the base in base_link, as Nav2 knows it (without its padding)."""
    parameters = yaml.safe_load(NAVIGATION_PARAMETERS.read_text())
    return shapely.Polygon(json.loads(parameters["local_costmap"]["local_costmap"]["ros__parameters"]["footprint"]))


def minimum_clearance(
    poses: list[Pose], obstacles: shapely.Geometry, tolerance: float = 0.0
) -> tuple[float, Optional[Pose]]:
    """The smallest distance between the robot's footprint, shrunk by `tolerance`, and an obstacle over `poses`, and
    where."""
    footprint = robot_footprint().buffer(-tolerance, join_style="mitre")
    closest: tuple[float, Optional[Pose]] = (math.inf, None)
    for x, y, yaw in poses:
        placed = affinity.translate(affinity.rotate(footprint, yaw, origin=(0, 0), use_radians=True), x, y)
        distance = placed.distance(obstacles)
        if distance < closest[0]:
            closest = (distance, (x, y, yaw))
    return closest


def reversing_distance(poses: list[Pose]) -> float:
    """How far a robot passing through `poses` (x, y, yaw) moves against its heading."""
    return sum(
        max(0.0, -((next_x - x) * math.cos(yaw) + (next_y - y) * math.sin(yaw)))
        for (x, y, yaw), (next_x, next_y, _) in zip(poses, poses[1:])
    )


def densify(corners: list[Point], spacing: float = 0.05) -> list[Point]:
    """Points every `spacing` metres along the polyline through `corners`."""
    points = [corners[0]]
    for (start_x, start_y), (end_x, end_y) in zip(corners, corners[1:]):
        steps = max(1, math.ceil(math.dist((start_x, start_y), (end_x, end_y)) / spacing))
        points += [(start_x + (end_x - start_x) * k / steps, start_y + (end_y - start_y) * k / steps)
                   for k in range(1, steps + 1)]  # fmt: skip
    return points


def path_headings(points: list[Point], look_ahead: float = 0.15) -> list[float]:
    """The heading of a differential drive following `points` forwards: towards the point `look_ahead` further on."""
    headings = []
    for index, point in enumerate(points):
        ahead = next((other for other in points[index + 1:] if math.dist(point, other) >= look_ahead), points[-1])
        if ahead == point:
            headings.append(headings[-1] if headings else 0.0)
        else:
            headings.append(math.atan2(ahead[1] - point[1], ahead[0] - point[0]))
    return headings


def swept_poses(
    points: list[Point],
    start_yaw: float,
    goal_yaw: float,
    headings: Optional[list[float]] = None,
    step: float = math.radians(5),
) -> list[Pose]:
    """The poses of a differential drive following `points` with `headings` (by default forwards along the path):
    turning on the spot to the heading at the start and wherever it changes, and to `goal_yaw` at the end."""
    headings = headings if headings is not None else path_headings(points)
    poses: list[Pose] = []
    yaw = start_yaw
    for point, heading in [*zip(points, headings), (points[-1], goal_yaw)]:
        turn = math.remainder(heading - yaw, 2 * math.pi)
        steps = max(1, math.ceil(abs(turn) / step))
        poses += [(point[0], point[1], yaw + turn * k / steps) for k in range(1, steps + 1)]
        yaw = heading
    return poses
