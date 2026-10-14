"""Automated tests of Nav2 (stretch_mpc) on the Stretch, against Isaac Sim or a fixed robot pose.

Every component runs in its own pixi workspace on a private ROS 2 graph: Fast DDS restricted to this machine
(config/fastdds_localhost.xml) on a random domain. /map is assumed correct (test_worlds.py): exact on the obstacle
courses, the scene's ground-truth occupancy grid in the house. Three kinds of scenarios:

- planner_*: Nav2 without a simulator, the robot fixed at the start by tf. The planner's paths must exist where the
  robot fits through, and the robot's footprint swept along them must not touch an obstacle.
- controller_*: the controller follows a fixed reference path through a course in Isaac Sim, without touching an
  obstacle and without getting stuck.
- navigation_*: NavigateToPose end to end in Isaac Sim, through the courses and along waypoint routes in the house.

    pixi run -e test behaviour-test list
    pixi run -e test behaviour-test run 'planner_*'
    pixi run -e test behaviour-test run controller_slalom [--gui] [--keep-running] [--params-file variant.yaml]
    pixi run -e test behaviour-test run all
    pixi run -e test behaviour-test stack [--course slalom] [--no-simulation] [--gui]

Each run writes the logs of every process, the trajectory and result.json to test_runs/<run>/.
"""

import argparse
import csv
import fnmatch
import json
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, TextIO

import shapely

import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import ComputePathToPose, FollowPath, NavigateToPose, Spin
from nav_msgs.msg import OccupancyGrid, Odometry, Path as PathMessage
from rclpy.action import ActionClient
from rclpy.client import Client
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.publisher import Publisher
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_action_status_default
from rclpy.task import Future
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Trigger

from test_worlds import (
    CLEARANCE_TOLERANCE,
    COURSES,
    GOAL_CLEARANCE,
    HOUSE_SCENE,
    HOUSE_START,
    ROUTES,
    STRETCH_ISAAC,
    Course,
    OccupancyMap,
    Point,
    Pose,
    minimum_clearance,
    reversing_distance,
    swept_poses,
)

MONOREPO = STRETCH_ISAAC.parent
STRETCH_MPC = MONOREPO / "stretch_mpc"
STRETCH_CORE = MONOREPO / "online_bringup_active_mapmaintenance" / "stretch_ros2" / "stretch_ros2" / "stretch_core"
LASER_FILTER_PARAMETERS = STRETCH_CORE / "config" / "laser_filter_params.yaml"
FAST_DDS_PROFILE = STRETCH_ISAAC / "config" / "fastdds_localhost.xml"
TEST_RUNS = STRETCH_ISAAC / "test_runs"
HOUSE_MAP_CACHE = TEST_RUNS / "cache" / f"{HOUSE_SCENE.stem}_map.npz"

PASSED_THROUGH_VARIABLES = ("XAUTHORITY", "XDG_RUNTIME_DIR", "WAYLAND_DISPLAY")
RUNNER_ACTIVATION_PREFIXES = ("PIXI_", "CONDA_", "ROS_", "AMENT_", "RMW_", "COLCON_", "FASTRTPS_", "PYTHONPATH")
SYSTEM_PATH = "/usr/local/bin:/usr/bin:/bin"
# Under the load of Isaac Sim starting, rmw_fastrtps sometimes drops the response to a lifecycle transition, because
# it waits only 100 ms to discover the client's response reader; Nav2's lifecycle manager then waits forever.
LOST_LIFECYCLE_RESPONSE = "change_state (timeout): client will not receive response"
# Under the same load, a server may also take longer to configure than the lifecycle manager waits.
ABORTED_BRINGUP = "Aborting bringup"
# Isaac Sim 5.1 occasionally crashes in carb.tasking within its first second, before loading anything.
ISAAC_SIM_STARTUP_CRASH = "Crash detected"
GOAL_TIMEOUT = 240.0  # simulated seconds for one navigation goal or one path to follow
GOAL_DISTANCE = 0.15  # how close the robot must end to a goal it can reach (the goal tolerance is 0.05 m)
PLANNERS = ("Circle", "GridBased", "Approach")  # in the order the behaviour tree tries them
STALL_DISTANCE = 0.1  # the robot counts as stalled while it stays within this distance
# The house map's 0.1 m cells count as occupied when an obstacle covers any part of them, so its walls are up to a cell
# too thick; the robot may reach half a cell into them.
HOUSE_COLLISION_TOLERANCE = 0.05
MAXIMUM_PATH_REVERSING = 0.15  # plans drive forwards, turning on the spot
MAXIMUM_REVERSING = 0.5  # RPP may back off to correct, e.g. in a tight corner


def ros_environment(domain_id: int) -> dict[str, str]:
    return {
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
        "ROS_DOMAIN_ID": str(domain_id),
        "FASTRTPS_DEFAULT_PROFILES_FILE": str(FAST_DDS_PROFILE),
    }


def pose_stamped(x: float, y: float, yaw: float = 0.0) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.pose.position.x, pose.pose.position.y = x, y
    pose.pose.orientation.z, pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return pose


class ManagedProcess:
    """A command run in a pixi environment of another workspace, isolated from this one's environment."""

    name: str
    log_path: Path
    _process: subprocess.Popen
    _log: TextIO

    def __init__(
        self,
        name: str,
        workspace: Path,
        environment: str,
        command: list[str],
        run_directory: Path,
        variables: dict[str, Optional[str]],
    ) -> None:
        """`variables` are set after the workspace's activation; a value of None unsets the variable."""
        self.name = name
        self.log_path = run_directory / f"{name}.log"
        self._log = self.log_path.open("w")
        passed_through = {key: os.environ[key] for key in PASSED_THROUGH_VARIABLES if key in os.environ}
        assignments = [
            f"--unset={key}" if value is None else f"{key}={value}"
            for key, value in {**passed_through, **variables}.items()
        ]
        full_command = [
            "pixi", "run", "--clean-env", "--manifest-path", str(workspace), "-e", environment,
            "/usr/bin/env", *assignments, "PYTHONUNBUFFERED=1",
            "/bin/bash", "-c", f'export PATH="$PATH:{SYSTEM_PATH}"; exec "$@"', "bash", *command,
        ]  # fmt: skip
        self._log.write(f"$ {' '.join(full_command)}\n")
        self._log.flush()
        # without the runner's own pixi and ROS activation, which would leak into (and partly suppress) the workspace's
        environment_of_pixi = {
            key: value for key, value in os.environ.items() if not key.startswith(RUNNER_ACTIVATION_PREFIXES)
        }
        self._process = subprocess.Popen(
            full_command,
            cwd=workspace,
            env=environment_of_pixi,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    def is_running(self) -> bool:
        return self._process.poll() is None

    def log_contains(self, text: str) -> bool:
        return text in self.log_path.read_text(errors="replace")

    def stop(self, timeout: float = 15.0) -> None:
        if self.is_running():
            os.killpg(self._process.pid, signal.SIGINT)
            try:
                self._process.wait(timeout)
            except subprocess.TimeoutExpired:
                os.killpg(self._process.pid, signal.SIGKILL)
                self._process.wait()
        self._log.close()


class Monitor(Node):
    """Observes the stack and drives it through Nav2's actions; its clock is simulation time when simulating."""

    trajectory: list[tuple[float, float, float, float]]
    odometry_velocities: list[tuple[float, float, float, float]]
    navigation_results: dict[bytes, int]
    costmap_received: bool
    _simulated: bool
    _clock_seconds: float
    _lock: threading.Lock
    _velocity_publisher: Publisher
    navigate_client: ActionClient
    follow_client: ActionClient
    plan_client: ActionClient
    circle_plan_client: ActionClient
    spin_client: ActionClient
    _navigation_lifecycle_client: Client

    def __init__(self, simulated: bool) -> None:
        super().__init__("behaviour_test_monitor", parameter_overrides=[Parameter("use_sim_time", value=simulated)])
        self.trajectory = []
        self.odometry_velocities = []
        self.navigation_results = {}
        self.costmap_received = False
        self._simulated = simulated
        self._clock_seconds = 0.0
        self._lock = threading.Lock()
        latched = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE
        )
        self.create_subscription(Clock, "/clock", self._on_clock, 10)
        self.create_subscription(Odometry, "/odom", self._on_odometry, 10)
        self.create_subscription(OccupancyGrid, "/global_costmap/costmap", self._on_costmap, latched)
        self.create_subscription(
            GoalStatusArray,
            "/navigate_to_pose/_action/status",
            self._on_navigation_status,
            qos_profile_action_status_default,
        )
        self._velocity_publisher = self.create_publisher(Twist, "/stretch/cmd_vel", 10)
        self.navigate_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self.follow_client = ActionClient(self, FollowPath, "/follow_path")
        self.plan_client = ActionClient(self, ComputePathToPose, "/compute_path_to_pose")
        self.circle_plan_client = ActionClient(self, ComputePathToPose, "/circle/compute_path_to_pose")
        self.spin_client = ActionClient(self, Spin, "/spin")
        self._navigation_lifecycle_client = self.create_client(Trigger, "/lifecycle_manager_navigation/is_active")

    def _on_clock(self, message: Clock) -> None:
        self._clock_seconds = message.clock.sec + message.clock.nanosec * 1e-9

    def _on_odometry(self, message: Odometry) -> None:
        pose = message.pose.pose
        yaw = 2.0 * math.atan2(pose.orientation.z, pose.orientation.w)
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        twist = message.twist.twist
        with self._lock:
            self.trajectory.append((stamp, pose.position.x, pose.position.y, yaw))
            self.odometry_velocities.append((stamp, twist.linear.x, twist.linear.y, twist.angular.z))

    def _on_costmap(self, _message: OccupancyGrid) -> None:
        self.costmap_received = True

    def _on_navigation_status(self, message: GoalStatusArray) -> None:
        with self._lock:
            for status in message.status_list:
                self.navigation_results[bytes(status.goal_info.goal_id.uuid)] = status.status

    def navigation_is_active(self) -> bool:
        """Whether Nav2's lifecycle manager has activated all servers; their actions reject goals before that."""
        if not self._navigation_lifecycle_client.service_is_ready():
            return False
        future = self._navigation_lifecycle_client.call_async(Trigger.Request())
        deadline = time.time() + 2.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        return future.done() and future.result().success

    def simulation_time(self) -> float:
        """Simulation time when simulating, wall-clock time otherwise."""
        return self._clock_seconds if self._simulated else time.time()

    def pose(self) -> Optional[Pose]:
        with self._lock:
            return self.trajectory[-1][1:] if self.trajectory else None

    def poses(self) -> list[tuple[float, float, float, float]]:
        """Every pose (stamp, x, y, yaw) of /odom so far."""
        with self._lock:
            return list(self.trajectory)

    def recent_odometry(
        self, count: int
    ) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float, float, float]]]:
        """The last `count` poses (stamp, x, y, yaw) and twists (stamp, forward, sideways, turn rate) of /odom."""
        with self._lock:
            return self.trajectory[-count:], self.odometry_velocities[-count:]

    def distance_travelled(self) -> float:
        points = [(x, y) for _, x, y, _ in self.poses()]
        return sum(math.dist(a, b) for a, b in zip(points, points[1:]))

    def longest_stall(self) -> float:
        """The longest time the robot stayed within STALL_DISTANCE of one place while a goal was running."""
        longest, anchor = 0.0, None
        for stamp, x, y, _ in self.poses():
            if anchor is None or math.dist(anchor[1:], (x, y)) > STALL_DISTANCE:
                anchor = (stamp, x, y)
            longest = max(longest, stamp - anchor[0])
        return longest

    def navigation_count(self, status: int) -> int:
        with self._lock:
            return sum(1 for value in self.navigation_results.values() if value == status)

    def command_velocity(self, linear: float, angular: float) -> None:
        """Drives the base directly, bypassing Nav2."""
        command = Twist()
        command.linear.x, command.angular.z = linear, angular
        self._velocity_publisher.publish(command)


def ensure_house_map(domain_id: int) -> None:
    """Computes the house scene's ground-truth occupancy grid with Isaac Sim once, into HOUSE_MAP_CACHE."""
    if HOUSE_MAP_CACHE.exists():
        return
    print("  computing the house's occupancy grid with Isaac Sim (once)")
    HOUSE_MAP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    command = ["python", "standalone_sim.py", "--headless", "--scene", str(HOUSE_SCENE)]
    command += ["--occupancy-map", str(HOUSE_MAP_CACHE)]
    simulation = ManagedProcess(
        "house_map", STRETCH_ISAAC, "default", command, HOUSE_MAP_CACHE.parent, ros_environment(domain_id)
    )
    deadline = time.time() + 900
    while not simulation.log_contains("Saved occupancy grid"):
        if not simulation.is_running() or time.time() > deadline:
            simulation.stop()
            raise RuntimeError(f"computing the house's occupancy grid failed; see {simulation.log_path}")
        time.sleep(1.0)
    simulation.stop()


@dataclass
class Stack:
    """The processes of one test run, and the monitor observing them.

    The world is a course or, without one, the house scene. Without simulation, static tf holds the robot at its start.
    """

    run_directory: Path
    domain_id: int
    headless: bool
    simulate: bool
    course: Optional[Course]
    params_file: Optional[Path]
    metrics: dict[str, object] = field(default_factory=dict)
    processes: list[ManagedProcess] = field(default_factory=list)
    monitor: Optional[Monitor] = None
    _executor: Optional[MultiThreadedExecutor] = None
    _spin_thread: Optional[threading.Thread] = None

    @property
    def start(self) -> Pose:
        return self.course.start if self.course else HOUSE_START

    @property
    def map_file(self) -> Path:
        return self.run_directory / "map.npz"

    def occupancy_map(self) -> OccupancyMap:
        return OccupancyMap.load(self.map_file)

    def obstacles(self) -> shapely.Geometry:
        return self.course.obstacles() if self.course else self.occupancy_map().obstacles()

    @property
    def collision_tolerance(self) -> float:
        return 0.0 if self.course else HOUSE_COLLISION_TOLERANCE

    def _start(self, name: str, workspace: Path, environment: str, command: list[str]) -> ManagedProcess:
        print(f"  starting {name}")
        variables = ros_environment(self.domain_id)
        process = ManagedProcess(name, workspace, environment, command, self.run_directory, variables)
        self.processes.append(process)
        return process

    def _start_navigation(self, name: str) -> ManagedProcess:
        navigation = ["ros2", "launch", "stretch_mpc_ros", "planner.launch.py", f"use_sim_time:={self.simulate}"]
        navigation += ["camera_oscillator:=false"]
        navigation += [f"params_file:={self.params_file.resolve()}"] if self.params_file else []
        return self._start(name, STRETCH_MPC, "dev", navigation)

    def wait_until(self, condition: Callable[[], bool], timeout: float, description: str) -> None:
        """Waits wall-clock time for a startup condition, failing early if a process died."""
        deadline = time.time() + timeout
        while not condition():
            for process in self.processes:
                if not process.is_running():
                    raise RuntimeError(f"{process.name} exited while waiting for {description}; see {process.log_path}")
            if time.time() > deadline:
                raise TimeoutError(f"timed out after {timeout:.0f} s waiting for {description}")
            time.sleep(0.5)

    def check_processes(self) -> None:
        for process in self.processes:
            if not process.is_running():
                raise RuntimeError(f"{process.name} exited; see {process.log_path}")

    def _write_map(self) -> None:
        if self.course:
            self.course.occupancy_map().save(self.map_file)
            return
        ensure_house_map(self.domain_id)
        self.map_file.write_bytes(HOUSE_MAP_CACHE.read_bytes())

    def _start_simulation(self) -> None:
        x, y, yaw = self.start
        command = ["python", "standalone_sim.py", "--robot-start", str(x), str(y), "0", str(math.degrees(yaw))]
        if self.course:
            obstacles_file = self.run_directory / "obstacles.json"
            self.course.write_obstacles(obstacles_file)
            command += ["--obstacles", str(obstacles_file)]
        else:
            command += ["--scene", str(HOUSE_SCENE)]
        command += ["--headless"] if self.headless else []
        simulation = self._start("simulation", STRETCH_ISAAC, "default", command)
        try:
            self.wait_until(lambda: simulation.log_contains("<robot>"), 900, "Isaac Sim to start simulating")
        except RuntimeError:
            if not simulation.log_contains(ISAAC_SIM_STARTUP_CRASH):
                raise
            print("  Isaac Sim crashed while starting; restarting it once")
            self.processes.remove(simulation)
            simulation = self._start("simulation_restarted", STRETCH_ISAAC, "default", command)
            self.wait_until(lambda: simulation.log_contains("<robot>"), 900, "Isaac Sim to start after its restart")
        # /scan -> /scan_filtered as on the real robot (stretch_core's rplidar.launch.py), removing the robot's own mast
        laser_filter = ["ros2", "run", "laser_filters", "scan_to_scan_filter_chain", "--ros-args"]
        laser_filter += ["-r", "__node:=laser_filter", "--params-file", str(LASER_FILTER_PARAMETERS)]
        laser_filter += ["-p", "use_sim_time:=true"]
        self._start("laser_filter", STRETCH_ISAAC, "test", laser_filter)

    def _start_fixed_robot(self) -> None:
        x, y, yaw = self.start
        transform = ["ros2", "run", "tf2_ros", "static_transform_publisher"]
        self._start("map_to_odom", STRETCH_MPC, "dev", [*transform, "--frame-id", "map", "--child-frame-id", "odom"])
        odom_to_base_link = ["--x", str(x), "--y", str(y), "--yaw", str(yaw), "--frame-id", "odom"]
        odom_to_base_link += ["--child-frame-id", "base_link"]
        self._start("odom_to_base_link", STRETCH_MPC, "dev", [*transform, *odom_to_base_link])

    def up(self) -> Monitor:
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self._write_map()
        self._start_simulation() if self.simulate else self._start_fixed_robot()
        map_publisher = ["ros2", "run", "stretch_mpc_ros", "static_map_publisher", "--ros-args"]
        map_publisher += ["-p", f"map_file:={self.map_file}", "-p", f"use_sim_time:={str(self.simulate).lower()}"]
        self._start("map", STRETCH_MPC, "dev", map_publisher)
        navigation = self._start_navigation("navigation")

        os.environ.update(ros_environment(self.domain_id))
        os.environ.pop("ROS_AUTOMATIC_DISCOVERY_RANGE", None)  # makes rmw_fastrtps open sockets on every interface
        rclpy.init()
        self.monitor = Monitor(self.simulate)
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self.monitor)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()
        monitor = self.monitor

        def failed() -> bool:
            return navigation.log_contains(ABORTED_BRINGUP) or navigation.log_contains(LOST_LIFECYCLE_RESPONSE)

        self.wait_until(lambda: failed() or monitor.navigation_is_active(), 300, "Nav2 to activate")
        if failed() and not monitor.navigation_is_active():
            print("  Nav2's lifecycle manager failed to bring it up; restarting Nav2 once")
            navigation.stop()
            self.processes.remove(navigation)
            self._start_navigation("navigation_restarted")
            self.wait_until(lambda: monitor.navigation_is_active(), 300, "Nav2 to activate after its restart")
        self.wait_until(lambda: monitor.costmap_received, 300, "the global costmap")
        return monitor

    def write_attach_script(self) -> Path:
        script = self.run_directory / "attach.sh"
        exports = " ".join(f"{key}={value}" for key, value in ros_environment(self.domain_id).items())
        script.write_text(
            "#!/bin/bash\n"
            "# Runs a command on this run's ROS graph, e.g. ./attach.sh ros2 topic list\n"
            f'exec pixi run --clean-env --manifest-path {STRETCH_ISAAC} -e test /usr/bin/env {exports} "$@"\n'
        )
        script.chmod(0o755)
        return script

    def down(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._spin_thread.join(timeout=10.0)  # destroying the node under a running callback raises at exit
        if self.monitor is not None:
            self.monitor.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for process in reversed(self.processes):
            process.stop()


@dataclass
class Scenario:
    name: str
    description: str
    run: Callable[[Stack, Monitor], list[str]]  # returns the failures; none means passed
    simulate: bool = True
    course: Optional[str] = None  # a key of COURSES; without one, the house scene


def wait_simulated(stack: Stack, monitor: Monitor, seconds: float, until: Callable[[], bool] = lambda: False) -> bool:
    """Lets `seconds` of simulation time pass, or less if `until` becomes true. Returns whether it did."""
    start = monitor.simulation_time()
    wall_deadline = time.time() + 10 * seconds + 120  # the simulation may run slower than real time, but not stall
    while monitor.simulation_time() - start < seconds:
        if until():
            return True
        stack.check_processes()
        if time.time() > wall_deadline:
            raise TimeoutError("simulation time stopped advancing")
        time.sleep(0.2)
    return until()


def run_action(
    stack: Stack, monitor: Monitor, client: ActionClient, goal: object, timeout: float
) -> tuple[Optional[int], Optional[object]]:
    """Sends an action goal; returns its final status and result, or (None, None) if it did not finish in time."""
    goal_future: Future = client.send_goal_async(goal)
    if not wait_simulated(stack, monitor, 10.0, goal_future.done) or not goal_future.result().accepted:
        return GoalStatus.STATUS_ABORTED, None
    result_future: Future = goal_future.result().get_result_async()
    if not wait_simulated(stack, monitor, timeout, result_future.done):
        goal_future.result().cancel_goal_async()
        return None, None
    return result_future.result().status, result_future.result().result


def check_reversing(label: str, poses: list[Pose], metrics: list[float], maximum: float) -> list[str]:
    reversing = reversing_distance(poses)
    metrics.append(reversing)
    return [f"{label} reverses {reversing:.2f} m"] if reversing > maximum else []


def check_trajectory(stack: Stack, monitor: Monitor) -> list[str]:
    """The robot's footprint must never have touched an obstacle, and it must not have reversed; records clearance
    and stalls as metrics."""
    poses = [pose[1:] for pose in monitor.poses()]
    clearance, closest = minimum_clearance(poses, stack.obstacles(), stack.collision_tolerance)
    stack.metrics["minimum_clearance"] = clearance
    stack.metrics["closest_pose"] = closest
    stack.metrics["longest_stall"] = monitor.longest_stall()
    failures = [f"the footprint touched an obstacle at (x, y, yaw) {closest}"] if clearance <= 0.0 else []
    return failures + check_reversing("the robot", poses, stack.metrics.setdefault("reversing", []), MAXIMUM_REVERSING)


def navigate_to(
    stack: Stack, monitor: Monitor, position: Point, yaw: float = 0.0, within: float = GOAL_DISTANCE
) -> list[str]:
    """NavigateToPose must succeed within GOAL_TIMEOUT and end `within` that distance of `position`."""
    goal = NavigateToPose.Goal()
    goal.pose = pose_stamped(*position, yaw)
    start = monitor.simulation_time()
    status, _ = run_action(stack, monitor, monitor.navigate_client, goal, GOAL_TIMEOUT)
    stack.metrics.setdefault("navigation_durations", []).append(monitor.simulation_time() - start)
    failures = []
    if status != GoalStatus.STATUS_SUCCEEDED:
        failures.append(f"NavigateToPose to {position} ended with status {status} instead of SUCCEEDED (4)")
    pose = monitor.pose()
    if pose is not None:
        stack.metrics.setdefault("goal_distances", []).append(math.dist(pose[:2], position))
    if pose is None or math.dist(pose[:2], position) > within:
        failures.append(f"robot ended at {pose}, more than {within} m from the goal {position}")
    return failures


def check_goals_feasible(occupancy_map: OccupancyMap, goals: list[Point]) -> None:
    """A test is only meaningful if the robot can stand and turn at every goal; this checks the test, not Nav2."""
    for goal in goals:
        if occupancy_map.clearance(goal) < GOAL_CLEARANCE - CLEARANCE_TOLERANCE:
            raise ValueError(f"goal {goal} is closer than {GOAL_CLEARANCE} m to an obstacle on the map")


def check_path(
    stack: Stack,
    start: Pose,
    goal: Point,
    goal_yaw: float = 0.0,
    closest_approach: Optional[float] = None,
) -> tuple[list[str], Pose]:
    """ComputePathToPose from `start` to `goal` with the planners in the behaviour tree's order, each where the one
    before fails. A path must exist, the
    footprint swept along it must not touch an obstacle, and it must not reverse. If
    the goal cannot be reached, the path must end within `closest_approach` of it. Returns the failures and the pose
    the robot would end in."""
    monitor = stack.monitor
    goal_message = ComputePathToPose.Goal()
    goal_message.goal, goal_message.start = pose_stamped(*goal, goal_yaw), pose_stamped(*start)
    goal_message.use_start = True
    for planner in PLANNERS:
        goal_message.planner_id = planner
        client = monitor.circle_plan_client if planner == "Circle" else monitor.plan_client
        status, result = run_action(stack, monitor, client, goal_message, 30.0)
        if status == GoalStatus.STATUS_SUCCEEDED:
            stack.metrics.setdefault("planners", []).append(planner)
            break
    label = f"path from {start[:2]} to {goal}"
    poses = result.path.poses if result else []
    points = [(pose.pose.position.x, pose.pose.position.y) for pose in poses]
    yaws = [2.0 * math.atan2(pose.pose.orientation.z, pose.pose.orientation.w) for pose in poses]
    # Planners that model the robot's heading (Smac) orient every pose; others (NavFn) leave them all at zero.
    headings = yaws if any(abs(yaw) > 1e-6 for yaw in yaws[:-1]) else None
    arrival = (*goal, goal_yaw)
    if status != GoalStatus.STATUS_SUCCEEDED or not points:
        return [f"no {label} (status {status})"], arrival
    paths_file = stack.run_directory / "planned_paths.json"
    paths = json.loads(paths_file.read_text()) if paths_file.exists() else []
    paths_file.write_text(json.dumps([*paths, {"start": start, "goal": goal, "path": points, "yaws": yaws}]))
    clearance, closest = minimum_clearance(swept_poses(points, start[2], goal_yaw, headings), stack.obstacles())
    stack.metrics.setdefault("path_clearances", []).append(clearance)
    planning_time = result.planning_time.sec + result.planning_time.nanosec * 1e-9
    stack.metrics.setdefault("planning_times", []).append(planning_time)
    stack.metrics.setdefault("path_lengths", []).append(sum(math.dist(a, b) for a, b in zip(points, points[1:])))
    failures = [f"the footprint swept along the {label} touches an obstacle at {closest}"] if clearance <= 0.0 else []
    end_distance = math.dist(points[-1], goal)
    stack.metrics.setdefault("path_end_distances", []).append(end_distance)
    if end_distance > (closest_approach or GOAL_DISTANCE):
        failures.append(f"the {label} ends {end_distance:.2f} m from the goal")
    if headings is not None:
        poses = [(x, y, yaw) for (x, y), yaw in zip(points, headings)]
        path_reversing = stack.metrics.setdefault("path_reversing", [])
        failures += check_reversing(f"the {label}", poses, path_reversing, MAXIMUM_PATH_REVERSING)
    return failures, arrival


def plan_course(stack: Stack, monitor: Monitor) -> list[str]:
    course = stack.course
    if course.closest_approach is None:
        check_goals_feasible(stack.occupancy_map(), [course.goal])
    return check_path(stack, course.start, course.goal, course.goal_yaw, course.closest_approach)[0]


def plan_routes(stack: Stack, monitor: Monitor) -> list[str]:
    failures = []
    for route in ROUTES.values():
        check_goals_feasible(stack.occupancy_map(), route.waypoints)
        start = route.start
        for _, goal in route.legs():
            leg_failures, start = check_path(stack, start, goal)
            failures += leg_failures
    return failures


def follow_reference_path(stack: Stack, monitor: Monitor) -> list[str]:
    """The controller follows the course's reference path to its goal without touching an obstacle."""
    course = stack.course
    reference_poses = course.reference_poses()
    goal = FollowPath.Goal()
    goal.controller_id = "FollowPath"
    goal.path = PathMessage()
    goal.path.header.frame_id = "map"
    goal.path.poses = [pose_stamped(*pose) for pose in reference_poses]
    start = monitor.simulation_time()
    status, _ = run_action(stack, monitor, monitor.follow_client, goal, GOAL_TIMEOUT)
    stack.metrics["follow_duration"] = monitor.simulation_time() - start
    reference = shapely.LineString([pose[:2] for pose in reference_poses])
    stack.metrics["maximum_path_deviation"] = max(
        (reference.distance(shapely.Point(x, y)) for _, x, y, _ in monitor.poses()), default=None
    )
    failures = [] if status == GoalStatus.STATUS_SUCCEEDED else [f"FollowPath ended with status {status}"]
    pose = monitor.pose()
    if pose is None or math.dist(pose[:2], course.goal) > 0.35:
        failures.append(f"robot ended at {pose}, more than 0.35 m from the end of the path {course.goal}")
    return failures + check_trajectory(stack, monitor)


def navigate_course(stack: Stack, monitor: Monitor) -> list[str]:
    course = stack.course
    within = course.closest_approach or GOAL_DISTANCE
    return navigate_to(stack, monitor, course.goal, course.goal_yaw, within) + check_trajectory(stack, monitor)


def drive_route(name: str) -> Callable[[Stack, Monitor], list[str]]:
    """NavigateToPose to every waypoint of the route in turn and back to the start, each within GOAL_TIMEOUT."""
    route = ROUTES[name]

    def run(stack: Stack, monitor: Monitor) -> list[str]:
        check_goals_feasible(stack.occupancy_map(), route.waypoints)
        failures = []
        for _, goal in route.legs():
            failures += navigate_to(stack, monitor, goal)
        return failures + check_trajectory(stack, monitor)

    return run


def spin_in_place(stack: Stack, monitor: Monitor) -> list[str]:
    start = monitor.pose()
    goal = Spin.Goal()
    goal.target_yaw = math.pi / 2
    status, _ = run_action(stack, monitor, monitor.spin_client, goal, timeout=60.0)
    end = monitor.pose()
    failures = []
    if status != GoalStatus.STATUS_SUCCEEDED:
        failures.append(f"Spin ended with status {status} instead of SUCCEEDED (4)")
    if start is None or end is None:
        return failures + ["no odometry received"]
    turned = math.remainder(end[2] - start[2], 2 * math.pi)
    if abs(turned - math.pi / 2) > 0.3:
        failures.append(f"turned {math.degrees(turned):.0f} deg instead of 90 deg")
    if math.dist(start[:2], end[:2]) > 0.1:
        failures.append(f"drifted {math.dist(start[:2], end[:2]):.2f} m while spinning")
    return failures


def odometry_consistency(stack: Stack, monitor: Monitor) -> list[str]:
    """/odom's twist must be the body-frame velocity that its own poses imply (Nav2's controllers rely on it)."""
    failures = []
    for linear, angular in ((0.15, 0.0), (0.0, 0.4), (0.1, 0.3)):
        start = monitor.simulation_time()
        while monitor.simulation_time() - start < 3.0:
            monitor.command_velocity(linear, angular)
            wait_simulated(stack, monitor, 0.05)
        poses, velocities = monitor.recent_odometry(60)
        implied = []
        for (time_a, x_a, y_a, yaw_a), (time_b, x_b, y_b, yaw_b) in zip(poses, poses[1:]):
            duration = time_b - time_a
            if duration <= 0:
                continue
            velocity_x, velocity_y = (x_b - x_a) / duration, (y_b - y_a) / duration
            implied.append(
                (
                    velocity_x * math.cos(yaw_a) + velocity_y * math.sin(yaw_a),
                    -velocity_x * math.sin(yaw_a) + velocity_y * math.cos(yaw_a),
                    math.remainder(yaw_b - yaw_a, 2 * math.pi) / duration,
                )
            )
        forward, sideways, turn_rate = (sum(values) / len(implied) for values in zip(*implied))
        reported_forward, reported_sideways, reported_turn_rate = (
            sum(values) / len(velocities) for values in list(zip(*velocities))[1:]
        )
        label = f"command ({linear}, {angular})"
        if abs(reported_forward - forward) > 0.03 or abs(reported_sideways - sideways) > 0.03:
            failures.append(
                f"{label}: odom twist linear ({reported_forward:.3f}, {reported_sideways:.3f}), "
                f"poses imply ({forward:.3f}, {sideways:.3f})"
            )
        if abs(reported_turn_rate - turn_rate) > 0.05:
            failures.append(f"{label}: odom twist angular {reported_turn_rate:.3f}, poses imply {turn_rate:.3f}")
    monitor.command_velocity(0.0, 0.0)
    return failures


COURSE_DESCRIPTIONS = {
    "doorway": "through a 0.6 m doorway offset from the start and the goal",
    "corridor": "through a 0.6 m wide, 2 m long corridor",
    "slalom": "through three walls with alternating 0.6 m gaps, 2 m apart",
    "narrow_gap_detour": "past a 0.3 m gap straight ahead, through a 0.6 m gap to the side",
    "dead_end": "out of a 0.6 m wide dead end it starts in, facing the closed end",
    "corner": "round the 90 degree turn of a 0.7 m wide corridor",
    "phantom_under": "away from map cells wrongly occupied under and beside the robot, turning through them",
    "phantom_side": "along a corridor whose wall the map has 0.25 m too far in, overlapping the robot",
    "margin": "on from 3 cm beside the wall of a 0.6 m corridor, inside the planner's margin",
    "margin_open": "on from 3 cm beside a single wall in the open, inside the planner's margin",
    "unmapped_obstacle": "round a box on the way that the map does not know",
    "blocked": "towards a goal behind a wall whose only gap (0.3 m) is narrower than the robot",
    "goal_on_table": "towards a goal in the middle of a table",
    "elevated_table": "round a table top in the way that only the map has, not the lidar",
    "goal_enclosed": "towards a goal inside a closed box",
}
SCENARIOS = {
    scenario.name: scenario
    for scenario in [
        *(
            Scenario(
                f"planner_{name}",
                f"Planner, {COURSE_DESCRIPTIONS[name]}: "
                + ("a path the footprint fits along." if course.closest_approach is None else "as close as it gets."),
                plan_course,
                simulate=False,
                course=name,
            )
            for name, course in COURSES.items()
            if not course.unmapped and not course.start_occupied
        ),
        Scenario(
            "planner_house",
            "Planner, every leg of the house routes: paths the footprint fits along.",
            plan_routes,
            simulate=False,
        ),
        *(
            Scenario(
                f"controller_{name}",
                f"Controller follows a reference path {COURSE_DESCRIPTIONS[name]}.",
                follow_reference_path,
                course=name,
            )
            for name, course in COURSES.items()
            if course.closest_approach is None and not course.unmapped
        ),
        *(
            Scenario(
                f"navigation_{name}", f"NavigateToPose {COURSE_DESCRIPTIONS[name]}.", navigate_course, course=name
            )
            for name, course in COURSES.items()
        ),
        *(
            Scenario(
                f"navigation_{name}", f"NavigateToPose along the {name} waypoint route.", drive_route(name)
            )
            for name in ROUTES
        ),
        Scenario("navigation_spin", "Nav2's Spin behavior turns the robot by 90 deg on the spot.", spin_in_place),
        Scenario(
            "simulation_odometry",
            "/odom's twist is the body-frame velocity its poses imply, for straight, turning and curved driving.",
            odometry_consistency,
        ),
    ]
}


def write_results(
    run_directory: Path, scenario: Scenario, failures: list[str], monitor: Optional[Monitor], metrics: dict[str, object]
) -> None:
    if monitor is not None and monitor.trajectory:
        with (run_directory / "trajectory.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["time", "x", "y", "yaw"])
            writer.writerows(monitor.trajectory)
    result = {
        "scenario": scenario.name,
        "passed": not failures,
        "failures": failures,
        "distance_travelled": monitor.distance_travelled() if monitor else None,
        "final_pose": monitor.pose() if monitor else None,
        **metrics,
    }
    (run_directory / "result.json").write_text(json.dumps(result, indent=2))


def new_stack(arguments: argparse.Namespace, name: str, simulate: bool, course: Optional[str]) -> Stack:
    domain_id = arguments.domain_id if arguments.domain_id is not None else random.randint(100, 200)
    run_directory = TEST_RUNS / f"{datetime.now():%Y%m%d_%H%M%S}_{name}"
    return Stack(
        run_directory,
        domain_id,
        not arguments.gui,
        simulate,
        COURSES[course] if course else None,
        arguments.params_file,
    )


def wait_for_interrupt(stack: Stack) -> None:
    print(f"  stack is running; attach with {stack.write_attach_script()} <command>, Ctrl-C to stop")
    try:
        while True:
            stack.check_processes()
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


def run_scenario(scenario: Scenario, arguments: argparse.Namespace) -> bool:
    stack = new_stack(arguments, scenario.name, scenario.simulate, scenario.course)
    print(f"[{scenario.name}] {scenario.description}")
    print(f"  run directory {stack.run_directory}, ROS domain {stack.domain_id}")
    monitor = None
    try:
        monitor = stack.up()
        failures = scenario.run(stack, monitor)
    except (RuntimeError, TimeoutError) as error:
        failures = [f"stack failure: {error}"]
    except ValueError as error:
        failures = [f"invalid test: {error}"]
    except KeyboardInterrupt:
        failures = ["interrupted"]
    try:
        write_results(stack.run_directory, scenario, failures, monitor, stack.metrics)
        print(f"  {'PASSED' if not failures else 'FAILED'}" + "".join(f"\n    - {failure}" for failure in failures))
        if arguments.keep_running and monitor is not None:
            wait_for_interrupt(stack)
    finally:
        stack.down()
    return not failures


def run_in_parallel(names: list[str], arguments: argparse.Namespace) -> dict[str, bool]:
    """Runs each scenario in its own process and ROS domain, `arguments.jobs` at a time."""
    domain_ids = random.sample(range(100, 200), len(names))
    if any(SCENARIOS[name].course is None for name in names):
        ensure_house_map(domain_ids[0])
    batch_directory = TEST_RUNS / f"{datetime.now():%Y%m%d_%H%M%S}_batch"
    batch_directory.mkdir(parents=True)
    options = ["--gui"] if arguments.gui else []
    options += ["--params-file", str(arguments.params_file.resolve())] if arguments.params_file else []
    print(f"running {len(names)} scenarios, {arguments.jobs} at a time; their output goes to {batch_directory}")

    def run(index: int, name: str) -> tuple[str, bool]:
        time.sleep(10.0 * index if index < arguments.jobs else 0.0)  # stagger the first Isaac Sim startups
        command = [sys.executable, "-u", __file__, "run", name, "--jobs", "1", "--domain-id", str(domain_ids[index])]
        with (batch_directory / f"{name}.log").open("w") as log:
            passed = subprocess.run([*command, *options], stdout=log, stderr=subprocess.STDOUT).returncode == 0
        print(f"{'PASSED' if passed else 'FAILED'}  {name}", flush=True)
        return name, passed

    with ThreadPoolExecutor(arguments.jobs) as pool:
        return dict(pool.map(run, range(len(names)), names))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List the scenarios.")
    run_parser = commands.add_parser("run", help="Run scenarios, each on a fresh stack.")
    run_parser.add_argument("scenarios", nargs="+", help="Scenario names or patterns ('planner_*'), or 'all'.")
    run_parser.add_argument("--keep-running", action="store_true", help="Keep each stack up after its scenario.")
    run_parser.add_argument("--jobs", type=int, default=3, help="Scenarios run at the same time (default 3).")
    stack_parser = commands.add_parser("stack", help="Only bring the stack up, for manual debugging.")
    stack_parser.add_argument("--course", choices=COURSES, help="An obstacle course instead of the house.")
    stack_parser.add_argument(
        "--no-simulation", action="store_true", help="Hold the robot at its start by tf instead of simulating it."
    )
    for subparser in (run_parser, stack_parser):
        subparser.add_argument(
            "--gui", action="store_true", help="Show the Isaac Sim window instead of running headless."
        )
        subparser.add_argument("--domain-id", type=int, default=None, help="ROS domain (default: random 100-200).")
        subparser.add_argument(
            "--params-file", type=Path, default=None, help="Nav2 parameters instead of stretch_mpc's nav2_params.yaml."
        )
    arguments = parser.parse_args()

    if arguments.command == "list":
        for scenario in SCENARIOS.values():
            print(f"{scenario.name:32} {scenario.description}")
        return 0

    if arguments.command == "stack":
        stack = new_stack(arguments, "stack", not arguments.no_simulation, arguments.course)
        print(f"run directory {stack.run_directory}, ROS domain {stack.domain_id}")
        try:
            stack.up()
            wait_for_interrupt(stack)
        except (RuntimeError, TimeoutError) as error:
            print(f"stack failure: {error}")
        finally:
            stack.down()
        return 0

    patterns = ["*" if pattern == "all" else pattern for pattern in arguments.scenarios]
    unknown = [pattern for pattern in patterns if not fnmatch.filter(SCENARIOS, pattern)]
    if unknown:
        parser.error(f"no scenarios match {unknown}; see 'list'")
    names = [name for name in SCENARIOS if any(fnmatch.fnmatch(name, pattern) for pattern in patterns)]
    if "all" in arguments.scenarios:
        course_names = {name: SCENARIOS[name].course for name in names}
        known_issues = [name for name, course in course_names.items() if course and COURSES[course].known_issue]
        names = [name for name in names if name not in known_issues]
        print(f"leaving out scenarios with a known issue: {', '.join(known_issues)}")
    if arguments.jobs > 1 and len(names) > 1 and not arguments.keep_running:
        results = run_in_parallel(names, arguments)  # reports each scenario as it finishes
    else:
        results = {name: run_scenario(SCENARIOS[name], arguments) for name in names}
        print("\n".join(f"{'PASSED' if passed else 'FAILED'}  {name}" for name, passed in results.items()))
    print(f"{sum(results.values())} of {len(results)} scenarios passed")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
