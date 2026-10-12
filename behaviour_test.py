"""Automated behaviour tests of the robot stack in Isaac Sim.

Every component runs in its own pixi workspace on a private ROS 2 graph: Fast DDS restricted to this machine
(config/fastdds_localhost.xml) on a random domain, all on simulation time. Without --perceive, /map is the
scene's ground-truth occupancy grid; with it, perceive_semantix builds /map from the simulated camera.

    pixi run -e test behaviour-test list
    pixi run -e test behaviour-test run navigate_across_house [--gui] [--keep-running]
    pixi run -e test behaviour-test run all
    pixi run -e test behaviour-test stack --coordinator [--perceive] [--gui]

Each run writes the logs of every process, the ground-truth trajectory and result.json to test_runs/<run>/.
"""

import argparse
import csv
import json
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, TextIO

import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose, Spin
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.action import ActionClient
from rclpy.client import Client
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.publisher import Publisher
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_action_status_default
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String
from std_srvs.srv import Trigger

STRETCH_ISAAC = Path(__file__).resolve().parent
MONOREPO = STRETCH_ISAAC.parent
STRETCH_MPC = MONOREPO / "stretch_mpc"
OFFLINE_BRINGUP = MONOREPO / "offline_bringup_active_mapmaintenance"
PERCEIVE_SEMANTIX = MONOREPO / "perceive_semantix"
FAST_DDS_PROFILE = STRETCH_ISAAC / "config" / "fastdds_localhost.xml"
TEST_RUNS = STRETCH_ISAAC / "test_runs"

DEFAULT_SCENE = STRETCH_ISAAC / "example_stage" / "hm3d_example.usd"
DEFAULT_START = (-7.26, 1.61, 0.0)  # x, y, yaw in the map frame: the open room on the west side of the HM3D scene

PASSED_THROUGH_VARIABLES = ("XAUTHORITY", "XDG_RUNTIME_DIR", "WAYLAND_DISPLAY")
SYSTEM_PATH = "/usr/local/bin:/usr/bin:/bin"


def ros_environment(domain_id: int) -> dict[str, str]:
    return {
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
        "ROS_DOMAIN_ID": str(domain_id),
        "FASTRTPS_DEFAULT_PROFILES_FILE": str(FAST_DDS_PROFILE),
    }


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
        keep_stdin_open: bool = False,
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
        self._process = subprocess.Popen(
            full_command,
            cwd=workspace,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if keep_stdin_open else subprocess.DEVNULL,
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
    """Observes the stack on simulation time and drives it like a user would."""

    trajectory: list[tuple[float, float, float, float]]
    odometry_velocities: list[tuple[float, float, float, float]]
    navigation_results: dict[bytes, int]
    thoughts: list[str]
    costmap_received: bool
    _clock_seconds: float
    _lock: threading.Lock
    _user_query_publisher: Publisher
    _velocity_publisher: Publisher
    navigate_client: ActionClient
    spin_client: ActionClient
    _navigation_lifecycle_client: Client

    def __init__(self) -> None:
        super().__init__("behaviour_test_monitor", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.trajectory = []
        self.odometry_velocities = []
        self.navigation_results = {}
        self.thoughts = []
        self.costmap_received = False
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
        self.create_subscription(String, "main_coordinator/thought_process", self._on_thought, 10)
        self._user_query_publisher = self.create_publisher(String, "main_coordinator/user_query", 10)
        self._velocity_publisher = self.create_publisher(Twist, "/stretch/cmd_vel", 10)
        self.navigate_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")
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

    def _on_thought(self, message: String) -> None:
        with self._lock:
            self.thoughts.append(message.data)

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
        return self._clock_seconds

    def pose(self) -> Optional[tuple[float, float, float]]:
        with self._lock:
            return self.trajectory[-1][1:] if self.trajectory else None

    def recent_odometry(
        self, count: int
    ) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float, float, float]]]:
        """The last `count` poses (stamp, x, y, yaw) and twists (stamp, forward, sideways, turn rate) of /odom."""
        with self._lock:
            return self.trajectory[-count:], self.odometry_velocities[-count:]

    def distance_travelled(self) -> float:
        with self._lock:
            points = [(x, y) for _, x, y, _ in self.trajectory]
        return sum(math.dist(a, b) for a, b in zip(points, points[1:]))

    def navigation_count(self, status: int) -> int:
        with self._lock:
            return sum(1 for value in self.navigation_results.values() if value == status)

    def command_velocity(self, linear: float, angular: float) -> None:
        """Drives the base directly, bypassing Nav2."""
        command = Twist()
        command.linear.x, command.angular.z = linear, angular
        self._velocity_publisher.publish(command)

    def send_user_query(self, query: str, timeout: float = 30.0) -> None:
        """Publishes once main_coordinator subscribes; the topic keeps no history for a late subscriber."""
        deadline = time.time() + timeout
        while self._user_query_publisher.get_subscription_count() == 0:
            if time.time() > deadline:
                raise TimeoutError("main_coordinator does not subscribe to its user query topic")
            time.sleep(0.1)
        self._user_query_publisher.publish(String(data=query))


@dataclass
class Stack:
    """The processes of one test run and the monitor observing them."""

    run_directory: Path
    domain_id: int
    headless: bool
    coordinator: bool
    perceive: bool
    scene: Path
    start: tuple[float, float, float]
    processes: list[ManagedProcess] = field(default_factory=list)
    monitor: Optional[Monitor] = None
    _executor: Optional[MultiThreadedExecutor] = None

    def _start(
        self,
        name: str,
        workspace: Path,
        environment: str,
        command: list[str],
        keep_stdin_open: bool = False,
        variables: Optional[dict[str, Optional[str]]] = None,
    ) -> ManagedProcess:
        print(f"  starting {name}")
        process = ManagedProcess(
            name,
            workspace,
            environment,
            command,
            self.run_directory,
            {**ros_environment(self.domain_id), **(variables or {})},
            keep_stdin_open,
        )
        self.processes.append(process)
        return process

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

    def up(self) -> Monitor:
        self.run_directory.mkdir(parents=True, exist_ok=True)
        map_file = self.run_directory / "map.npz"
        x, y, yaw = self.start
        simulation_command = ["python", "standalone_sim.py", "--scene", str(self.scene)]
        simulation_command += ["--robot-start", str(x), str(y), "0", str(math.degrees(yaw))]
        simulation_command += [] if self.perceive else ["--occupancy-map", str(map_file)]
        simulation_command += ["--headless"] if self.headless else []
        simulation = self._start("simulation", STRETCH_ISAAC, "default", simulation_command)
        self.wait_until(lambda: simulation.log_contains("<robot>"), 900, "Isaac Sim to start simulating")

        if self.perceive:
            perception = ["ros2", "run", "perceive_semantix_ros2", "perceive_semantix_node", "--ros-args"]
            perception += ["-p", "use_sim_time:=true", "-p", "rerun_mode:=disabled"]
            perception += ["-p", "camera_depth_scale_to_m:=1.0"]  # Isaac Sim publishes depth in metres (32FC1)
            # Without a key the OpenAI-based priors stay off, so a test run sends nothing to OpenAI.
            self._start("perceive_semantix", PERCEIVE_SEMANTIX, "zenoh", perception, variables={"OPENAI_API_KEY": None})
        else:
            map_publisher = ["ros2", "run", "stretch_mpc_ros", "static_map_publisher", "--ros-args"]
            self._start(
                "map", STRETCH_MPC, "dev", [*map_publisher, "-p", f"map_file:={map_file}", "-p", "use_sim_time:=true"]
            )
        navigation = ["ros2", "launch", "stretch_mpc_ros", "planner.launch.py", "use_sim_time:=true"]
        self._start("navigation", STRETCH_MPC, "dev", [*navigation, "camera_oscillator:=false"])

        os.environ.update(ros_environment(self.domain_id))
        os.environ.pop(
            "ROS_AUTOMATIC_DISCOVERY_RANGE", None
        )  # makes rmw_fastrtps open sockets on every interface again
        rclpy.init()
        self.monitor = Monitor()
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self.monitor)
        threading.Thread(target=self._executor.spin, daemon=True).start()
        monitor = self.monitor
        self.wait_until(lambda: monitor.navigation_is_active(), 300, "Nav2 to activate")
        self.wait_until(lambda: monitor.costmap_received, 300, "the global costmap")

        if self.coordinator:
            coordinator = ["ros2", "run", "offline_bringup_active_mapmaintenance", "main_coordinator", "--ros-args"]
            coordinator_process = self._start(
                "coordinator", OFFLINE_BRINGUP, "dev", [*coordinator, "-p", "use_sim_time:=true"], keep_stdin_open=True
            )
            # its log rather than thought_process, which keeps no history and may announce this before we match it
            self.wait_until(
                lambda: coordinator_process.log_contains("Ready to receive tasks"),
                300,
                "main_coordinator to initialise",
            )
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
    coordinator: bool = False
    perceive: bool = False
    start: tuple[float, float, float] = DEFAULT_START
    scene: Path = DEFAULT_SCENE


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


def run_action(stack: Stack, monitor: Monitor, client: ActionClient, goal: object, timeout: float) -> Optional[int]:
    """Sends an action goal and returns its final status, or None if it did not finish within `timeout` sim seconds."""
    goal_future = client.send_goal_async(goal)
    if not wait_simulated(stack, monitor, 10.0, goal_future.done) or not goal_future.result().accepted:
        return GoalStatus.STATUS_ABORTED
    result_future = goal_future.result().get_result_async()
    if not wait_simulated(stack, monitor, timeout, result_future.done):
        goal_future.result().cancel_goal_async()
        return None
    return result_future.result().status


def navigate_across_house(stack: Stack, monitor: Monitor) -> list[str]:
    goal_x, goal_y = -1.5, 3.0
    goal = NavigateToPose.Goal()
    goal.pose = PoseStamped()
    goal.pose.header.frame_id = "map"
    goal.pose.pose.position.x, goal.pose.pose.position.y = goal_x, goal_y
    goal.pose.pose.orientation.w = 1.0
    status = run_action(stack, monitor, monitor.navigate_client, goal, timeout=240.0)
    failures = []
    if status != GoalStatus.STATUS_SUCCEEDED:
        failures.append(f"NavigateToPose ended with status {status} instead of SUCCEEDED (4)")
    pose = monitor.pose()
    if pose is None or math.dist(pose[:2], (goal_x, goal_y)) > 0.35:
        failures.append(f"robot ended at {pose}, more than 0.35 m from the goal ({goal_x}, {goal_y})")
    return failures


def spin_in_place(stack: Stack, monitor: Monitor) -> list[str]:
    start = monitor.pose()
    goal = Spin.Goal()
    goal.target_yaw = math.pi / 2
    status = run_action(stack, monitor, monitor.spin_client, goal, timeout=60.0)
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


def coordinator_task(query: str, minimum_waypoints: int, timeout: float) -> Callable[[Stack, Monitor], list[str]]:
    """The coordinator task `query` must reach `minimum_waypoints` navigation goals and abort no more than it reaches."""

    def run(stack: Stack, monitor: Monitor) -> list[str]:
        monitor.send_user_query(query)
        wait_simulated(
            stack, monitor, timeout, lambda: monitor.navigation_count(GoalStatus.STATUS_SUCCEEDED) >= minimum_waypoints
        )
        succeeded = monitor.navigation_count(GoalStatus.STATUS_SUCCEEDED)
        aborted = monitor.navigation_count(GoalStatus.STATUS_ABORTED)
        failures = []
        if succeeded < minimum_waypoints:
            failures.append(f"reached {succeeded} of {minimum_waypoints} waypoints within {timeout:.0f} s")
        if aborted > succeeded:
            failures.append(f"{aborted} navigation goals aborted, more than the {succeeded} reached")
        return failures

    return run


SCENARIOS = {
    scenario.name: scenario
    for scenario in [
        Scenario(
            "navigate_across_house", "Nav2 drives from the west room to the centre of the house.", navigate_across_house
        ),
        Scenario("spin_in_place", "Nav2's Spin behavior turns the robot by 90 deg on the spot.", spin_in_place),
        Scenario(
            "odometry_consistency",
            "/odom's twist is the body-frame velocity its poses imply, for straight, turning and curved driving.",
            odometry_consistency,
        ),
        Scenario(
            "coordinator_patrol",
            "main_coordinator patrols the ground-truth map: three waypoints reached.",
            coordinator_task("patrol", minimum_waypoints=3, timeout=600.0),
            coordinator=True,
        ),
        Scenario(
            "coordinator_random_walk",
            "main_coordinator random walk on the ground-truth map: three waypoints reached.",
            coordinator_task("random", minimum_waypoints=3, timeout=600.0),
            coordinator=True,
        ),
        Scenario(
            "coordinator_explore",
            "main_coordinator explores frontiers of perceive_semantix's map: three waypoints reached.",
            coordinator_task("explore", minimum_waypoints=3, timeout=900.0),
            coordinator=True,
            perceive=True,
        ),
    ]
}


def write_results(run_directory: Path, scenario: Scenario, failures: list[str], monitor: Optional[Monitor]) -> None:
    if monitor is not None:
        with (run_directory / "trajectory.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["simulation_time", "x", "y", "yaw"])
            writer.writerows(monitor.trajectory)
    result = {
        "scenario": scenario.name,
        "passed": not failures,
        "failures": failures,
        "distance_travelled": monitor.distance_travelled() if monitor else None,
        "final_pose": monitor.pose() if monitor else None,
        "navigation_succeeded": monitor.navigation_count(GoalStatus.STATUS_SUCCEEDED) if monitor else None,
        "navigation_aborted": monitor.navigation_count(GoalStatus.STATUS_ABORTED) if monitor else None,
        "thoughts": monitor.thoughts if monitor else [],
    }
    (run_directory / "result.json").write_text(json.dumps(result, indent=2))


def new_stack(
    arguments: argparse.Namespace, name: str, coordinator: bool, perceive: bool, scene: Path, start: tuple
) -> Stack:
    domain_id = arguments.domain_id if arguments.domain_id is not None else random.randint(100, 200)
    run_directory = TEST_RUNS / f"{datetime.now():%Y%m%d_%H%M%S}_{name}"
    return Stack(run_directory, domain_id, not arguments.gui, coordinator, perceive, scene, start)


def wait_for_interrupt(stack: Stack) -> None:
    print(f"  stack is running; attach with {stack.write_attach_script()} <command>, Ctrl-C to stop")
    try:
        while True:
            stack.check_processes()
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


def run_scenario(scenario: Scenario, arguments: argparse.Namespace) -> bool:
    stack = new_stack(arguments, scenario.name, scenario.coordinator, scenario.perceive, scenario.scene, scenario.start)
    print(
        f"[{scenario.name}] {scenario.description}\n  run directory {stack.run_directory}, ROS domain {stack.domain_id}"
    )
    monitor = None
    try:
        monitor = stack.up()
        failures = scenario.run(stack, monitor)
    except (RuntimeError, TimeoutError) as error:
        failures = [f"stack failure: {error}"]
    except KeyboardInterrupt:
        failures = ["interrupted"]
    write_results(stack.run_directory, scenario, failures, monitor)
    print(f"  {'PASSED' if not failures else 'FAILED'}" + "".join(f"\n    - {failure}" for failure in failures))
    if arguments.keep_running and monitor is not None:
        wait_for_interrupt(stack)
    stack.down()
    return not failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List the scenarios.")
    run_parser = commands.add_parser("run", help="Run scenarios, each on a fresh stack.")
    run_parser.add_argument("scenarios", nargs="+", help="Scenario names, or 'all'.")
    run_parser.add_argument("--keep-running", action="store_true", help="Keep each stack up after its scenario.")
    stack_parser = commands.add_parser("stack", help="Only bring the stack up, for manual debugging.")
    stack_parser.add_argument("--coordinator", action="store_true", help="Also start main_coordinator.")
    stack_parser.add_argument(
        "--perceive", action="store_true", help="perceive_semantix instead of the ground-truth map."
    )
    stack_parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    stack_parser.add_argument("--start", type=float, nargs=3, default=DEFAULT_START, metavar=("X", "Y", "YAW"))
    for subparser in (run_parser, stack_parser):
        subparser.add_argument(
            "--gui", action="store_true", help="Show the Isaac Sim window instead of running headless."
        )
        subparser.add_argument("--domain-id", type=int, default=None, help="ROS domain (default: random 100-200).")
    arguments = parser.parse_args()

    if arguments.command == "list":
        for scenario in SCENARIOS.values():
            requirements = ", ".join(
                n for n, needed in (("coordinator", scenario.coordinator), ("perceive", scenario.perceive)) if needed
            )
            print(f"{scenario.name:28} {scenario.description}" + (f" [{requirements}]" if requirements else ""))
        return 0

    if arguments.command == "stack":
        stack = new_stack(
            arguments,
            "stack",
            arguments.coordinator,
            arguments.perceive,
            arguments.scene.resolve(),
            tuple(arguments.start),
        )
        print(f"run directory {stack.run_directory}, ROS domain {stack.domain_id}")
        try:
            stack.up()
            wait_for_interrupt(stack)
        except (RuntimeError, TimeoutError) as error:
            print(f"stack failure: {error}")
        finally:
            stack.down()
        return 0

    names = list(SCENARIOS) if arguments.scenarios == ["all"] else arguments.scenarios
    unknown = [name for name in names if name not in SCENARIOS]
    if unknown:
        parser.error(f"unknown scenarios {unknown}; see 'list'")
    results = {name: run_scenario(SCENARIOS[name], arguments) for name in names}
    print("\n".join(f"{'PASSED' if passed else 'FAILED'}  {name}" for name, passed in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
