import argparse
import json
import os
from pathlib import Path
import pty
import shutil
import signal
import subprocess
import sys
import threading
import time
from enum import Enum
from typing import Any, Literal, Optional, Union
import pickle

import numpy as np


class OutMode(Enum):
    CONSOLE = 0
    DISABLED = 1
    ERRORS_ONLY = 2


COLORS = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "reset": "\033[0m",
}

STATE_LOCK = threading.Lock()
STATE_LIST: list = []

SUCCESS_FEEDBACK_DEFAULT: str = ""
SUCCESS_FEEDBACK_LOCK = threading.Lock()
SUCCESS_FEEDBACK: str = SUCCESS_FEEDBACK_DEFAULT

GOAL_POSITIONS_LOCK = threading.Lock()
GOAL_POSITIONS: Optional[np.ndarray] = None

GOAL_SHORTEST_DISTANCE_LOCK = threading.Lock()
GOAL_SHORTEST_DISTANCE: Optional[float] = None


class BufferClass:
    def __init__(self, max_size: int = 4096):
        self.data = ""
        self.max_size = max_size

    def cut(self):
        if len(self.data) > self.max_size:
            self.data = self.data[-self.max_size :]

    @staticmethod
    def last_tag(buffer: str, tag: str) -> Optional[tuple[int, int]]:
        end_tag = f"</{tag}>"
        start_tag = f"<{tag}>"

        end = buffer.rfind(end_tag)
        if end == -1:
            return None

        start = buffer.rfind(start_tag, 0, end)
        if start == -1:
            return None

        return start, end + len(end_tag)


def parse_sim_state(text: str, buffer: BufferClass):
    """Try to deserialize JSON from text; return state tuple or None if invalid."""
    buffer.data += text
    buffer.cut()

    last_robot_tag = BufferClass.last_tag(buffer.data, "robot")
    last_goals_tag = BufferClass.last_tag(buffer.data, "goals")
    if last_robot_tag is not None:
        robot_text = buffer.data[last_robot_tag[0] + len("<robot>") : last_robot_tag[1] - len("</robot>")]
        try:
            data = json.loads(robot_text)
            time = data["time"]
            position = [data["position"]["x"], data["position"]["y"], data["position"]["z"]]
            orientation = [
                data["orientation"]["w"],
                data["orientation"]["x"],
                data["orientation"]["y"],
                data["orientation"]["z"],
            ]
            linear_velocity = [
                data["linear_velocity"]["vx"],
                data["linear_velocity"]["vy"],
                data["linear_velocity"]["vz"],
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        else:
            global STATE_LOCK, STATE_LIST
            with STATE_LOCK:
                STATE_LIST.append((time, position, orientation, linear_velocity))

    if last_goals_tag is not None:
        goals_text = buffer.data[last_goals_tag[0] + len("<goals>") : last_goals_tag[1] - len("</goals>")]
        try:
            data = json.loads(goals_text)
            positions = []
            shortest_distance = None
            for key in data.keys():
                if key == "shortest_distance":
                    shortest_distance = data[key]
                else:
                    pos = data[key]
                    positions.append([pos["x"], pos["y"], pos["z"]])
            positions_array = np.array(positions)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print("Failed to parse goal positions from sim output:", e)
        else:
            if np.size(positions_array) == 0:
                return
            global GOAL_POSITIONS_LOCK, GOAL_POSITIONS
            with GOAL_POSITIONS_LOCK:
                GOAL_POSITIONS = positions_array
            global GOAL_SHORTEST_DISTANCE_LOCK, GOAL_SHORTEST_DISTANCE
            with GOAL_SHORTEST_DISTANCE_LOCK:
                GOAL_SHORTEST_DISTANCE = shortest_distance

    if last_robot_tag is not None or last_goals_tag is not None:
        # Remove processed data from buffer
        last_end = max(
            last_robot_tag[1] if last_robot_tag is not None else 0,
            last_goals_tag[1] if last_goals_tag is not None else 0,
        )
        buffer.data = buffer.data[last_end:]


def print_line(line: str, prefix: str = ""):
    sys.stdout.write(f"{prefix}{line.strip()}\n")


def print_error_line(line: str, prefix: str = ""):
    if "error" in line.lower() or "exception" in line.lower():
        sys.stdout.write(f"{prefix}{line.strip()}\n")


def success_monitor(success_distance_threshold: float):
    global SUCCESS_FEEDBACK_LOCK, SUCCESS_FEEDBACK, GOAL_POSITIONS_LOCK, GOAL_POSITIONS, STATE_LOCK, STATE_LIST
    success_reported = False
    with SUCCESS_FEEDBACK_LOCK:
        if "SUCCESS" in SUCCESS_FEEDBACK:
            success_reported = True

    goal_positions = None
    with GOAL_POSITIONS_LOCK:
        if GOAL_POSITIONS is not None:
            goal_positions = np.array(GOAL_POSITIONS[:, :2])  # x, y

    last_state = None
    with STATE_LOCK:
        if STATE_LIST:
            last_state = STATE_LIST[-1]

    distance = float("inf")
    if goal_positions is not None and last_state is not None:
        position = np.array(last_state[1][:2])  # x, y
        distances = np.linalg.norm(goal_positions - position, axis=1)
        distance = float(np.min(distances))

    return success_reported and distance < success_distance_threshold


class ProcessHandler:
    def __init__(
        self,
        proc,
        master_fd: int,
        name: str,
        color: str,
        triggers: dict[Union[str, float], str],
        mode=OutMode.CONSOLE,
        line_handlers: list = [],
    ):
        self.proc = proc
        # master_fd is the integer file descriptor for the PTY master
        self.master_fd = master_fd
        self.name = name
        self.color = color
        self.triggers = triggers
        self.mode = mode

        self.start = time.time()
        self.fired_triggers = dict()

        self.prefix = f"{self.color}[{self.name}]{COLORS['reset']} "
        self.line_handler = line_handlers
        if self.mode == OutMode.CONSOLE:
            self.line_handler.append(lambda line: print_line(line, self.prefix))
        elif self.mode == OutMode.ERRORS_ONLY:
            self.line_handler.append(lambda line: print_error_line(line, self.prefix))

    def forward_output_and_handle_input(self):
        # Read raw bytes from the PTY master fd so prompts without newlines are shown
        accum = ""
        try:
            while True:
                try:
                    data = os.read(self.master_fd, 1024)
                except OSError:
                    break
                if not data:
                    break
                try:
                    text = data.decode("utf-8", errors="ignore")
                except Exception:
                    text = ""

                # Print text to stdout, adding prefix at line starts
                parts = text.splitlines(keepends=False)
                for line in parts:
                    if not line:
                        continue
                    for handler in self.line_handler:
                        handler(line)
                sys.stdout.flush()

                # Check string triggers against the accumulated text
                accum += text
                now = time.time() - self.start
                for pattern, response in self.triggers.items():
                    if isinstance(pattern, str) and pattern in accum and not self._recently_fired(pattern, now):
                        self.fired_triggers[pattern] = now
                        global SUCCESS_FEEDBACK_LOCK, SUCCESS_FEEDBACK
                        if "SUCCESS" in response:
                            if SUCCESS_FEEDBACK == "FAILURE":
                                if self.mode == OutMode.CONSOLE:
                                    sys.stdout.write(
                                        f"{self.prefix}SUCCESS condition detected but detected FAILURE earlier!\n"
                                    )
                                    sys.stdout.flush()
                            else:
                                with SUCCESS_FEEDBACK_LOCK:
                                    SUCCESS_FEEDBACK = "SUCCESS"
                                if self.mode == OutMode.CONSOLE:
                                    sys.stdout.write(f"{self.prefix}SUCCESS condition detected!\n")
                                    sys.stdout.flush()
                        elif "FAILURE" in response:
                            with SUCCESS_FEEDBACK_LOCK:
                                SUCCESS_FEEDBACK = "FAILURE"
                            if self.mode == OutMode.CONSOLE:
                                sys.stdout.write(f"{self.prefix}FAILURE condition detected!\n")
                                sys.stdout.flush()
                        else:
                            self._write_to_input(response)
                            if self.mode == OutMode.CONSOLE:
                                sys.stdout.write(f"{self.prefix}Fired string trigger '{pattern}': {response.strip()}\n")
                                sys.stdout.flush()

                accum = accum.splitlines(keepends=False)[-1]
        finally:
            try:
                os.close(self.master_fd)
            except Exception:
                pass

    def _recently_fired(self, pattern: Union[str, float], now: float, cooldown: float = 2.0) -> bool:
        """Check if a trigger was fired within the cooldown period."""
        if pattern not in self.fired_triggers:
            return False
        last_fired = self.fired_triggers[pattern]
        return now - last_fired < cooldown

    def _write_to_input(self, response: str):
        try:
            os.write(self.master_fd, response.encode())
        except Exception:
            pass

    def handle_time_triggers(self):
        if self.proc.poll() is not None:
            return  # process has exited
        now = time.time() - self.start
        for pattern, response in self.triggers.items():
            if (
                isinstance(pattern, (int, float))
                and pattern <= now
                and not self._recently_fired(pattern, now, cooldown=float("inf"))
            ):
                if self.mode == OutMode.CONSOLE:
                    sys.stdout.write(f"{self.prefix}Fired time trigger at {now:.1f}s: {response.strip()}\n")
                self.fired_triggers[pattern] = now
                self._write_to_input(response)


def launch_processes(
    processes: dict[str, Any],
) -> tuple[list[subprocess.Popen], list[ProcessHandler]]:
    for p in processes:
        cwd = p.get("cwd")
        if not cwd:
            continue
        if not os.path.isdir(cwd):
            sys.stderr.write(f"Error: cwd for process '{p.get('name', '<unknown>')}' does not exist: {cwd}\n")
            sys.exit(1)

    running_processes = []
    process_handlers = []
    for p in processes:
        master_fd, slave_fd = pty.openpty()
        proc = subprocess.Popen(
            p["cmd"],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=subprocess.STDOUT,
            cwd=p["cwd"],
            preexec_fn=os.setsid,
            close_fds=True,
            text=False,
            shell=p.get("shell", False),
        )
        try:
            os.close(slave_fd)
        except Exception:
            pass

        # Launch thread to read output from the master fd
        handler = ProcessHandler(
            proc,
            master_fd,
            p["name"],
            p["color"],
            p["triggers"],
            p["output"],
            line_handlers=p.get("line_handlers", []),
        )
        t = threading.Thread(target=handler.forward_output_and_handle_input)
        t.daemon = True
        t.start()

        process_handlers.append(handler)
        running_processes.append(proc)

        if p["name"] == "IsaacSim":
            while True:
                with STATE_LOCK:
                    if len(STATE_LIST) > 0:
                        break
                print("Waiting for IsaacSim to start up...")
                time.sleep(1.0)
    return running_processes, process_handlers


def terminate_processes(procs, timeout=2):
    """
    Terminate a list of subprocesses reliably.
    - First sends SIGTERM.
    - Waits up to `timeout` seconds.
    - Sends SIGKILL to any remaining processes.
    """
    pids = [proc.pid for proc in procs]
    gpids = []
    for pid in pids:
        try:
            gpid = os.getpgid(pid)
            gpids.append(gpid)
        except ProcessLookupError:
            pass
    all_pids = pids + gpids

    for id in all_pids:
        try:
            os.killpg(id, signal.SIGINT)
        except ProcessLookupError:
            pass

    time.sleep(1.0)
    for id in all_pids:
        try:
            os.killpg(id, signal.SIGTERM)
        except ProcessLookupError:
            pass

    # Wait for processes to exit gracefully
    print("Waiting for processes to terminate...")
    end_time = time.time() + timeout
    for proc in procs:
        while proc.poll() is None and time.time() < end_time:
            time.sleep(0.05)

    print("Sending SIGTERM again to ensure termination...")
    for id in all_pids:
        try:
            os.killpg(id, signal.SIGTERM)
        except ProcessLookupError:
            pass

    # Wait for processes to exit gracefully
    print("Waiting for processes to terminate...")
    end_time = time.time() + timeout
    for proc in procs:
        while proc.poll() is None and time.time() < end_time:
            time.sleep(0.05)

    # Force kill any remaining processes
    print(f"Force killing remaining processes... (overall pids {pids}, and gpids {gpids})")
    for id in all_pids:
        try:
            os.killpg(id, signal.SIGKILL)
        except ProcessLookupError as e:
            print(f"Process already exited: {e}")


def latest_pkl(folder: Path) -> Optional[Path]:
    files = [f for f in folder.glob("*.pkl")]
    return max(files, key=lambda f: int(f.stem.split("-")[0])) if files else None


def check_existing_record(record: str, output_file: Path) -> bool:
    try:
        with open(output_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return False

    return any(d.get("name") == record for d in data)

def delete_existing_record(record: str, output_file: Path) -> None:
    try:
        with open(output_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return

    data = [d for d in data if d.get("name") != record]
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

def store_results(
    record: str,
    app: str,
    output_file: Path,
    experiment: dict,
    output_root: Path,
    path_length: float,
    state_array: np.ndarray,
    success: bool,
    time_to_complete: float,
    log_files: list[Path],
    goal_shortest_distance: Optional[float] = None,
    genmap: bool = False,
):
    if genmap:
        new_record = {
            "name": record,
            "app": app,
            "experiment": experiment,
            "log_files": [str(log_file) for log_file in log_files],
        }
    else:
        state_file = output_root / f"{record}_state_trajectory.npy"
        log_files = list(log_files)
        log_files += [state_file]

        new_record = {
            "name": record,
            "app": app,
            "experiment": experiment,
            "state_trajectory_file": state_file.resolve().absolute().as_posix(),
            "time_to_complete": time_to_complete,
            "path_length": path_length,
            "success": success,
            "log_files": [str(log_file) for log_file in log_files],
        }

        if goal_shortest_distance is not None:
            new_record["goal_shortest_distance"] = goal_shortest_distance

    try:
        with open(output_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []
    if not any(d.get("name") == new_record.get("name") for d in data):
        data.append(new_record)
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        
        if not genmap:
            np.save(state_file, state_array)
    else:
        print(f"Experiment record '{record}' already exists in results file.")


def build_proccesses(
    app: Literal["dynamem", "perceivesemantix", "random", "genmap"], experiment: dict, output_root: Path
) -> tuple[list[dict[str, Any]], list[Path]]:
    if app.lower() not in ["dynamem", "perceivesemantix", "random", "genmap"]:
        raise ValueError(f"Unsupported app: {app}")

    issac_sim_options = []
    if "asset" in experiment["goal"]:
        asset = experiment["goal"]["asset"]
        if "position" in experiment["goal"]:
            position = experiment["goal"]["position"]
            theta = experiment["goal"].get("theta", 0.0)
            issac_sim_options += [
                "--asset",
                str(asset),
                str(position[0]),
                str(position[1]),
                str(position[2]),
                str(theta),
            ]
        else:
            issac_sim_options += [
                "--gasset",
                str(asset),
            ]
    if "exclude_goal_asset" in experiment["goal"]:
        exclude_asset = experiment["goal"]["exclude_goal_asset"]
        if isinstance(exclude_asset, str):
            exclude_asset = [exclude_asset]
        if len(exclude_asset) > 0:
            issac_sim_options += ["--gasset-exclude"] + exclude_asset
    if "remove_assets" in experiment:
        if experiment["remove_assets"]:
            if isinstance(experiment["remove_assets"], str):
                experiment["remove_assets"] = [experiment["remove_assets"]]
            if len(experiment["remove_assets"]) > 0:
                issac_sim_options += ["--rasset"] + experiment["remove_assets"]
    if "exclude_remove_assets" in experiment:
        if experiment["exclude_remove_assets"]:
            if isinstance(experiment["exclude_remove_assets"], str):
                experiment["exclude_remove_assets"] = [experiment["exclude_remove_assets"]]
            if len(experiment["exclude_remove_assets"]) > 0:
                issac_sim_options += ["--rasset-exclude"] + experiment["exclude_remove_assets"]
    if "robot_start" in experiment:
        position = experiment["robot_start"]["position"]
        theta = experiment["robot_start"].get("theta", 0.0)
        issac_sim_options += [
            "--robot-start",
            str(position[0]),
            str(position[1]),
            str(position[2]),
            str(theta),
        ]


    output_dir = output_root / app.lower() / Path(experiment.get("scene")).stem / experiment["name"]
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if app.lower() == "genmap":
        issac_sim_options += ["--generate-map", (output_dir / "map.npz").as_posix()]
        if "prior_map_object" in experiment["goal"]:
            if isinstance(experiment["goal"]["prior_map_object"], str):
                experiment["goal"]["prior_map_object"] = [experiment["goal"]["prior_map_object"]]
            issac_sim_options += ["--generate-map-objs"] + experiment["goal"]["prior_map_object"]

    issac_sim_parse_buffer = BufferClass()
    processes = [
        {
            "name": "DiscoveryServer",
            "cmd": ["fastdds discovery -i 0 -l 127.0.0.1 -p 14520"],
            "cwd": "/home/benni/",
            "color": COLORS["magenta"],
            "triggers": {},
            "output": OutMode.CONSOLE,
            "shell": True,
        },
        {
            "name": "IsaacSim",
            "cmd": [
                "pixi",
                "run",
                "python",
                "standalone_sim.py",
                "--scene",
                str(experiment.get("scene")),
                "--lighting",
                experiment.get("lighting", "stage"),
                *issac_sim_options,
            ],
            "cwd": "/home/benni/repos/stretch_isaac/",
            "color": COLORS["red"],
            "triggers": {},
            "output": OutMode.DISABLED if app != "genmap" else OutMode.CONSOLE,
            "line_handlers": [lambda x: parse_sim_state(x, issac_sim_parse_buffer)],
        },
    ]

    do_explore = experiment["goal"]["task"] == "explore"
    if "initialmap_experiment" in experiment:
        input_app = app.lower() if app.lower() != "random" else "perceivesemantix"
        input_path = (
            output_root / input_app / Path(experiment.get("scene")).stem / experiment["initialmap_experiment"]
        )
        if app.lower() == "dynamem":
            input_path = input_path.with_suffix(".pkl")
        elif app.lower() == "perceivesemantix" or app.lower() == "random":
            input_path = input_path / "output"
            input_file = latest_pkl(input_path)
            if input_file is None:
                raise FileNotFoundError(f"No exploration pkl files found in {input_path}")
            input_path = input_file
        input_path = input_path.resolve()
    else:
        input_path = None

    output_files = []
    if app.lower() == "dynamem":
        if do_explore:
            dynamem_log = Path("/home/benni/repos/stretch_ai/dynamem_log")
            rel_out_dir = Path(os.path.relpath(output_dir, dynamem_log))
            options = [
                "--output-path",
                str(rel_out_dir),
                "--explore-iter",
                "4",
                "--max-search-steps",
                "1",
            ]
            # in exploration mode the map is not saved, so instead we search for an object (volcano) which is never present
            triggers = {
                "Enter desired mode [E (explore and mapping) / M (Open vocabulary pick and place)]": "E\n",
                "Enter the target object:": "volcano\n",
                "Enter the target receptacle:": "volcano\n",
                "Do you want to run navigation? [Y/n]:": "Y\n",
                "Do you want to run picking? [Y/n]:": "n\n",
                "Do you want to run placement? [Y/n]:": "n\n",
            }
            output_files += [str(output_dir), str(output_dir.with_suffix(".pkl"))]
        else:
            triggers = {
                "Enter desired mode [E (explore and mapping) / M (Open vocabulary pick and place)]": "M\n",
                "Enter the target object:": f"{experiment['goal']['label']}\n",
                "Enter the target receptacle:": f"{experiment['goal']['label']}\n",
                "Do you want to run navigation? [Y/n]:": "Y\n",
                "Do you want to run picking? [Y/n]:": "SUCCESS\n",
                "Do you want to run picking? [Y/n]": "n\n",
                "Navigation Failure: Could not find": "FAILURE\n",
                "Do you want to run placement? [Y/n]": "n\n",
            }
            options = [
                "--max-search-steps",
                "30",
            ]
        if input_path is not None:
            options += [
                "--input-path",
                str(input_path),
            ]
        processes += [
            {
                "name": "Ros2BridgeServer",
                "cmd": ["../scripts/run_stretch_ai_ros2_bridge_server.sh"],
                "cwd": "/home/benni/repos/stretch_ai/docker",
                "color": COLORS["yellow"],
                "triggers": {},
                "output": OutMode.DISABLED,
            },
            {
                "name": "DynaMem",
                "cmd": [
                    "pixi",
                    "run",
                    "python",
                    "-m",
                    "stretch.app.run_dynamem",
                    "--robot_ip",
                    "127.0.0.1",
                    *options,
                ],
                "cwd": "/home/benni/repos/stretch_ai",
                "color": COLORS["green"],
                "triggers": triggers,
                "output": OutMode.CONSOLE,
            },
        ]
    elif app.lower() == "perceivesemantix" or app.lower() == "random":
        initial_scene_path = str(input_path) if input_path is not None else '""'
        prefix = "random_" if app.lower() == "random" else ""
        triggers = {15.0: "explore\n"} if do_explore else {15.0: f"{prefix}{experiment['goal']['label']}\n"}
        triggers[" found at "] = "SUCCESS\n"
        output_files += [str(output_dir)]
        processes += [
            {
                "name": "PerceiveSemantix",
                "cmd": [
                    "pixi",
                    "run",
                    "ros2",
                    "run",
                    "perceive_semantix_ros2",
                    "perceive_semantix_node",
                    "--ros-args",
                    "-p",
                    "camera_depth_scale_to_m:=1.0",
                    "-p",
                    "image_rotations_clockwise:=1",
                    "-p",
                    "occupancy_map/floor_height:=0.15",
                    "-p",
                    f"store_output:={str(do_explore)}",
                    "-p",
                    "publishing_rate_background_pointcloud:=0.0",
                    "-p",
                    "objects/point_cloud/publishing_rate:=0.0",
                    "-p",
                    "occupancy_map/publishing_rate:=0.5",
                    "-p",
                    f"initial_scene_path:={initial_scene_path}",
                    "-p",
                    f"output_path:={str(output_dir)}",
                ],
                "cwd": "/home/benni/repos/bringup_active_mapmaintenance/perceive_semantix/",
                "color": COLORS["blue"],
                "triggers": {},
                "output": OutMode.DISABLED,
            },
            {
                "name": "StretchMPC",
                "cmd": [
                    "pixi",
                    "run",
                    "ros2",
                    "launch",
                    "stretch_mpc_ros",
                    "planner.launch.py",
                ],
                "cwd": "/home/benni/repos/bringup_active_mapmaintenance/stretch_mpc/",
                "color": COLORS["yellow"],
                "triggers": {},
                "output": OutMode.DISABLED,
            },
            {
                "name": "MainCoordinator",
                "cmd": [
                    "pixi",
                    "run",
                    "ros2",
                    "run",
                    "offline_bringup_active_mapmaintenance",
                    "main_coordinator",
                ],
                "cwd": "/home/benni/repos/bringup_active_mapmaintenance/offline_bringup_active_mapmaintenance/",
                "color": COLORS["green"],
                "triggers": triggers,
                "output": OutMode.CONSOLE,
            },
            # {
            #     "name": "NavigationGoalActionClient",
            #     "cmd": [
            #         "pixi",
            #         "run",
            #         "ros2",
            #         "run",
            #         "stretch_mpc_ros",
            #         "navigation_goal_action_client",
            #     ],
            #     "cwd": "/home/benni/repos/bringup_active_mapmaintenance/stretch_mpc/",
            #     "color": COLORS["green"],
            #     "triggers": {},
            #     "output": OutMode.DISABLED,
            # },
        ]
    elif app == "genmap":
        output_files += [str(output_dir / "map.npz")]
    return processes, output_files


def robot_has_moved(translation_threshold: float = 0.01, orientation_threshold: float = 1) -> bool:
    global STATE_LOCK, STATE_LIST
    with STATE_LOCK:
        if len(STATE_LIST) < 2:
            return False
        previous_state = STATE_LIST[-2]
        previous_position = np.array(previous_state[1][:2])  # x, y
        previous_orientation = np.array(previous_state[2])  # w, x, y, z
        previous_orientation /= np.linalg.norm(previous_orientation)

        state_now = STATE_LIST[-1]
        position_now = np.array(state_now[1][:2])  # x, y
        orientation_now = np.array(state_now[2])  # w, x, y, z
        orientation_now /= np.linalg.norm(orientation_now)

    position_change = np.linalg.norm(position_now - previous_position)
    angular_change = 2 * np.arccos(np.clip(np.abs(np.dot(previous_orientation, orientation_now)), -1.0, 1.0))
    return position_change > translation_threshold or angular_change > np.deg2rad(orientation_threshold)


def run_expriment(app: Literal["dynamem", "perceivesemantix"], experiment: dict, output_root: Path, overwrite: bool = False) -> bool:
    global STATE_LOCK, STATE_LIST, GOAL_POSITIONS_LOCK, GOAL_POSITIONS, SUCCESS_FEEDBACK_LOCK, SUCCESS_FEEDBACK, GOAL_SHORTEST_DISTANCE_LOCK, GOAL_SHORTEST_DISTANCE
    processes, log_files = build_proccesses(app, experiment, output_root)

    record_key = f"{experiment['name']}_{app.lower()}"
    output_file = output_root / "experiments_results.json"
    if check_existing_record(record_key, output_file):
        if overwrite:
            print(f"Experiment record '{record_key}' already exists. Overwriting as requested.")
            delete_existing_record(record_key, output_file)
        else:
            print(f"Experiment record '{record_key}' already exists. Skipping experiment.")
            return True
    
    print(f" ############################ Running experiment '{record_key}' ############################")
    print("Making sure FASTDDS is not running...")
    subprocess.run("ps aux | grep discovery | grep -v grep | awk '{print $2}' | xargs kill", shell=True, check=False)
    time.sleep(1.0)

    # This command finds all GPU processes using >1GB and kills them
    print("Killing any existing GPU processes using >1GB...")
    cmd = """
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | \
    while IFS=',' read -r pid mem; do
        if [ "$mem" -gt 1000 ]; then
            echo "Killing PID $pid using $mem MiB"
            kill -9 "$pid"
        fi
    done
    """
    subprocess.run(cmd, shell=True, check=False, executable="/bin/bash")
    time.sleep(1.0)

    for p in processes:
        cmd = " ".join(p["cmd"])
        cwd = str(p["cwd"])
        print(f'"{cmd}", cwd = "{cwd}"')

    running_processes, process_handlers = launch_processes(processes)
    active_timeout: Optional[float] = experiment.get("max_runtime", None)
    total_timeout: Optional[float] = (active_timeout + 5 * 60.0) if active_timeout is not None else None
    regular_exit = True

    do_explore = experiment["goal"]["task"] == "explore"
    if not do_explore:
        if "goal" in experiment and "position" in experiment["goal"]:
            with GOAL_POSITIONS_LOCK:
                GOAL_POSITIONS = np.array(
                    [
                        [experiment["goal"]["position"][0], experiment["goal"]["position"][1], 0.0],
                    ]
                )
        success = False
    else:
        success = True  # exploration always "succeeds"

    # setup timeout watcher
    initial_movement_detected = False
    active_timeout_start: Optional[float] = None
    total_timeout_start: float = time.time()

    # Wait for all to finish
    try:
        while all([proc.poll() is None for proc in running_processes]):
            time.sleep(0.1)

            # If max runtime was provided, start watcher thread
            if total_timeout is not None:
                elapsed = time.time() - total_timeout_start
                if elapsed >= total_timeout:
                    sys.stdout.write(f"Total runtime of {total_timeout}s exceeded. Terminating processes.\n")
                    break

            if active_timeout is not None:
                if not initial_movement_detected and robot_has_moved():
                    initial_movement_detected = True
                    active_timeout_start = time.time()
                    sys.stdout.write("Detected robot movement. Starting timeout timer.\n")
                if active_timeout_start is not None:
                    elapsed = time.time() - active_timeout_start
                    if elapsed >= active_timeout:
                        sys.stdout.write(f"Max runtime of {active_timeout}s exceeded. Terminating processes.\n")
                        break

            for handler in process_handlers:
                handler.handle_time_triggers()

            if not do_explore and success_monitor(1.5):
                success = True
                sys.stdout.write("Success condition met. Terminating processes.\n")
                break
        
        exit_codes = [proc.poll() for proc in running_processes]
        for exit_code, proc, handler in zip(exit_codes, running_processes, process_handlers):
            if exit_code is None:
                continue
            elif exit_code == 0:
                print(f"Process {handler.name} exited with code {exit_code}.")
            else:
                print(f"Process {handler.name} exited with code {exit_code}. Something went wrong.")
                regular_exit = False
      
    except KeyboardInterrupt:
        print("\nStopping processes...")
    finally:
        time_to_complete = time.time() - (active_timeout_start or total_timeout_start)
        terminate_processes(running_processes, timeout=2)
        print("All processes terminated.")

    # Reset globals
    with STATE_LOCK:
        state_trajectory = STATE_LIST.copy()
        STATE_LIST.clear()
    with SUCCESS_FEEDBACK_LOCK:
        SUCCESS_FEEDBACK = SUCCESS_FEEDBACK_DEFAULT
    with GOAL_POSITIONS_LOCK:
        GOAL_POSITIONS = None
    with GOAL_SHORTEST_DISTANCE_LOCK:
        goal_shortest_distance = GOAL_SHORTEST_DISTANCE
        GOAL_SHORTEST_DISTANCE = None

    # Compute path length
    state_array = np.array(
        [[time] + pos + ori + vel for time, pos, ori, vel in state_trajectory],
        dtype=float,
    )
    path_length = np.linalg.norm(np.diff(state_array[:, 1:3], axis=0), axis=1).sum()

    # Store results
    store_results(
        record_key, app, output_file, experiment, output_root, path_length, state_array, success, time_to_complete, log_files, goal_shortest_distance=goal_shortest_distance, genmap=(app=="genmap")
    )

    if app == "dynamem" and len(log_files) > 0:
        # check that pkl can be loaded
        standard_pkl = Path(log_files[-1])
        out_pkls = [standard_pkl.with_suffix(".0.pkl"), standard_pkl.with_suffix(".1.pkl")]
        out_pkls_loadable = [False, False]
        for i, pkl in enumerate(out_pkls):
            try:
                with open(pkl, "rb") as f:
                    _data = pickle.load(f)
                out_pkls_loadable[i] = True
            except Exception as e:
                print(f"Failed to load dynamem output pkl file '{pkl}': {e}")

        if not any(out_pkls_loadable):
            print(f"Failed to load dynamem output pkl files '{out_pkls[0]}' and '{out_pkls[1]}'")
            regular_exit = False

        # get newest loadable pkl by modification time
        newest_pkl = None
        newest_mtime = 0.0
        for i, pkl in enumerate(out_pkls):
            if out_pkls_loadable[i]:
                mtime = pkl.stat().st_mtime
                if mtime > newest_mtime:
                    newest_mtime = mtime
                    newest_pkl = pkl
        if newest_pkl is not None:
            # rename this file to standard output pkl name and delete the other one
            if newest_pkl != standard_pkl:
                shutil.move(newest_pkl, standard_pkl)
                print(f"Renamed dynamem output pkl file '{newest_pkl}' to standard output name '{standard_pkl}'")
            for pkl in out_pkls:
                if pkl != newest_pkl and pkl.exists():
                    pkl.unlink()
                    print(f"Deleted unused dynamem output pkl file '{pkl}'")
    
    if app != "genmap" and path_length < 0.4:
        print(f"Robot did not move enough (path length {path_length:.3f}m < 0.4m). Likely something went wrong.")
        regular_exit = False

    return regular_exit


def main():
    parser = argparse.ArgumentParser(description="Launch multiple helper processes and stop after an optional timeout.")
    parser.add_argument(
        "--experiment-json",
        type=Path,
        help="Path to experiment JSON file.",
        action="append",
    )
    parser.add_argument(
        "--app",
        type=str,
        choices=["dynamem", "perceivesemantix", "random", "genmap"],
        nargs="+",
        help="One or more apps to run (e.g. --app dynamem perceivesemantix)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        help="Root output folder for experiment results.",
        default=Path("/home/benni/datasets/sim_results"),
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name for the experiment run.",
        default=None,
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        help="Maximum number of attempts to run the experiment.",
        default=2,
    )
    args = parser.parse_args()

    for expirment_config in args.experiment_json:
        experiments: dict = json.loads(expirment_config.read_text())
        for experiment in experiments["experiments"]:
            if args.name is not None and experiment["name"] != args.name:
                continue
            for app in args.app:
                if experiment["goal"]["task"] == "explore" and app.lower() == "random":
                    print("Skipping 'random' app for exploration task.")
                    continue

                overwrite = False
                for _attempt in range(args.max_attempts):
                    regular_exit = run_expriment(app, experiment, args.out_root, overwrite=overwrite)
                    if regular_exit:
                        break
                    else:
                        print(f"Experiment '{experiment['name']}' with app '{app}' did not exit regularly.")
                        overwrite = True


if __name__ == "__main__":
    main()
