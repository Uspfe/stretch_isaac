import argparse
import json
import os
from pathlib import Path
import pty
import signal
import subprocess
import sys
import threading
import time
from enum import Enum
from typing import Any, Literal, Optional, Union

import numpy as np


class OutMode(Enum):
    CONSOLE = 0
    DISABLED = 1


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


def parse_sim_state(text: str):
    """Try to deserialize JSON from text; return state tuple or None if invalid."""
    global STATE_LOCK, STATE_LIST
    try:
        data = json.loads(text)
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
        with STATE_LOCK:
            STATE_LIST.append((time, position, orientation, linear_velocity))


def print_line(line: str, prefix: str = ""):
    sys.stdout.write(f"{prefix}{line.strip()}\n")


def success_monitor(goal_position: np.ndarray, success_distance_threshold: float):
    global SUCCESS_FEEDBACK_LOCK, SUCCESS_FEEDBACK
    success_reported = False
    with SUCCESS_FEEDBACK_LOCK:
        if "SUCCESS" in SUCCESS_FEEDBACK:
            success_reported = True

    with STATE_LOCK:
        if STATE_LIST:
            last_state = STATE_LIST[-1]
            position = np.array(last_state[1][:2])  # x, y
            distance = np.linalg.norm(position - goal_position)
        else:
            distance = float("inf")

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
                        if "SUCCESS" in response:
                            global SUCCESS_FEEDBACK_LOCK, SUCCESS_FEEDBACK
                            with SUCCESS_FEEDBACK_LOCK:
                                SUCCESS_FEEDBACK = "SUCCESS"
                            if self.mode == OutMode.CONSOLE:
                                sys.stdout.write(f"{self.prefix}SUCCESS condition detected!\n")
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
    names = [p.get("name") for p in processes]
    if "DynaMem" in names:
        subprocess.run(["rm", "-r", "/home/benni/repos/stretch_ai/.pixi"], check=False)
        print("Removed .pixi directory before launching DynaMem.")

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


def store_results(
    record: str,
    app: str,
    output_file: Path,
    experiment: dict,
    output_root: Path,
    state_trajectory: list,
    success: bool,
):
    state_file = output_root / f"{record}_state_trajectory.npy"

    try:
        with open(output_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []

    # Convert list of tuples to a 2D NumPy array
    arr = np.array(
        [[time] + pos + ori + vel for time, pos, ori, vel in state_trajectory],
        dtype=float,
    )
    path_length = np.linalg.norm(np.diff(arr[:, 1:3], axis=0), axis=1).sum()
    time_to_complete = arr[-1, 0] - arr[0, 0] if len(arr) > 1 else 0.0

    new_record = {
        "name": record,
        "app": app,
        "experiment": experiment,
        "state_trajectory_file": state_file.resolve().absolute().as_posix(),
        "time_to_complete": time_to_complete,
        "path_length": path_length,
        "success": success,
    }

    if not any(d.get("name") == new_record.get("name") for d in data):
        data.append(new_record)
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        np.save(state_file, arr)
    else:
        print(f"Experiment record '{record}' already exists in results file.")


def build_proccesses(
    app: Literal["dynamem", "perceivesemantix"], experiment: dict, output_root: Path
) -> list[dict[str, Any]]:
    if app.lower() not in ["dynamem", "perceivesemantix"]:
        raise ValueError(f"Unsupported app: {app}")

    processes = [
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
            ],
            "cwd": "/home/benni/repos/stretch_isaac/",
            "color": COLORS["red"],
            "triggers": {},
            "output": OutMode.DISABLED,
            "line_handlers": [parse_sim_state],
        },
    ]

    output_dir = output_root / app.lower() / Path(experiment.get("scene")).stem / experiment["name"]
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    do_explore = experiment["goal"]["task"] == "explore"
    if not do_explore:
        input_path = (
            output_root / app.lower() / Path(experiment.get("scene")).stem / experiment["initialmap_experiment"]
        )
        if app.lower() == "dynamem":
            input_path = input_path.with_suffix(".pkl")
        elif app.lower() == "perceivesemantix":
            input_path = input_path / "output"
            input_file = latest_pkl(input_path)
            if input_file is None:
                raise FileNotFoundError(f"No exploration pkl files found in {input_path}")
            input_path = input_file
        input_path = input_path.resolve()
    else:
        input_path = None

    if app.lower() == "dynamem":
        if do_explore:
            dynamem_log = Path("/home/benni/repos/stretch_ai/dynamem_log")
            rel_out_dir = Path(os.path.relpath(output_dir, dynamem_log))
            options = [
                "--output-path",
                str(rel_out_dir),
                "--explore-iter",
                "10",
            ]
            # in exploration mode the map is not saved, so instead we search for an object (volcano) which is never present
            triggers = {
                "Enter desired mode [E (explore and mapping) / M (Open vocabulary pick and place)]": "M\n",
                "Enter the target object:": "volcano\n",
                "Enter the target receptacle:": "volcano\n",
                "Do you want to run navigation? [Y/n]:": "Y\n",
                "Do you want to run picking? [Y/n]:": "n\n",
                "Do you want to run placement? [Y/n]:": "n\n",
            }
        else:
            options = [
                "--input-path",
                str(input_path),
            ]
            triggers = {
                "Enter desired mode [E (explore and mapping) / M (Open vocabulary pick and place)]": "M\n",
                "Enter the target object:": f"{experiment['goal']['label']}\n",
                "Enter the target receptacle:": "table\n",
                "Do you want to run navigation? [Y/n]:": "Y\n",
                "Do you want to run picking? [Y/n]:": "SUCCESS\n",
                "Do you want to run placement? [Y/n]:": "n\n",
            }
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
    elif app.lower() == "perceivesemantix":
        initial_scene_path = str(input_path) if not do_explore else '""'
        triggers = {15.0: "explore\n"} if do_explore else {15.0: f"{experiment['goal']['label']}\n"}
        triggers[" found at "] = "SUCCESS\n"
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
                    "occupancy_map/floor_height:=0.05",
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
                "output": OutMode.CONSOLE,
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
    return processes


def run_expriment(app: Literal["dynamem", "perceivesemantix"], experiment: dict, output_root: Path):
    processes = build_proccesses(app, experiment, output_root)

    record_key = f"{experiment['name']}_{app.lower()}"
    output_file = output_root / "experiments_results.json"
    if check_existing_record(record_key, output_file):
        print(f"Experiment record '{record_key}' already exists. Skipping experiment.")
        return

    running_processes, process_handlers = launch_processes(processes)
    max_runtime: Optional[float] = experiment.get("max_runtime", None)

    def timeout_watcher(procs, timeout: float):
        time.sleep(timeout)
        sys.stdout.write(f"Maximum runtime of {timeout} seconds reached. Terminating processes.\n")
        terminate_processes(procs)

    # If max runtime was provided, start watcher thread
    if max_runtime is not None:
        w = threading.Thread(target=timeout_watcher, args=(running_processes, max_runtime))
        w.daemon = True
        w.start()

    do_explore = experiment["goal"]["task"] == "explore"
    if not do_explore:
        goal_position = np.array([experiment["goal"]["position"][0], experiment["goal"]["position"][1]])
        success = False
    else:
        success = True  # exploration always "succeeds"

    # Wait for all to finish
    try:
        while any([proc.poll() is None for proc in running_processes]):
            time.sleep(0.1)
            for handler in process_handlers:
                handler.handle_time_triggers()
            if not do_explore and success_monitor(
                goal_position,
                2.0,
            ):
                success = True
                sys.stdout.write("Success condition met. Terminating processes.\n")
                break
    except KeyboardInterrupt:
        print("\nStopping processes...")
    finally:
        terminate_processes(running_processes, timeout=2)
        print("All processes terminated.")

    # Store results
    global STATE_LOCK, STATE_LIST
    with STATE_LOCK:
        state_trajectory = STATE_LIST.copy()
        STATE_LIST.clear()

    global SUCCESS_FEEDBACK_LOCK, SUCCESS_FEEDBACK
    with SUCCESS_FEEDBACK_LOCK:
        SUCCESS_FEEDBACK = SUCCESS_FEEDBACK_DEFAULT

    store_results(record_key, app, output_file, experiment, output_root, state_trajectory, success)


def main():
    parser = argparse.ArgumentParser(description="Launch multiple helper processes and stop after an optional timeout.")
    parser.add_argument(
        "--experiment-json",
        type=Path,
        help="Path to experiment JSON file.",
    )
    parser.add_argument(
        "--app",
        type=str,
        choices=["dynamem", "perceivesemantix"],
        nargs="+",
        help="One or more apps to run (e.g. --app dynamem perceivesemantix)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        help="Root output folder for experiment results.",
        default=Path("/home/benni/datasets/sim_results"),
    )
    args = parser.parse_args()

    experiments: dict = json.loads(args.experiment_json.read_text())

    for experiment in experiments["experiments"]:
        for app in args.app:
            run_expriment(app, experiment, args.out_root)


if __name__ == "__main__":
    main()
