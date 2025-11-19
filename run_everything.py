import argparse
import os
import pty
import signal
import subprocess
import sys
import threading
import time
from enum import Enum
from typing import Literal, Optional, Union


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


class ProcessHandler:
    def __init__(
        self,
        proc,
        master_fd,
        name: str,
        color: str,
        triggers: dict[Union[str, float], str],
        mode=OutMode.CONSOLE,
    ):
        self.proc = proc
        # master_fd is the integer file descriptor for the PTY master
        self.master_fd = master_fd
        self.name = name
        self.color = color
        self.triggers = triggers
        self.mode = mode

        self.start = time.time()
        self.fired_triggers = set()

    def forward_output_and_handle_input(self):
        # Read raw bytes from the PTY master fd so prompts without newlines are shown
        prefix = f"{self.color}[{self.name}]{COLORS['reset']} "
        at_line_start = True
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

                accum += text
                # Keep accum bounded
                if len(accum) > 8192:
                    accum = accum[-8192:]

                # Print text to stdout, adding prefix at line starts
                if self.mode == OutMode.CONSOLE:
                    parts = text.split("\n")
                    for i, part in enumerate(parts):
                        if at_line_start:
                            sys.stdout.write(prefix)
                        sys.stdout.write(part)
                        if i < len(parts) - 1:
                            sys.stdout.write("\n")
                            at_line_start = True
                        else:
                            at_line_start = False
                    sys.stdout.flush()

                # Check string triggers against the accumulated text
                for pattern, response in self.triggers.items():
                    if (
                        isinstance(pattern, str)
                        and pattern in accum
                        and pattern not in self.fired_triggers
                    ):
                        self.fired_triggers.add(pattern)
                        try:
                            os.write(self.master_fd, response.encode())
                            if self.mode == OutMode.CONSOLE:
                                sys.stdout.write(
                                    f"{prefix}Fired string trigger '{pattern}': {response.strip()}\n"
                                )
                                sys.stdout.flush()
                        except Exception:
                            pass
        finally:
            try:
                os.close(self.master_fd)
            except Exception:
                pass

    def handle_time_triggers(self):
        if self.proc.poll() is not None:
            return  # process has exited
        if getattr(self, "master_fd", None) is None:
            return
        now = time.time() - self.start
        for pattern, response in self.triggers.items():
            if (
                isinstance(pattern, (int, float))
                and pattern <= now
                and pattern not in self.fired_triggers
            ):
                if self.mode == OutMode.CONSOLE:
                    sys.stdout.write(
                        f"{self.color}[{self.name}]{COLORS['reset']} Fired time trigger at {now:.1f}s: {response.strip()}\n"
                    )
                self.fired_triggers.add(pattern)
                try:
                    os.write(self.master_fd, response.encode())
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(
        description="Launch multiple helper processes and stop after an optional timeout."
    )
    parser.add_argument(
        "--max-runtime",
        type=float,
        default=None,
        help="Maximum runtime in seconds after which all launched processes are stopped.",
    )
    parser.add_argument(
        "--app",
        type=str,
        default=None,
        choices=["PerceiveSemantix", "DynaMem"],
        help="Application to run (PerceiveSemantix or DynaMem). If not provided, only IsaacSim is launched.",
    )
    parser.add_argument(
        "--explore",
        action="store_true",
    )
    args = parser.parse_args()
    max_runtime = args.max_runtime
    app: Optional[Literal["PerceiveSemantix", "DynaMem"]] = args.app

    processes = [
        {
            "name": "IsaacSim",
            "cmd": [
                "pixi",
                "run",
                "python",
                "standalone_sim.py",
            ],
            "cwd": "/home/benni/repos/stretch_isaac/",
            "color": COLORS["red"],
            "triggers": {},
            "output": OutMode.DISABLED,
        },
    ]

    if app is None:
        pass
    elif app.lower() == "dynamem":
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
                    "--output-path",
                    "exploration/hm3d-0",
                    "--explore-iter",
                    "40",
                ],
                "cwd": "/home/benni/repos/stretch_ai",
                "color": COLORS["green"],
                "triggers": {
                    "Enter desired mode [E (explore and mapping) / M (Open vocabulary pick and place)]": "E\n"
                },
                "output": OutMode.CONSOLE,
            },
        ]
    elif app.lower() == "perceivesemantix":
        # initial_scene_path = "/home/benni/datasets/sim_results/perceive_semantix/kujiale_0003/exploration/output/1763558507-887001.pkl" if not args.explore else "\"\""
        initial_scene_path = (
            "/home/benni/datasets/sim_results/perceive_semantix/hm3d-0/exploration/output/1763560487-865056.pkl"
            if not args.explore
            else '""'
        )
        triggers = (
            {"What would you like to do? (type 'maintain'": "explore\n"}
            if args.explore
            else {15.0: "sink\n"}
        )
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
                    f"store_output:={str(args.explore)}",
                    "-p",
                    "publishing_rate_background_pointcloud:=0.0",
                    "-p",
                    "objects/point_cloud/publishing_rate:=0.0",
                    "-p",
                    "occupancy_map/publishing_rate:=0.5",
                    "-p",
                    f"initial_scene_path:={initial_scene_path}",
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
            #     "output": OutMode.CONSOLE,
            # },
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
        ]
    else:
        sys.stderr.write(f"Error: Unknown app '{app}'\n")
        sys.exit(1)

    # Verify that each configured working directory exists
    for p in processes:
        cwd = p.get("cwd")
        if not cwd:
            continue
        if not os.path.isdir(cwd):
            sys.stderr.write(
                f"Error: cwd for process '{p.get('name', '<unknown>')}' does not exist: {cwd}\n"
            )
            sys.exit(1)

    running_processes = []
    process_handlers = []
    for p in processes:
        # Create a pseudo-terminal so the subprocess believes it's attached to a terminal
        master_fd, slave_fd = pty.openpty()

        proc = subprocess.Popen(
            p["cmd"],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=p["cwd"],
            preexec_fn=os.setsid,
            close_fds=True,
            text=False,
        )

        # Close the slave fd in the parent; the child has it open.
        try:
            os.close(slave_fd)
        except Exception:
            pass

        # Launch thread to read output from the master fd
        handler = ProcessHandler(
            proc, master_fd, p["name"], p["color"], p["triggers"], p["output"]
        )
        t = threading.Thread(target=handler.forward_output_and_handle_input)
        t.daemon = True
        t.start()

        process_handlers.append(handler)
        running_processes.append(proc)

    def stop_processes(procs):
        sys.stdout.write("Stopping processes...\n")
        for proc in procs:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception as e:
                sys.stderr.write(f"Error terminating process: {e}\n")

    def timeout_watcher(procs, timeout: float):
        time.sleep(timeout)
        sys.stdout.write(
            f"Maximum runtime of {timeout} seconds reached. Terminating processes.\n"
        )
        stop_processes(procs)

    # If max runtime was provided, start watcher thread
    if max_runtime is not None:
        w = threading.Thread(
            target=timeout_watcher, args=(running_processes, max_runtime)
        )
        w.daemon = True
        w.start()

    # Wait for all to finish
    try:
        while any([proc.poll() is None for proc in running_processes]):
            time.sleep(0.1)
            for handler in process_handlers:
                handler.handle_time_triggers()

    except KeyboardInterrupt:
        stop_processes(running_processes)


if __name__ == "__main__":
    main()
