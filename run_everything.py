import os
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
        name: str,
        color: str,
        triggers: dict[Union[str, float], str],
        mode=OutMode.CONSOLE,
    ):
        self.proc = proc
        self.name = name
        self.color = color
        self.triggers = triggers
        self.mode = mode

        self.start = time.monotonic()
        self.fired_time_triggers = set()

    def forward_output_and_handle_input(self):
        for line in self.proc.stdout:
            if self.mode == OutMode.CONSOLE:
                prefix = f"{self.color}[{self.name}]{COLORS['reset']} "
                sys.stdout.write(prefix + line)

            # string triggers
            for pattern, response in self.triggers.items():
                if isinstance(pattern, str) and pattern in line:
                    if self.proc.stdin:
                        self.proc.stdin.write(response)
                        self.proc.stdin.flush()

            # time-based triggers
            now = time.monotonic() - self.start
            for pattern, response in self.triggers.items():
                if (
                    isinstance(pattern, (int, float))
                    and pattern <= now
                    and pattern not in self.fired_time_triggers
                ):
                    self.fired_time_triggers.add(pattern)
                    if self.proc.stdin:
                        self.proc.stdin.write(response)
                        self.proc.stdin.flush()

        self.proc.stdout.close()


def main():
    # List of processes to launch
    app: Optional[Literal["PerceiveSemantix", "DynaMem"]] = None

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
                "output": OutMode.CONSOLE,
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
                ],
                "cwd": "/home/benni/repos/stretch_ai",
                "color": COLORS["green"],
                "triggers": {},
                "output": OutMode.CONSOLE,
            },
        ]
    elif app.lower() == "perceivesemantix":
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
                    "store_output:=False",
                    "-p",
                    "publishing_rate_background_pointcloud:=0.0",
                    "-p",
                    "objects/point_cloud/publishing_rate:=0.0",
                ],
                "cwd": "/home/benni/repos/bringup_active_mapmaintenance/perceive_semantix/",
                "color": COLORS["blue"],
                "triggers": {},
                "output": OutMode.CONSOLE,
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
                "triggers": {10.0: "explore\n"},
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

    running = []

    for p in processes:
        proc = subprocess.Popen(
            p["cmd"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if p["triggers"] else None,
            cwd=p["cwd"],
            text=True,
        )

        # Launch thread to read output
        handler = ProcessHandler(
            proc, p["name"], p["color"], p["triggers"], p["output"]
        )
        t = threading.Thread(target=handler.forward_output_and_handle_input)
        t.daemon = True
        t.start()

        running.append(proc)

    # Wait for all to finish
    try:
        for proc in running:
            proc.wait()
    except KeyboardInterrupt:
        for proc in running:
            proc.terminate()


if __name__ == "__main__":
    main()
