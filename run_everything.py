from enum import Enum
import subprocess
import threading
import sys

class OutMode(Enum):
    CONSOLE = 0
    DISABLED = 1

COLORS = {
    "red":    "\033[31m",
    "green":  "\033[32m",
    "yellow": "\033[33m",
    "blue":   "\033[34m",
    "magenta":"\033[35m",
    "cyan":   "\033[36m",
    "reset":  "\033[0m",
}

def forward_output_and_handle_input(proc, name: str, color: str, triggers: dict[str, str], mode: OutMode = OutMode.CONSOLE):
    if mode == OutMode.CONSOLE:
        prefix = f"{color}[{name}]{COLORS['reset']} "
        for line in proc.stdout:
            sys.stdout.write(prefix + line)
            # Check triggers
            for pattern, response in triggers.items():
                if pattern in line:
                    proc.stdin.write(response)   # send input to process
                    proc.stdin.flush()
        proc.stdout.close()

def main():
    # List of processes to launch
    processes = [
        {
            "name": "IsaacSim",
            "cmd": [
                "pixi",
                "run",
                "python",
                "standalone_sim.py",
            ],
            "cwd": "/home/benni/repos/benni_stretch_issac/",
            "color": COLORS["red"],
            "triggers": {},
            "output": OutMode.DISABLED,
        },
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
            "cwd": "/home/benni/bringup_active_mapmaintenance/perceive_semantix/",
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
            "cwd": "/home/benni/bringup_active_mapmaintenance/stretch_mpc/",
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
            "cwd": "/home/benni/bringup_active_mapmaintenance/offline_bringup_active_mapmaintenance/",
            "color": COLORS["green"],
            "triggers": {"What would you like to do? (type 'maintain', 'explore', 'random', 'patrol', 'home' or type anything else to search a specific object)": "explore\n"},
            "output": OutMode.CONSOLE,
        },
    ]

    running = []

    for p in processes:
        proc = subprocess.Popen(
            p["cmd"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=p["cwd"],
            text=True,
        )

        # Launch thread to read output
        t = threading.Thread(target=forward_output_and_handle_input, args=(proc, p["name"], p["color"], p["triggers"], p["output"]))
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
