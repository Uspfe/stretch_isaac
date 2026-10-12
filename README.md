# IsaacSim setup for the Hello Robot Stretch 3

Files to simulate Hello Robot's [Stretch 3](https://hello-robot.com/stretch-3-product) in [IsaacSim](https://developer.nvidia.com/isaac/sim) with ROS2 integration.

<p align="center">
  <img src=".img/default.png" alt="alt text" height="400">
  <img src=".img/kitchen.png" alt="alt text" height="400">
</p>

## Prerequisites

- NVIDIA GPU with up-to-date drivers
- [Pixi](https://pixi.prefix.dev/latest/installation/)


## Quick Start
This repository provides an Isaac Sim environment for deploying and testing the Hello Robot Stretch with ROS 2.
A minimal end-to-end workflow is as follows:
1. Install [Pixi](https://pixi.prefix.dev/latest/installation/)
2. Activate/install the Pixi environment (this installs IsaacSim)
    ```bash
      pixi shell
    ```
3. Start IsaacSim (or use the [scripted start below](#simulation-script))
    ```bash
      isaacsim
    ```
3. Open the [provided example scene](stretch_isaac/example_stage/hm3d_example.usd) from the Habitat Matterport 3D
4. Import the [stretch3_no_arm_collider.usd](robot_usd/stretch3_no_arm_collider.usd) (make sure it is imported below the root prim "/map". See [here](#manual-robot-import)). Only this asset has the [Nav2 ROS interface](#ros-interface-nav2); `stretch3.usd` still carries the older MPC-era graphs.
5. Play the simulation.
6. Test ROS2 integration (Seperate terminal. To run these you need a ROS2 installation, which is not included in the minimal ROS2 install included with IsaacSim.)
    - `ros2 topic echo /tf`
    - `ros2 topic pub /stretch/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}}"`

This README explains each step in more detail below.

## Manual robot import

If your scene does not already include the robot:
  - Import `robot_usd/stretch3_no_arm_collider.usd` into the current stage
  - Use **File > Import Reference** or drag-and-drop it via the content browser
  - In the scene graph, make sure to import the robot below the root prim `/map` as the name of this prim is used as the global reference frame when publishing to ROS
  ![scene tree](.img/scene_tree.png)

## Simulation script

- `pixi run python standalone_sim.py` to launch IsaacSim loading an empty scene (only ground plane)
- `pixi run python standalone_sim.py --scene ./stretch_isaac/example_stage/hm3d_example.usd` to load the provided example scene
- `pixi run python standalone_sim.py --scene ... --occupancy-map map.npz` additionally writes the scene's ground-truth occupancy grid (free = floor without obstacle, unknown = no floor), which stretch_mpc's `static_map_publisher` serves as `/map`
- `pixi run python standalone_sim.py --help` to check available options

## ROS interface (Nav2)

The robot's action graphs mirror the real robot under Nav2 (REP 105), with every stamp on simulation time:

| Interface | Notes |
|---|---|
| `/clock` | simulation time; run every ROS node with `use_sim_time:=true` |
| tf `map -> odom` | identity, on `/tf` every step, standing in for the SLAM stack |
| tf `odom -> base_link`, `/odom` | ground-truth pose; the twist is `base_link`'s body-frame velocity, like wheel odometry |
| `/scan_filtered` | RTX lidar in frame `laser` |
| `/femtobolt/color/image_raw`, `/femtobolt/color/camera_info`, `/femtobolt/depth/image_raw` | head camera; depth is `32FC1` in metres (perceive_semantix needs `camera_depth_scale_to_m:=1.0`) |
| `/joint_states`, `/joint_command` | head pan and tilt |
| `/stretch/cmd_vel` | drives the base |
| `/home_the_robot`, `/stow_the_robot`, `/navigation_control/move_camera_oscillate`, `/is_homed` | stand-ins that succeed immediately; launch Nav2 with `camera_oscillator:=false` |

`update_ros_graphs.py` applies this interface to a robot USD in place (`pixi run python update_ros_graphs.py robot_usd/<asset>.usd`); it is how `stretch3_no_arm_collider.usd` was made and can bring `stretch3.usd` up to date.

Isaac Sim's ROS bridge ships only Fast DDS and Cyclone DDS, not zenoh. To keep the traffic on this machine, give every process (Isaac Sim and all ROS nodes talking to it) `RMW_IMPLEMENTATION=rmw_fastrtps_cpp FASTRTPS_DEFAULT_PROFILES_FILE=<this repo>/config/fastdds_localhost.xml` and a common `ROS_DOMAIN_ID`. Do not set `ROS_AUTOMATIC_DISCOVERY_RANGE`: rmw_fastrtps then opens sockets on every network interface again.

## Behaviour tests

`behaviour_test.py` runs the robot stack against Isaac Sim and checks its behaviour. It starts every component in its own pixi workspace of the mono-repo (Isaac Sim here, Nav2 from `../stretch_mpc`, `main_coordinator` from `../offline_bringup_active_mapmaintenance`, optionally `../perceive_semantix`) on a private, localhost-only ROS graph with a random domain.

```bash
pixi run -e test behaviour-test list                         # scenarios
pixi run -e test behaviour-test run navigate_across_house    # one scenario, headless
pixi run -e test behaviour-test run all --gui                # all of them, with the Isaac Sim window
pixi run -e test behaviour-test run spin_in_place --keep-running   # leave the stack up afterwards
pixi run -e test behaviour-test stack --coordinator          # only bring the stack up, for debugging
```

Each scenario gets a fresh stack and writes `test_runs/<time>_<scenario>/`: the log of every process, the ground-truth trajectory (`trajectory.csv`), the verdict and metrics (`result.json`) and `attach.sh`, which runs a command on that run's ROS graph (e.g. `./attach.sh ros2 topic list`). Without perceive_semantix, `/map` is the scene's ground-truth occupancy grid. Scenarios that need perceive_semantix run it without `OPENAI_API_KEY`, so they send nothing to OpenAI.

> API Documentation of [IsaacSim](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.api/docs/index.html)


## Directory Layout

- `robot_usd/` - ready made robot usd for import into other scenes (ROS enabled)
- `example_scene_hm3d.usd/` - Issac Sim USD stage with the imported robot (Habitat Matterport 3D dataset)
- `Robot_Import_Files/` - modified URDF with updated collision meshes
- `README.md` - this file
- `standalone_sim.py` - python script for [scripted simulation ](#simulation-script)
- `update_ros_graphs.py` - applies the [Nav2 ROS interface](#ros-interface-nav2) to a robot USD
- `behaviour_test.py` - [behaviour tests](#behaviour-tests) of the robot stack; results go to `test_runs/` (not tracked)
- `config/fastdds_localhost.xml` - Fast DDS profile that keeps ROS traffic on this machine

## ros2-bridge

- `ros2-bridge` plugin enabled in Isaac Sim 
  - It is already automatically enabled. 
  - Go to Window > Extensions, find "ROS 2 Bridge," and verify it is **Enabled**.

## Recreating the ready-made robot models

### Import Process

Adapted from the Isaac Sim docs:  
- https://docs.isaacsim.omniverse.nvidia.com/5.1.0/importer_exporter/importers_exporters.html 
- https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/ros2_landing_page.html

1. **Create or Open an Isaac Sim Scene**  
   You may either open an existing prepared scene or create your own.
   > **DON'T WORRY** 
   >
   >It may take a **long time during the first launch**. (roughly 15 min for work stations and 20 min for laptops)

   - InteriorAgent Dataset: https://huggingface.co/datasets/spatialverse/InteriorAgent/tree/main

   - `example_scene_hm3d.usd`
      - Contains an HM3D environment with Stretch already imported  
       Dataset: https://github.com/matterport/habitat-matterport-3dresearch

    > **Note:**  
    > If you try to open either of these two scenes, make sure you have downloaded the corresponding datasets and that Isaac Sim can locate them on your local system.  
    >  
    > After opening one of these scenes, you may skip **Step 2**.

   You may also use built-in Isaac Sim assets:
   - Navigate to **Content > Isaac Sim** to browse default environments and props.

   **Physics note:**  
   For objects to interact physically with the robot:

   - Select the object in the **Stage** window  
   - Right-click → **Add > Physics > Rigid Body**  
   - Add a **Collider Preset**

2. **Import the Stretch as USD File** 
  If your scene does not already include the robot:
    - Import `robot_usd/stretch3.usd` into the current stage
    - Use **File > Import Reference** so the robot remains reusable
    - In the scene graph, make sure to import the robot below the root prim `/map` as the name of this prim is used as the global reference frame when publishing to ROS

    Model details:
    - Original URDF used square collision meshes on the wheels, which caused physics artifacts.  
    - Replace them with cylinders; see `Robot_Import_Files/`.  
    - Enable self-collision and set the base link movable.
  
3. **Tune joint dynamics**
  Proper joint tuning is critical for stable simulation.

  - **Wheels** 
    - Joints: `joint_right_wheel` and `joint_left_wheel`
    - Recommended parameters
      - Armature: 2.0 kg·m² (reduces jitter)  
      - Damping: 1000
      - Stiffness: 0  
      - Max torque and break force clamped

    > Where to set this in the UI:
    > - Select the wheel joint in the Stage window
    > - Open the Property panel
    > - Navigate to **Physics > Articulation > Drive**

  - **Positional joints (arm, lift, wrist)**  
    - Armature: 0.1 kg·m²  
    - Damping & stiffness hand-tuned via GUI  
      - **Tools > Robotics > Asset Editors > Gain Tuner**

4. **ROS 2 Bridge configuration**
  (synchronized to system time)  
    - Adapt or reuse OmniGraph templates from **Window > Graph Editors > Action Graph**

      <img src=".img/action_graph.png" width="50%">

    - **ROS2 Topic Overview**
    
      | Component | Topics                           | Direction | Purpose                     |
      | :--------- | :-------------------------------- | :--------- | :--------------------------- |
      | Base      | `/stretch/cmd_vel`               | Sub       | Differential drive control  |
      | Joints    | `/joint_command`, `/joint_state` | Sub / Pub | Joint commands and feedback |
      | Camera    | `/spectacular_ai/*`              | Pub       | RGB, depth, point cloud     |
      | Lidar     | `/scan_filtered`                 | Pub       | Laser scan                  |
      | TF        | `/tf`, `/tf_static`              | Pub       | Coordinate transforms       |
      | State     | `/state_estimator/pose_filtered` | Pub       | Estimated robot pose        |
      | Homing    | `/is_homed`                      | Pub / Srv | Robot homing status         |

### Launching the Simulation
1. Enter the Pixi environment in the root diretory of this repo:
    ```bash
    pixi shell
    ```
2. Launch Isaac Sim
    ```bash
    isaacsim
    ```
3. Open a scene or import the Stretch USD.
4. Press **Play** to start the simulation.
5. Run ROS2 nodes in a separate terminal.

### Testing

1. **Base Motion Control (`/stretch/cmd_vel`)**
  - This command controls the differential drive of the robot base.
      ```bash
      ros2 topic pub --once /stretch/cmd_vel \
      geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}"
      ```
    > Expected behavior:
    > - The robot rotates in place.
    > 
    > If the robot does not move:
    > - Check wheel material
    > - Check wheel joint drive settings
    > - Verify ground plane has a collider
    > - Ensure simulation is playing

2. **Joint-Level Control (`/joint_command`)**
  - This command controls individual articulated joints (arm, lift, wrist).
    ```bash
    ros2 topic pub /joint_command \
    sensor_msgs/JointState "{name: ['joint_lift'], position: [0.2]}"
    ```

  - Verify feedback:
    ```bash
    ros2 topic echo /joint_state
    ```
    > Expected behavior:
    > - The joint moves to the commanded position.
    > - `/joint_state` reflects the correct value.
    > 
    > If the joints do not move:
    > - Check the lower/ upper limit or max force value of the joints.
    > - Recheck armature, damping, and stiffness values.

    <img src=".img/joint_drive.png" width="50%">
