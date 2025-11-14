# stretch_isaac

NVIDIA Isaac Sim 4.5.0 environment for ROS 2 testing of the SE3 Hello Robot.


## To install Isaac Sim with Pixi do

- `pixi shell` (in the root diretory of this repo)
- launch by executing `isaacsim`

## The stretch model

- open or import `importable_stretch.usd` into your IsaacSim stage

## Interior scenes

- the file `interior_agent_scene.usd` contains the scene (including imported stretch) `kujale_0003` from the dataset https://huggingface.co/datasets/spatialverse/InteriorAgent/tree/main

- the file `hm3d_scene.usd` contains a scene (including imported stretch) from the [Habitat-Matterport3D](https://github.com/matterport/habitat-matterport-3dresearch?tab=readme-ov-file) dataset

## TODO

- integrade ROS functionality into propely set-up stretch model from https://github.com/hello-robot/stretch_isaacsim

---
---

# Outdated instructions

## Prerequisites

- Omniverse Isaac Sim 4.5.0  
- ROS 2 (e.g. Galactic or Humble) installed and sourced  
- Python 3.10 
- NVIDIA GPU with up-to-date drivers  
- `ros2-bridge` plugin enabled in Isaac Sim  

## Directory Layout

- `Robot_Import_Files/` – modified URDF with updated collision meshes  
- `SE3_ROS2.usd/` – Issac Sim USD stage with the imported robot
- `stretch_refrence.usd/` - Refrence ready USD, for import into environments.
- `README.md`         – this file

## Import Process

Adapted from the Isaac Sim docs:  
- https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot_setup/import_urdf.html  
- https://docs.isaacsim.omniverse.nvidia.com/4.5.0/ros2_tutorials/index.html  

1. **Create a new Isaac Sim project**  
2. **Import the URDF** (File > Import)  
   - Original URDF used square collision meshes on the wheels, which caused physics artifacts.  
   - Replaced them with cylinders; see `Robot_Import_Files/`.  
   - Enabled self-collision and set the base link movable.  
3. **Tune joint dynamics**  
   - **Wheels**  
     - Armature: 2.0 kg·m² (reduces jitter)  
     - Damping: 1000; Stiffness: 0  
     - Clamped max torque and brake force  
   - **Positional joints**  
     - Armature: 0.1 kg·m²  
     - Damping & stiffness hand-tuned via GUI  
       (Tools > Robotics > Asset Editors > Gain Tuner)  
4. **ROS 2 Bridge configuration** (synchronized to system time)  
   - Adapt or reuse OmniGraph templates from  
     Tools > Robotics > ROS 2 OmniGraphs  
   - Key graphs:  
     1. Camera broadcast  
     2. TF broadcast  
     3. Differential controller  
     4. Joint state publisher/subscriber  

## Launching the Simulation
1. Launch isaac sim with `./run_isaac.sh`
2. Select an environment or create a scene for exploration in the robot editor.
   - https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_environments.html
4. Add the stretch as a refrence to the scene `file->import reference`
5. Play simulation and run ROS2 nodes

## Future Work
- Camera intrinsics investigaiton
     - Problems are arrising in a distortion of the pc when using the rgb-d data. Suspect that there is a scalling on the camera intrinsics data set in isaac sim.
- Code for programically controlled environmental changes
     - An example of how this could be implimented is seen in env_man_script.py
     - This could be extended to automate evaluation using geometry found in the USD    
