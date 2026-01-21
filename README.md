# IsaacSim Evaluation Suite for Object-Goal Navigation in Semi-Static Scenes

Contains specifications as well as code to load the 60 tasks object-goal navigation tasks used in the paper ["Where Did I Leave My Glasses? Open-Vocabulary Semantic Exploration in Real-World Semi-Static Environments"](https://utiasdsl.github.io/semi-static-semantic-exploration/).

## Prerequisites

- NVIDIA GPU with up-to-date drivers (for IsaacSim)
- Installed [Pixi](https://pixi.prefix.dev/latest/installation/)
- Downloaded [InteriorAgent](https://huggingface.co/datasets/spatialverse/InteriorAgent) dataset in directory "$InteriorAgentRoot"


## Quick Start

This repository uses a [Pixi](https://pixi.prefix.dev/latest/installation/) environment which includes a IsaacSim installation.

A minimal end-to-end workflow is as follows:
1. Install [Pixi](https://pixi.prefix.dev/latest/installation/)
2. Activate/install the Pixi environment (this installs IsaacSim)
    ```bash
      pixi shell
    ```
3. Check that IsaacSim is working by running
    ```bash
      isaacsim
    ```

## Loading experiments

[experiments.json](experiments.json) is setup as
```json
{
  "experiments": [
    {
      "name": "kujiale_0020_explore_moved",                    // unique name of the experiment   
      "scene": "InteriorAgent/kujiale_0020/kujiale_0020.usda", // path to the scene file   
      "goal": {                                                // what kind of task the robot has ("explore" or "search")   
          "task": "explore"                                       
      },                                                       
      "max_runtime": 900.0,                                    // maximum runtime of this experiment   
      "remove_assets": [                                       // assets to remove from this scene (by substring match, i.e., here all prims with substring "bottle" are removed)   
        "bottle",
      ],                                                          
      "exclude_remove_assets": [                               // assets to NOT remove (by substring match)   
        "bottle_0012"                               
      ],                                                          
      "robot_start": { "position": [ -0.6, 0.0, 0.0 ] }        // starting position of the robot   
    },                                                              

    {                                                              
      "name": "kujiale_0020_bottle_moved",                        
      "scene": "InteriorAgent/kujiale_0020/kujiale_0020.usda",    
      "initialmap_experiment": "kujiale_0020_explore_moved",   // if provided, the map generated during this experiment should be pre-loaded
      "goal": {                                                   
          "task": "search",                                    // search tasks provide some more task information
          "label": "bottle",                                   // the label the robot is instructed to search for
          "asset": "bottle_0010",                              // the specific goal asset(s) (by substring match), the simulation script returns the position(s) of the goal asset(s) for evaluation    
          "prior_map_object": "bottle_0012"                    // the corresponding goal asset which was present in the "initialmap_experiment", useful for evaluation
      },                                                          
      "max_runtime": 300.0,                                       
      "remove_assets": [                                          
        "ornament",                                             
        "bottle",                                               
      ],                                                          
      "exclude_remove_assets": [                                  
        "bottle_0010"                                           
      ],                                                          
      "robot_start": { "position": [-0.6, 0.0, 0.0] }             
    },                                                               
    ...
  ]
}
```

Some of these properties should be given to the robot controller, such as `initialmap_experiment`, some to the simulation script `standalone_sim.py`. For example, `kujiale_0020_bottle_moved` is loaded with
```bash
pixi run python standalone_sim.py \
  --scene $InteriorAgentRoot/kujiale_0020/kujiale_0020.usda \
  --robot-start -0.6 0.0 0.0 0.0 \
  --gasset bottle_0010 \
  --rasset ornament bottle \
  --rasset-exclude bottle_0010
```

This script starts IsaacSim and periodically prints information about the goal asset and the robot state:

```
<goals>{"bottle_0010": {"x": -3.59, "y": 5.19, "z": 0.95}, "shortest_distance": 6.44}</goals>
<robot>{"time": 6.08333365060389, "position": {"x": -0.6012449264526367, "y": -8.089374750852585e-05, "z": 0.04939977824687958}, "orientation": {"w": 0.9999939203262329, "x": -0.0004203889984637499, "y": 0.0012234784662723541, "z": 0.0032800277695059776}, "linear_velocity": {"vx": -0.012691676616668701, "vy": -0.003575022565200925, "vz": 0.06355907768011093}}</robot>
```

The `shortest_distance` is the length of the shortest path between robot start position and the closest given goal asset. Before simulation is started a 2D occupancy map of the scene is generated based on which the shortest path is computed. The result can be visualized by providing `--visualize-shortest-path`:

![Example occupancy map and shortest path](.img/occ_map.png)

For more information check
```bash
pixi run python standalone_sim.py -h
```