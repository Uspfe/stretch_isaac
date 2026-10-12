"""Brings the ROS action graphs of a Stretch robot USD in line with the Nav2 interface and saves it in place.

Usage: pixi run python update_ros_graphs.py robot_usd/stretch3_no_arm_collider.usd

- Every stamp is simulation time, and /clock is published, so ROS nodes run with use_sim_time.
- tf follows REP 105: the ground-truth world pose of base_link is published as odom -> base_link together with /odom
  (its twist is the body-frame velocity of base_link, like the wheel odometry of the real robot),
  and map -> odom is the identity, published on /tf with every simulation step like the SLAM stack it stands in
  for (perceive_semantix only accepts camera frames with a map -> * sample at their stamp).
- The camera publishes under /femtobolt/*, the layout of the real robot's SLAM stacks.
- /home_the_robot, /stow_the_robot and /navigation_control/move_camera_oscillate are answered with success by a
  single graph (they were served twice before, once without success).
- The MPC-era BasePose graph (/state_estimator/pose_filtered) is removed; /odom and tf carry the pose.
"""

import argparse
import os
import traceback
from pathlib import Path

from isaacsim import SimulationApp

ACTION_GRAPHS = "/Root/stretch/Action_graphs"
BASE_LINK = "/Root/stretch/base_link"

CAMERA_TOPICS = {
    "CameraInfoPublish": "/femtobolt/color/camera_info",
    "RGBPublish": "/femtobolt/color/image_raw",
    "DepthPublish": "/femtobolt/depth/image_raw",
    "PointCloudPublish": "/femtobolt/point_cloud",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("usd", type=Path, help="Robot USD to update in place.")
    return parser.parse_args()


args = parse_args()
app = SimulationApp({"headless": True})

import omni.graph.core as og  # pyright: ignore[reportMissingImports] # noqa: E402
import omni.usd  # pyright: ignore[reportMissingImports] # noqa: E402
from isaacsim.core.utils import extensions  # pyright: ignore[reportMissingImports] # noqa: E402
from pxr import Sdf, Usd  # pyright: ignore[reportMissingImports] # noqa: E402

keys = og.Controller.Keys


def graph_path(name: str) -> str:
    return f"{ACTION_GRAPHS}/{name}"


def delete_if_present(stage: Usd.Stage, paths: list[str]) -> None:
    for path in paths:
        if stage.GetPrimAtPath(path).IsValid():
            stage.RemovePrim(path)


def set_target(stage: Usd.Stage, attribute_path: str, target: str) -> None:
    node_path, attribute_name = attribute_path.split(".")
    stage.GetPrimAtPath(node_path).GetRelationship(attribute_name).SetTargets([Sdf.Path(target)])


def stamp_with_simulation_time(graph_name: str, consumers: list[str]) -> None:
    """Replaces the graph's system time source by simulation time for the given timeStamp inputs."""
    stage = omni.usd.get_context().get_stage()
    graph = graph_path(graph_name)
    og.Controller.edit(graph, {keys.DISCONNECT_ALL: [f"{graph}/{consumer}.inputs:timeStamp" for consumer in consumers]})
    delete_if_present(stage, [f"{graph}/isaac_read_system_time"])
    if not stage.GetPrimAtPath(f"{graph}/simulation_time").IsValid():
        og.Controller.edit(
            graph, {keys.CREATE_NODES: [("simulation_time", "isaacsim.core.nodes.IsaacReadSimulationTime")]}
        )
    og.Controller.edit(
        graph,
        {
            keys.CONNECT: [
                (f"{graph}/simulation_time.outputs:simulationTime", f"{graph}/{consumer}.inputs:timeStamp")
                for consumer in consumers
            ]
        },
    )


def update_transform_graph(stage: Usd.Stage) -> None:
    graph = graph_path("ROS_TF")
    delete_if_present(
        stage, [f"{graph}/{name}" for name in ("PublisherTF", "Get_Parent_Root_Prim", "on_loaded", "ReadSimTime")]
    )
    delete_if_present(
        stage,
        [
            f"{graph}/{name}"
            for name in (
                "base_link_world_pose",
                "odometry",
                "center_of_mass",
                "turn_velocity_of_center_of_mass",
                "base_link_velocity",
                "odom_to_base_link",
                "map_to_odom",
                "publish_odometry",
            )
        ],
    )
    center_of_mass = stage.GetPrimAtPath(BASE_LINK).GetAttribute("physics:centerOfMass").Get()
    og.Controller.edit(
        graph,
        {
            keys.CREATE_NODES: [
                ("base_link_world_pose", "isaacsim.core.nodes.IsaacReadWorldPose"),
                ("odometry", "isaacsim.core.nodes.IsaacComputeOdometry"),
                ("center_of_mass", "omni.graph.nodes.ConstantDouble3"),
                ("turn_velocity_of_center_of_mass", "omni.graph.nodes.CrossProduct"),
                ("base_link_velocity", "omni.graph.nodes.Subtract"),
                ("odom_to_base_link", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                ("map_to_odom", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                ("publish_odometry", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
            ],
            keys.SET_VALUES: [
                (f"{graph}/center_of_mass.inputs:value", tuple(center_of_mass)),
                (f"{graph}/odom_to_base_link.inputs:parentFrameId", "odom"),
                (f"{graph}/odom_to_base_link.inputs:childFrameId", "base_link"),
                (f"{graph}/odom_to_base_link.inputs:topicName", "/tf"),
                (f"{graph}/map_to_odom.inputs:parentFrameId", "map"),
                (f"{graph}/map_to_odom.inputs:childFrameId", "odom"),
                (f"{graph}/map_to_odom.inputs:topicName", "/tf"),
                (f"{graph}/publish_odometry.inputs:topicName", "/odom"),
                (f"{graph}/publish_odometry.inputs:odomFrameId", "odom"),
                (f"{graph}/publish_odometry.inputs:chassisFrameId", "base_link"),
                # the projection of world velocities that the node does otherwise leaves them in the world frame
                (f"{graph}/publish_odometry.inputs:publishRawVelocities", True),
            ],
            keys.CONNECT: [
                (f"{graph}/OnPlaybackTick.outputs:tick", f"{graph}/odometry.inputs:execIn"),
                (f"{graph}/odometry.outputs:execOut", f"{graph}/publish_odometry.inputs:execIn"),
                (f"{graph}/OnPlaybackTick.outputs:tick", f"{graph}/odom_to_base_link.inputs:execIn"),
                (f"{graph}/OnPlaybackTick.outputs:tick", f"{graph}/map_to_odom.inputs:execIn"),
                (f"{graph}/Context.outputs:context", f"{graph}/odom_to_base_link.inputs:context"),
                (f"{graph}/Context.outputs:context", f"{graph}/map_to_odom.inputs:context"),
                (f"{graph}/Context.outputs:context", f"{graph}/publish_odometry.inputs:context"),
                (f"{graph}/base_link_world_pose.outputs:translation", f"{graph}/odom_to_base_link.inputs:translation"),
                (f"{graph}/base_link_world_pose.outputs:orientation", f"{graph}/odom_to_base_link.inputs:rotation"),
                (f"{graph}/base_link_world_pose.outputs:translation", f"{graph}/publish_odometry.inputs:position"),
                (f"{graph}/base_link_world_pose.outputs:orientation", f"{graph}/publish_odometry.inputs:orientation"),
                # odometry reports the velocity of the center of mass, which circles base_link when turning
                (f"{graph}/odometry.outputs:angularVelocity", f"{graph}/turn_velocity_of_center_of_mass.inputs:a"),
                (f"{graph}/center_of_mass.inputs:value", f"{graph}/turn_velocity_of_center_of_mass.inputs:b"),
                (f"{graph}/odometry.outputs:linearVelocity", f"{graph}/base_link_velocity.inputs:a"),
                (f"{graph}/turn_velocity_of_center_of_mass.outputs:product", f"{graph}/base_link_velocity.inputs:b"),
                (f"{graph}/base_link_velocity.outputs:difference", f"{graph}/publish_odometry.inputs:linearVelocity"),
                (f"{graph}/odometry.outputs:angularVelocity", f"{graph}/publish_odometry.inputs:angularVelocity"),
            ],
        },
    )
    set_target(stage, f"{graph}/base_link_world_pose.inputs:prim", BASE_LINK)
    set_target(stage, f"{graph}/odometry.inputs:chassisPrim", BASE_LINK)
    stamp_with_simulation_time(
        "ROS_TF",
        [
            "PublisherTF_static",
            "PublisherTF_static_01",
            "PublisherTF_static_02",
            "PublisherTF_static_03",
            "odom_to_base_link",
            "map_to_odom",
            "publish_odometry",
        ],
    )


def update_clock_graph(stage: Usd.Stage) -> None:
    graph = graph_path("ROS_Clock")
    delete_if_present(stage, [graph])
    og.Controller.edit(
        {"graph_path": graph, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("publish_clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            keys.SET_VALUES: [(f"{graph}/publish_clock.inputs:topicName", "/clock")],
            keys.CONNECT: [
                (f"{graph}/OnPlaybackTick.outputs:tick", f"{graph}/publish_clock.inputs:execIn"),
                (f"{graph}/Context.outputs:context", f"{graph}/publish_clock.inputs:context"),
            ],
        },
    )
    stamp_with_simulation_time("ROS_Clock", ["publish_clock"])


def update_camera_graph(stage: Usd.Stage) -> None:
    graph = graph_path("ROS_Camera")
    delete_if_present(
        stage,
        [
            f"{graph}/{name}"
            for name in (
                "ros2_publish_camera_info",
                "isaac_read_camera_info",
                "isaac_read_system_time",
                "make_array",
                "make_array_01",
                "make_array_02",
                "make_array_03",
            )
        ],
    )
    og.Controller.edit(
        graph,
        {
            keys.SET_VALUES: [(f"{graph}/{node}.inputs:topicName", topic) for node, topic in CAMERA_TOPICS.items()]
            + [(f"{graph}/{node}.inputs:useSystemTime", False) for node in CAMERA_TOPICS]
        },
    )


def update_lidar_and_joint_state_graphs() -> None:
    og.Controller.edit(
        graph_path("ROS_Lidar"),
        {keys.SET_VALUES: [(f"{graph_path('ROS_Lidar')}/ros2_rtx_lidar_helper.inputs:useSystemTime", False)]},
    )
    stamp_with_simulation_time("ROS_JointStates", ["PublisherJointState"])


def update_service_graphs(stage: Usd.Stage) -> None:
    delete_if_present(stage, [graph_path("ROS_Service_Server"), graph_path("BasePose")])
    og.Controller.edit(
        graph_path("ROS_Is_homed"),
        {
            keys.SET_VALUES: [
                (
                    f"{graph_path('ROS_Is_homed')}/ros2_service_server_request_02.inputs:serviceName",
                    "/navigation_control/move_camera_oscillate",
                )
            ]
        },
    )


def main() -> None:
    extensions.enable_extension("isaacsim.ros2.bridge")
    app.update()
    usd_path = args.usd.resolve()
    omni.usd.get_context().open_stage(str(usd_path))
    app.update()
    stage = omni.usd.get_context().get_stage()

    update_transform_graph(stage)
    update_clock_graph(stage)
    update_camera_graph(stage)
    update_lidar_and_joint_state_graphs()
    update_service_graphs(stage)

    stage.GetRootLayer().Save()
    print(f"Updated {usd_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        os._exit(1)  # SimulationApp.close() would exit with 0
    app.close()
