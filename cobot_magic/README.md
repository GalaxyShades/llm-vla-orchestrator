# Cobot Magic Setup

This directory contains the vendor Cobot Magic, Piper, ALOHA, and camera ROS
assets used by the chess hardware executor.

The chess pipeline uses the direct ROS joint publisher path:

1. `orchestrator.executor.CobotChessExecutor`
2. `cobot_magic/mobile_aloha_AgileX/robot_utils.py`
3. Cobot/Piper ROS topics such as `/master/joint_left` and `/master/joint_right`

It does not require OpenPI.

## Prerequisites

- Ubuntu with ROS Noetic available.
- CAN adapter configured for the Piper arms.
- The project Conda environment:

```bash
conda activate llm-vla-orchestrator
```

## CAN Setup

From the Piper ROS directory, inspect and configure CAN interfaces:

```bash
cd cobot_magic/Piper_ros_private-ros-noetic
bash find_all_can_port_en.sh
bash can_config_en.sh
bash can_activate.sh
```

Use the non-English variants if that is what your local hardware guide expects:

```bash
bash find_all_can_port.sh
bash can_config.sh
bash can_activate.sh
```

## ROS Workspace Setup

Build or source the supplied workspaces as required by your machine image.
Typical source commands are:

```bash
source /opt/ros/noetic/setup.bash
source cobot_magic/aloha_ws/devel/setup.bash
source cobot_magic/camera_ws/devel/setup.bash
source cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash
```

If `Piper_ros_private-ros-noetic/devel/setup.bash` is missing, build it first
from that workspace using your standard Catkin workflow.

## Start Piper ROS Nodes

For master/slave dual-arm operation, start the Piper launch file:

```bash
source /opt/ros/noetic/setup.bash
source cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash
roslaunch piper start_ms_piper.launch
```

For a single arm, use:

```bash
roslaunch piper start_single_piper.launch
```

Confirm the expected topics exist:

```bash
rostopic list | grep -E 'master|puppet|joint'
```

The chess executor publishes joint commands through the topic names configured
in `cobot_magic/mobile_aloha_AgileX/robot_utils.py`.

## Chess Integration

Calibrate board squares into 7D joint targets:

```yaml
squares:
  e2: [joint0, joint1, joint2, joint3, joint4, joint5, gripper]
  e4: [joint0, joint1, joint2, joint3, joint4, joint5, gripper]
```

Save them in:

```text
configs/cobot_chess_square_poses.yaml
```

Then enable the hardware executor in `configs/chess_move.yaml`:

```yaml
chess:
  execution:
    type: cobot
    cobot_module_path: cobot_magic/mobile_aloha_AgileX
    square_poses_path: configs/cobot_chess_square_poses.yaml
    active_arm: left
    move_method: continuous
```

Run the chess backend from the repository root:

```bash
/home/lem/miniconda3/envs/llm-vla-orchestrator/bin/python -m orchestrator.run \
  --config configs/chess_move.yaml \
  --serve-api
```

Keep `type: dry_run` until the square poses have been measured and the ROS
topics are verified on the real robot.
