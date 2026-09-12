"""
Registry of the robot embodiments supported in robocasa kitchen environments.

Everything that differs between embodiments (short-name aliases, spawn pose, action-vector
layout, which controller config to load) lives here, so switching arms is a one-flag change:

    python robocasa/scripts/collect_demos.py --environment CoffeePressButton --robots xarm6
    python robocasa/scripts/collect_demos.py --environment CoffeePressButton --robots xarm6_leap

The robots themselves are defined in robosuite (``robosuite/models/robots/compositional.py``).
All four kitchen robots pair an arm with the Omron mobile base and a torso lift:

    PandaOmron             Franka Panda        + Franka parallel gripper
    PandaDexLeapRHOmron    Franka Panda        + LEAP right hand
    XArm6Omron             UFACTORY xArm 6     + UFACTORY parallel gripper
    XArm6DexLeapRHOmron    UFACTORY xArm 6     + LEAP right hand
"""

import os
import pathlib
from copy import deepcopy

import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config

# Short names accepted anywhere a robot can be specified, plus back-compat aliases.
ROBOT_ALIASES = {
    # franka
    "panda": "PandaOmron",
    "franka": "PandaOmron",
    "panda_leap": "PandaDexLeapRHOmron",
    "franka_leap": "PandaDexLeapRHOmron",
    # ufactory xarm 6
    "xarm6": "XArm6Omron",
    "xarm": "XArm6Omron",
    "xarm6_leap": "XArm6DexLeapRHOmron",
    "xarm_leap": "XArm6DexLeapRHOmron",
    # renamed in robosuite v1.5
    "pandamobile": "PandaOmron",
}

# Action-vector layout used for every kitchen robot. Keeping this identical across embodiments
# means datasets collected with different arms index the same way.
DEFAULT_BODY_PART_ORDERING = ["right", "right_gripper", "base", "torso"]

# Spawn poses, tuned for kitchen scenes: more retracted than the robosuite defaults so the
# hand does not clip counter-mounted fixtures, and torso all the way down.
#
# The Panda pose is dataset-derived (avg at 0.7s of human demos). Each xArm pose was solved with
# IK to put ``gripper0_right_grip_site`` on the same pose (position *and* finger orientation) as
# its Panda counterpart, so task tuning carries over between embodiments.
KITCHEN_ROBOTS = {
    "PandaOmron": dict(
        init_qpos=np.array(
            [
                -0.01612974,
                -1.03446714,
                -0.02397936,
                -2.27550888,
                0.03932365,
                1.51639493,
                0.69615947,
            ]
        ),
        init_torso_qpos=np.array([0.0]),
        body_part_ordering=DEFAULT_BODY_PART_ORDERING,
        dexterous=False,
    ),
    "PandaDexLeapRHOmron": dict(
        # same arm pose as PandaOmron; the LEAP hand is bulkier but mounts on the same flange
        init_qpos=np.array(
            [
                -0.01612974,
                -1.03446714,
                -0.02397936,
                -2.27550888,
                0.03932365,
                1.51639493,
                0.69615947,
            ]
        ),
        init_torso_qpos=np.array([0.0]),
        body_part_ordering=DEFAULT_BODY_PART_ORDERING,
        dexterous=True,
    ),
    "XArm6Omron": dict(
        init_qpos=np.array(
            [-0.107722, -0.575594, -1.673465, 0.087910, 1.977543, 3.013432]
        ),
        init_torso_qpos=np.array([0.0]),
        body_part_ordering=DEFAULT_BODY_PART_ORDERING,
        dexterous=False,
    ),
    "XArm6DexLeapRHOmron": dict(
        init_qpos=np.array(
            [-0.050244, -0.765494, -1.274084, 0.073706, 1.598932, 0.744908]
        ),
        init_torso_qpos=np.array([0.0]),
        body_part_ordering=DEFAULT_BODY_PART_ORDERING,
        dexterous=True,
    ),
}

# Controller config variants. "joint_pos" records absolute joint targets instead of Cartesian
# deltas, which is what we replay on the real arm (sim2real).
CONTROL_MODES = ("osc", "joint_pos")


def resolve_robot_name(robot):
    """
    Maps a robot alias (eg "xarm6") to its robosuite robot name (eg "XArm6Omron").
    Names that are already robosuite robot names pass through untouched.

    Args:
        robot (str): robot name or alias

    Returns:
        str: robosuite robot name
    """
    if not isinstance(robot, str):
        raise TypeError(f"expected a robot name string, got {type(robot)}")
    return ROBOT_ALIASES.get(robot.lower(), robot)


def resolve_robot_names(robots):
    """
    Same as @resolve_robot_name, but accepts a single name or a list of names.

    Args:
        robots (str or list): robot name(s) or alias(es)

    Returns:
        str or list: robosuite robot name(s), matching the input type
    """
    if isinstance(robots, str):
        return resolve_robot_name(robots)
    return [resolve_robot_name(robot) for robot in robots]


def get_robot_config(robot):
    """
    Looks up the kitchen-specific config for a robot. Falls back to the config of the nearest
    registered ancestor class, so robot subclasses inherit their parent's spawn pose.

    Args:
        robot (str or RobotModel): robot name, alias, or robot model instance

    Returns:
        dict or None: robot config, or None if neither the robot nor any of its ancestors
            are registered here (in which case robosuite's own defaults apply)
    """
    if isinstance(robot, str):
        config = KITCHEN_ROBOTS.get(resolve_robot_name(robot))
    else:
        config = None
        for cls in type(robot).__mro__:
            if cls.__name__ in KITCHEN_ROBOTS:
                config = KITCHEN_ROBOTS[cls.__name__]
                break

    # hand back a copy so callers cannot mutate the registry through it
    return deepcopy(config) if config is not None else None


def get_controller_config(robot, control_mode="osc"):
    """
    Loads the controller config for a robot in the requested control mode.

    Args:
        robot (str): robot name or alias

        control_mode (str): "osc" for Cartesian delta control (teleop with a spacemouse /
            keyboard), or "joint_pos" for absolute joint position control (teleop with a device
            that solves IK itself, eg quest_rokoko, and the representation we replay on hardware)

    Returns:
        dict: composite controller config
    """
    assert (
        control_mode in CONTROL_MODES
    ), f"unknown control_mode {control_mode}, expected one of {CONTROL_MODES}"
    robot = resolve_robot_name(robot)

    if control_mode == "osc":
        return load_composite_controller_config(controller=None, robot=robot)

    config_dir = pathlib.Path(robosuite.__file__).parent / "controllers/config/robots"
    fpath = config_dir / f"default_{robot.lower()}_joint_pos.json"
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"No joint position controller config for robot {robot} (looked for {fpath}). "
            f"Add one next to the other default_*_joint_pos.json configs in robosuite."
        )
    return load_composite_controller_config(controller=str(fpath), robot=robot)


def get_null_action(env):
    """
    Builds an action that tells every robot part to stay where it is.

    A vector of zeros only means "hold" for controllers taking delta inputs. Under absolute
    inputs (eg the joint position control we collect sim2real demos with) zeros command joint
    angles of 0, which yanks the arm out of its pose, so those parts are filled with the
    currently measured state instead.

    Args:
        env (MujocoEnv): environment to build the action for

    Returns:
        np.array: action of size env.action_dim
    """

    def takes_absolute_input(robot, arm):
        controller = robot.part_controllers.get(arm)
        return getattr(controller, "input_type", "delta") == "absolute"

    if not any(
        takes_absolute_input(robot, arm) for robot in env.robots for arm in robot.arms
    ):
        # every part holds on zeros, no need to inspect the robot state
        return np.zeros(env.action_dim)

    actions = []
    for robot in env.robots:
        action_dict = {}
        for arm in robot.arms:
            controller = robot.part_controllers[arm]
            if not takes_absolute_input(robot, arm):
                action_dict[arm] = np.zeros(controller.control_dim)
            elif hasattr(controller, "delta_to_abs_action"):
                # eg OSC_POSE: the absolute action corresponding to a zero delta
                action_dict[arm] = controller.delta_to_abs_action(
                    np.zeros(controller.control_dim), goal_update_mode=None
                )
            else:
                # eg JOINT_POSITION: hold the joints where they are
                joint_indexes = robot._ref_joints_indexes_dict[arm]
                action_dict[arm] = np.array(env.sim.data.qpos[joint_indexes])
        actions.append(robot.create_action_vector(action_dict))
    return np.concatenate(actions)


def is_dexterous(robot):
    """
    Whether a robot uses a dexterous hand (as opposed to a parallel jaw gripper).

    Args:
        robot (str or RobotModel): robot name, alias, or robot model instance

    Returns:
        bool: True if the robot has a dexterous hand
    """
    config = get_robot_config(robot)
    return bool(config is not None and config["dexterous"])
