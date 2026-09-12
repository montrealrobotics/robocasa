"""
Relabel a collected demonstration dataset into multiple joint-space action representations.

Motivation
----------
Demos collected with a JOINT_POSITION arm controller store, per step, the *absolute joint
targets* that were commanded (this is the arm slice of ``actions``) plus the full MuJoCo state
(from which the *measured* joint positions are recovered exactly). Those two signals are enough to
derive every joint-space action space offline, so a single collection run can train policies on
absolute targets, deltas (either reference), or velocities without re-teleoperating.

This mirrors the real robot: DexCap's teleop server streams the same absolute joint targets to the
xArm via deoxys' ``set_servo_angle_j``, so ``abs_joint_pos`` here is the exact quantity commanded on
hardware.

What it writes
--------------
For each demo, an ``action_dict`` group with:

  abs_joint_pos          (T, A)  commanded absolute joint targets      == recorded arm action
  delta_joint_pos        (T, A)  command-referenced delta: qc[t]-qc[t-1]
  delta_joint_pos_state  (T, A)  state-referenced delta:   qc[t]-qm[t]
  joint_vel              (T, A)  command-referenced delta / dt
  gripper_abs            (T, G)  commanded hand/gripper joint targets   == recorded gripper action
  gripper_delta          (T, G)  gripper command-referenced delta

Command- vs state-referenced deltas are both emitted on purpose: a state-referenced delta closes a
loop through the plant (qc[t] - qm[t]), so a policy can learn to lean on tracking error and then
behave differently on a robot whose tracking differs. Command-referenced is usually the safer target
for sim2real; keep both so it stays a deliberate choice.

End-effector representations (--cartesian)
------------------------------------------
With --cartesian, two more representations are added, both in the robot BASE frame (matching our OSC
configs' input_ref_frame="base"):

  abs_pose_pos           (T, 3)  commanded eef position:  FK(qc)
  abs_pose_rot_axis_angle(T, 3)  commanded eef orientation
  abs_pose_rot_6d        (T, 6)  same, 6D rotation (robocasa training convention)
  twist_pos              (T, 3)  base-frame translation from achieved to commanded pose
  twist_rot_axis_angle   (T, 3)  base-frame orientation error, achieved -> commanded

Base-frame eef pose is a pure function of the joint angles through the robot model (independent of
where the robot sits in the scene), so this reconstructs the robot as a forward-kinematics engine
from the dataset's own env metadata -- it is NOT hardcoded to a specific robot. The twist uses
robosuite's OSC delta convention (base-frame translation difference + orientation_error), so it is
directly consumable by an OSC_POSE controller at rollout. "Commanded" pose is FK(qc); "achieved"
pose is FK(qm) -- the tracking-error distinction carries over from the joint-space deltas.

This script assumes the source arm action is absolute joint targets (i.e. the dataset was collected
with a JOINT_POSITION, input_type="absolute" arm controller). It refuses to guess for other layouts,
because joint targets cannot be recovered from recorded OSC deltas.

Example
-------
    python robocasa/scripts/relabel_actions.py --dataset demo.hdf5 --arm-dof 6
    python robocasa/scripts/relabel_actions.py --dataset demo.hdf5 --output demo_relabeled.hdf5
    python robocasa/scripts/relabel_actions.py --dataset demo.hdf5 --cartesian
"""

import argparse
import json
import os
import shutil

import h5py
import numpy as np


def _get_control_freq(f, override):
    """Resolve control frequency (Hz) for velocity scaling: CLI override wins, else dataset metadata."""
    if override is not None:
        return float(override)
    if "env_args" in f["data"].attrs:
        try:
            env_meta = json.loads(f["data"].attrs["env_args"])
            cf = env_meta.get("env_kwargs", {}).get("control_freq", None)
            if cf is not None:
                return float(cf)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return 20.0  # robocasa/robosuite default


def _infer_arm_dof(demo, override):
    """Arm DOF from the measured-joint observable if present, else require an explicit value."""
    if override is not None:
        return int(override)
    for key in ("obs/robot0_joint_pos", "obs/robot0_joint_pos_cos"):
        if key in demo:
            return int(demo[key].shape[-1])
    raise ValueError(
        "Could not infer arm DOF (no obs/robot0_joint_pos in demo). Pass --arm-dof explicitly."
    )


def _measured_joint_pos(demo, arm_dof):
    """Measured arm joint positions per step, used for the state-referenced delta."""
    if "obs/robot0_joint_pos" in demo:
        return np.asarray(demo["obs/robot0_joint_pos"][:], dtype=np.float64)[
            :, :arm_dof
        ]
    return None


# ── Base-frame forward-kinematics engine (for --cartesian) ──────────────────────────────────────


def make_fk_fn(base_env, arm=None):
    """
    Build a base-frame FK closure from a robosuite/robocasa base env.

    Returns (fk_fn, arm_dof, arm), where fk_fn(q) sets the arm joints to q, forwards the sim, and
    returns (pos, rot_matrix) of the eef in the robot base frame. Robot/gripper/arm-dof are read
    from the env, so nothing here is robot-specific.
    """
    robot = base_env.robots[0]
    if arm is None:
        arm = robot.arms[0]
    qpos_idx = np.array(robot._ref_joint_pos_indexes, dtype=int)

    def fk_fn(q):
        base_env.sim.data.qpos[qpos_idx] = q
        base_env.sim.forward()
        pos = np.array(robot._hand_pos[arm], dtype=np.float64)
        rot = np.array(robot._hand_orn[arm], dtype=np.float64)
        return pos, rot

    return fk_fn, len(qpos_idx), arm


def build_fk_from_dataset(dataset_path):
    """
    Reconstruct the collected env from the dataset's own metadata and expose it as an FK engine.
    Returns (env, fk_fn, arm_dof). The env must be closed by the caller.
    """
    import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils
    import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils

    env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=[],
        camera_height=84,
        camera_width=84,
        reward_shaping=False,
    )
    fk_fn, arm_dof, _ = make_fk_fn(env.base_env)
    return env, fk_fn, arm_dof


def _matrix_to_rot_6d(R):
    """6D rotation rep (Zhou et al.): first two rows of the matrix, flattened. Matches robocasa's
    matrix_to_rotation_6d so the key is interchangeable with its training pipeline."""
    return R[:2, :].reshape(6)


def compute_ee_actions(qc, qm, fk_fn):
    """
    Base-frame end-effector action representations from commanded (qc) and measured (qm) joints.

    abs_pose_*  : FK(qc) -- the commanded eef pose in the base frame.
    twist_*     : delta from achieved pose FK(qm) to commanded pose FK(qc), in OSC delta convention
                  (base-frame translation difference + orientation_error), so an OSC_POSE controller
                  reconstructs the target when it applies the delta to the current pose at rollout.

    Depends only on robosuite (no robocasa/torch), so it is unit-testable in isolation.
    """
    import robosuite.utils.transform_utils as T
    from robosuite.utils.control_utils import orientation_error

    n = qc.shape[0]
    abs_pos = np.zeros((n, 3), dtype=np.float64)
    abs_aa = np.zeros((n, 3), dtype=np.float64)
    abs_6d = np.zeros((n, 6), dtype=np.float64)
    tw_pos = np.zeros((n, 3), dtype=np.float64)
    tw_aa = np.zeros((n, 3), dtype=np.float64)

    for t in range(n):
        p_target, R_target = fk_fn(qc[t])
        p_cur, R_cur = fk_fn(qm[t])
        abs_pos[t] = p_target
        abs_aa[t] = T.quat2axisangle(T.mat2quat(R_target))
        abs_6d[t] = _matrix_to_rot_6d(R_target)
        tw_pos[t] = p_target - p_cur
        tw_aa[t] = orientation_error(R_target, R_cur)

    return {
        "abs_pose_pos": abs_pos.astype(np.float32),
        "abs_pose_rot_axis_angle": abs_aa.astype(np.float32),
        "abs_pose_rot_6d": abs_6d.astype(np.float32),
        "twist_pos": tw_pos.astype(np.float32),
        "twist_rot_axis_angle": tw_aa.astype(np.float32),
    }


def relabel_demo(demo, arm_dof, dt, fk_fn=None):
    """Compute the action_dict for a single demo group and write it in place."""
    actions = np.asarray(demo["actions"][:], dtype=np.float64)
    T = actions.shape[0]

    qc = actions[:, :arm_dof]  # commanded absolute arm joint targets
    gripper = actions[:, arm_dof:]  # commanded hand/gripper targets (0-width if none)

    qm = _measured_joint_pos(demo, arm_dof)

    # command-referenced previous target: for the first step, treat the measured initial pose as the
    # "previous command" so the first delta is the real move from where the arm actually started.
    qc_prev = np.empty_like(qc)
    qc_prev[1:] = qc[:-1]
    qc_prev[0] = qm[0] if qm is not None else qc[0]

    delta_cmd = qc - qc_prev
    joint_vel = delta_cmd / dt

    out = {
        "abs_joint_pos": qc.astype(np.float32),
        "delta_joint_pos": delta_cmd.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
    }
    if qm is not None:
        out["delta_joint_pos_state"] = (qc - qm).astype(np.float32)

    if gripper.shape[1] > 0:
        g_prev = np.empty_like(gripper)
        g_prev[1:] = gripper[:-1]
        g_prev[0] = gripper[0]  # no measured hand target reference; hold first
        out["gripper_abs"] = gripper.astype(np.float32)
        out["gripper_delta"] = (gripper - g_prev).astype(np.float32)

    if fk_fn is not None:
        if qm is None:
            raise ValueError(
                "--cartesian requires measured joint positions (obs/robot0_joint_pos); "
                "run dataset_states_to_obs first."
            )
        out.update(compute_ee_actions(qc, qm, fk_fn))

    grp = demo.require_group("action_dict")
    for key, data in out.items():
        if key in grp:
            del grp[key]
        grp.create_dataset(key, data=data)
    return {k: v.shape for k, v in out.items()}, T


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to collected robomimic-format hdf5",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write to this new file instead of modifying --dataset in place (dataset is copied first)",
    )
    parser.add_argument(
        "--arm-dof",
        type=int,
        default=None,
        help="Number of arm joints (default: inferred from obs/robot0_joint_pos)",
    )
    parser.add_argument(
        "--control-freq",
        type=float,
        default=None,
        help="Control frequency (Hz) for velocity scaling (default: from dataset env_args, else 20)",
    )
    parser.add_argument(
        "--cartesian",
        action="store_true",
        help="Also emit base-frame end-effector pose + twist (reconstructs the env for FK)",
    )
    args = parser.parse_args()

    path = os.path.expanduser(args.dataset)
    if args.output is not None:
        out_path = os.path.expanduser(args.output)
        shutil.copyfile(path, out_path)
        path = out_path

    fk_env, fk_fn = None, None
    if args.cartesian:
        print("Building FK engine from dataset env metadata...")
        fk_env, fk_fn, fk_arm_dof = build_fk_from_dataset(path)
        print(f"FK engine ready (arm DOF = {fk_arm_dof})")

    try:
        with h5py.File(path, "r+") as f:
            dt = 1.0 / _get_control_freq(f, args.control_freq)
            demos = list(f["data"].keys())
            print(f"Relabeling {len(demos)} demos in {path}  (dt={dt:.4f}s)")

            arm_dof = None
            for i, name in enumerate(demos):
                demo = f["data"][name]
                if arm_dof is None:
                    arm_dof = _infer_arm_dof(demo, args.arm_dof)
                    if fk_fn is not None and fk_arm_dof != arm_dof:
                        raise ValueError(
                            f"Arm DOF mismatch: obs says {arm_dof}, FK env says {fk_arm_dof}."
                        )
                    print(f"Arm DOF = {arm_dof}")
                shapes, T = relabel_demo(demo, arm_dof, dt, fk_fn=fk_fn)
                if i == 0:
                    print("action_dict keys:")
                    for k, s in shapes.items():
                        print(f"  {k:24s} {s}")
            f["data"].attrs["relabeled_action_spaces"] = json.dumps(list(shapes.keys()))
    finally:
        if fk_env is not None:
            fk_env.env.close()

    print("Done.")


if __name__ == "__main__":
    main()
