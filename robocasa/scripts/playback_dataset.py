import argparse
import json
import os
import random
from copy import deepcopy
import time

import h5py
import imageio
import numpy as np
import robosuite
from termcolor import colored

import robocasa
from robocasa.models.scenes.scene_registry import StyleType


LEAP_HAND_DOF = 16
# ROBOT_SWAPPED = (
#     True  # Set to True when replaying with a different robot than the dataset
# )


def remap_panda_omron_actions_to_leap(actions):
    """
    Remap 12-dim PandaOmron actions to 27-dim PandaDexLeapRHOmron actions.

    PandaOmron body_part_ordering:  [right(6), right_gripper(1), base(3), torso(1), base_mode(1)] = 12
    PandaDexLeapRHOmron ordering:   [right(6), right_gripper(16), base(3), torso(1), base_mode(1)] = 27

    The original 1-DOF gripper action is discarded; 16 zeros are inserted for the LEAP hand.
    """
    n = actions.shape[0]
    arm = actions[:, 0:6]  # 6-dim arm OSC_POSE
    # actions[:, 6] is the 1-DOF gripper — discard
    base_torso_mode = actions[:, 7:]  # base(3) + torso(1) + base_mode(1) = 5
    leap_zeros = np.zeros((n, LEAP_HAND_DOF))
    return np.concatenate([arm, leap_zeros, base_torso_mode], axis=1)


def playback_trajectory_with_env(
    env,
    initial_state,
    states,
    actions=None,
    render=False,
    video_writer=None,
    video_skip=5,
    camera_names=None,
    first=False,
    verbose=False,
    camera_height=512,
    camera_width=512,
    skip_model_and_state=False,
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state.
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load
        actions (np.array): if provided, play actions back open-loop instead of using @states
        render (bool): if True, render on-screen
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
        skip_model_and_state (bool): only for replaying with a swapped robot, whose geoms and
            joint dimensions do not match the recorded ones. Leave False otherwise: without
            the dataset's model and initial state the scene is rebuilt from scratch and the
            object placements are re-sampled, so replayed actions reach for objects that are
            no longer where they were recorded.
    """
    write_video = video_writer is not None
    video_count = 0
    assert not (render and write_video)

    # load the initial state
    ## this reset call doesn't seem necessary.
    ## seems ok to remove but haven't fully tested it.
    ## removing for now
    # env.reset()

    if verbose:
        ep_meta = json.loads(initial_state["ep_meta"])
        lang = ep_meta.get("lang", None)
        if lang is not None:
            print(colored(f"Instruction: {lang}", "green"))
        print(colored("Spawning environment...", "yellow"))
    reset_to(env, initial_state, skip_model_and_state=skip_model_and_state)

    traj_len = states.shape[0]
    action_playback = actions is not None
    if action_playback:
        assert states.shape[0] == actions.shape[0]

    # Check for state dimension mismatch (can happen with style override)
    env_state_dim = env.sim.get_state().flatten().shape[0]
    dataset_state_dim = states.shape[1] if len(states.shape) > 1 else states.shape[0]
    state_dim_match = env_state_dim == dataset_state_dim
    if not state_dim_match:
        print(
            colored(
                f"WARNING: state dimension mismatch! "
                f"env={env_state_dim}, dataset={dataset_state_dim}. "
                f"Different style likely uses different fixture models. "
                f"Skipping state replay — only the initial scene will be rendered.",
                "red",
            )
        )

    if render is False:
        print(colored("Running episode...", "yellow"))

    for i in range(traj_len):
        start = time.time()

        if action_playback:
            env.step(actions[i])
            if False and i < traj_len - 1:
                # check whether the actions deterministically lead to the same recorded states
                # (skip when robot is swapped since state dimensions differ)
                state_playback = np.array(env.sim.get_state().flatten())
                if not np.all(np.equal(states[i + 1], state_playback)):
                    err = np.linalg.norm(states[i + 1] - state_playback)
                    if verbose or i == traj_len - 2:
                        print(
                            colored(
                                "warning: playback diverged by {} at step {}".format(
                                    err, i
                                ),
                                "yellow",
                            )
                        )
        elif state_dim_match:
            reset_to(env, {"states": states[i]})
        # else: skip state loading due to dimension mismatch

        # on-screen render
        if render:
            if env.viewer is None:
                env.initialize_renderer()
                if camera_names is not None and len(camera_names) > 0:
                    try:
                        cam_id = env.sim.model.camera_name2id(camera_names[0])
                        env.viewer.set_camera(cam_id)
                    except Exception as e:
                        print(f"Could not set camera {camera_names[0]}: {e}")

            # so that mujoco viewer renders
            env.viewer.update()

            max_fr = 60
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

        # video render
        if write_video:
            if video_count % video_skip == 0:
                video_img = []
                for cam_name in camera_names:
                    im = env.sim.render(
                        height=camera_height, width=camera_width, camera_name=cam_name
                    )[::-1]
                    video_img.append(im)
                video_img = np.concatenate(
                    video_img, axis=1
                )  # concatenate horizontally
                video_writer.append_data(video_img)

            video_count += 1

        if first:
            break

    if render:
        env.viewer.close()
        env.viewer = None


def playback_trajectory_with_obs(
    traj_grp,
    video_writer,
    video_skip=5,
    image_names=None,
    first=False,
):
    """
    This function reads all "rgb" observations in the dataset trajectory and
    writes them into a video.

    Args:
        traj_grp (hdf5 file group): hdf5 group which corresponds to the dataset trajectory to playback
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        image_names (list): determines which image observations are used for rendering. Pass more than
            one to output a video with multiple image observations concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
        skip_model_and_state (bool): only for replaying with a swapped robot, whose geoms and
            joint dimensions do not match the recorded ones. Leave False otherwise: without
            the dataset's model and initial state the scene is rebuilt from scratch and the
            object placements are re-sampled, so replayed actions reach for objects that are
            no longer where they were recorded.
    """
    assert (
        image_names is not None
    ), "error: must specify at least one image observation to use in @image_names"
    video_count = 0

    traj_len = traj_grp["obs/{}".format(image_names[0] + "_image")].shape[0]
    for i in range(traj_len):
        if video_count % video_skip == 0:
            # concatenate image obs together
            im = [traj_grp["obs/{}".format(k + "_image")][i] for k in image_names]
            frame = np.concatenate(im, axis=1)
            video_writer.append_data(frame)
        video_count += 1

        if first:
            break


def get_env_metadata_from_dataset(dataset_path, ds_format="robomimic"):
    """
    Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    if ds_format == "robomimic":
        if "env_args" in f["data"].attrs:
            env_meta = json.loads(f["data"].attrs["env_args"])
            # Override robot to PandaDexLeapRHOmron for dexterous hand playback
            # env_meta["env_kwargs"]["robots"] = "PandaDexLeapRHOmron"
            # Disable data-collection-only features that can fail during playback
            env_meta["env_kwargs"].pop("generative_textures", None)
            env_meta["env_kwargs"].pop("randomize_cameras", None)
        else:
            env_meta = {
                "env_name": "CoffeePressButton",
                "type": "kitchen",
                "env_kwargs": {
                    # "robots": "PandaDexLeapRHOmron"
                },
            }
    else:
        raise ValueError
    f.close()
    return env_meta


class ObservationKeyToModalityDict(dict):
    """
    Custom dictionary class with the sole additional purpose of automatically registering new "keys" at runtime
    without breaking. This is mainly for backwards compatibility, where certain keys such as "latent", "actions", etc.
    are used automatically by certain models (e.g.: VAEs) but were never specified by the user externally in their
    config. Thus, this dictionary will automatically handle those keys by implicitly associating them with the low_dim
    modality.
    """

    def __getitem__(self, item):
        # If a key doesn't already exist, warn the user and add default mapping
        if item not in self.keys():
            print(
                f"ObservationKeyToModalityDict: {item} not found,"
                f" adding {item} to mapping with assumed low_dim modality!"
            )
            self.__setitem__(item, "low_dim")
        return super(ObservationKeyToModalityDict, self).__getitem__(item)


def reset_to(env, state, skip_model_and_state=False):
    """
    Reset to a specific simulator state.

    Args:
        state (dict): current simulator state that contains one or more of:
            - states (np.ndarray): initial state of the mujoco environment.
                Corresponds to the flattened (states) in the spec.
            - model (str): mujoco scene xml
        skip_model_and_state (bool): if True, skip loading the dataset's XML model
            and sim state. Used when replaying with a different robot, since the
            dataset XML and state dimensions belong to the original robot.

    Returns:
        None
    """
    should_ret = False
    if skip_model_and_state:
        # When robot is swapped, we can't reuse the dataset's XML or state
        # (different geoms, different joint dimensions). Just reset the env.
        if state.get("ep_meta", None) is not None:
            ep_meta = json.loads(state["ep_meta"])
        else:
            ep_meta = {}
        if hasattr(env, "set_attrs_from_ep_meta"):
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):
            env.set_ep_meta(ep_meta)
        env.reset()
    elif "model" in state:
        if state.get("ep_meta", None) is not None:
            # set relevant episode information
            ep_meta = json.loads(state["ep_meta"])
        else:
            ep_meta = {}
        if hasattr(env, "set_attrs_from_ep_meta"):  # older versions had this function
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):  # newer versions
            env.set_ep_meta(ep_meta)
        # this reset is necessary.
        # while the call to env.reset_from_xml_string does call reset,
        # that is only a "soft" reset that doesn't actually reload the model.
        env.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(state["model"])
        else:
            # v1.4 and above use the class-based edit_model_xml function
            xml = env.edit_model_xml(state["model"])

        env.reset_from_xml_string(xml)
        env.sim.reset()
        # hide teleop visualization after restoring from model
        # env.sim.model.site_rgba[env.eef_site_id] = np.array([0., 0., 0., 0.])
        # env.sim.model.site_rgba[env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
    if not skip_model_and_state and "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
        should_ret = True

    # update state as needed
    if hasattr(env, "update_sites"):
        # older versions of environment had update_sites function
        env.update_sites()
    if hasattr(env, "update_state"):
        # later versions renamed this to update_state
        env.update_state()

    # if should_ret:
    #     # only return obs if we've done a forward call - otherwise the observations will be garbage
    #     return get_observation()
    return None


def _override_style_in_initial_state(initial_state, style_id):
    """
    Return a copy of initial_state with the ep_meta style_id overridden.
    """
    from copy import deepcopy

    new_state = deepcopy(initial_state)
    if new_state.get("ep_meta") is not None:
        ep_meta = json.loads(new_state["ep_meta"])
    else:
        ep_meta = {}
    ep_meta["style_id"] = int(style_id)
    new_state["ep_meta"] = json.dumps(ep_meta, indent=4)
    return new_state


def _style_name(style_id):
    """Return human-readable style name for a style id."""
    int_to_name = {s.value: s.name.lower() for s in StyleType if s.value >= 0}
    return int_to_name.get(int(style_id), f"style{style_id}")


# Coffee machine model used by each style (from kitchen_styles/*.yaml)
STYLE_COFFEE_MACHINE = {
    0: "delonghi_espresso",  # industrial
    1: "nespresso",  # scandanavian
    2: "delonghi_espresso_2",  # coastal
    3: "delonghi_espresso_2",  # modern_1
    4: "delonghi_espresso",  # modern_2
    5: "delonghi_espresso",  # traditional_1
    6: "delonghi_espresso_2",  # traditional_2
    7: "nespresso",  # farmhouse
    8: "nespresso",  # rustic
    9: "delonghi_espresso",  # mediterranean
    10: "nespresso",  # transitional_1
    11: "delonghi_espresso_2",  # transitional_2
}


def _styles_with_same_coffee_machine(style_id):
    """Return list of style IDs that use the same coffee machine model."""
    target = STYLE_COFFEE_MACHINE.get(int(style_id))
    if target is None:
        return list(range(12))
    return [sid for sid, cm in STYLE_COFFEE_MACHINE.items() if cm == target]


def playback_dataset(args):
    # some arg checking
    write_video = args.render is not True
    if args.video_path is None:
        args.video_path = args.dataset.split(".hdf5")[0] + ".mp4"
        if args.use_actions:
            args.video_path = args.dataset.split(".hdf5")[0] + "_use_actions.mp4"
        elif args.use_abs_actions:
            args.video_path = args.dataset.split(".hdf5")[0] + "_use_abs_actions.mp4"
    assert not (args.render and write_video)  # either on-screen or video but not both

    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        # We fill in the automatic values
        env_meta = get_env_metadata_from_dataset(dataset_path=args.dataset)
        args.render_image_names = "robot0_eye_in_hand"  # "robot0_agentview_center"

    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.render_image_names) == 1

    if args.use_obs:
        assert write_video, "playback with observations can only write to video"
        assert (
            not args.use_actions and not args.use_abs_actions
        ), "playback with observations is offline and does not support action playback"

    # Determine which styles to play back
    style_ids = getattr(args, "style_ids", None)
    if style_ids is not None:
        # Expand "all" to the full list of style IDs
        if style_ids == ["all"]:
            style_ids = list(range(12))
        else:
            style_ids = [int(s) for s in style_ids]
        print(
            colored(
                f"Style override enabled: will replay each trajectory with styles {style_ids}",
                "cyan",
            )
        )

    env = None

    # create environment only if not playing back with observations
    if not args.use_obs:
        env_meta = get_env_metadata_from_dataset(dataset_path=args.dataset)
        if args.use_abs_actions:
            env_meta["env_kwargs"]["controller_configs"][
                "control_delta"
            ] = False  # absolute action space

        dataset_robots = deepcopy(env_meta["env_kwargs"].get("robots"))

        env_kwargs = env_meta["env_kwargs"]
        env_kwargs["env_name"] = env_meta["env_name"]
        env_kwargs["has_renderer"] = False
        env_kwargs["renderer"] = "mjviewer"
        env_kwargs["has_offscreen_renderer"] = write_video
        env_kwargs["use_camera_obs"] = False

        # When overriding styles, allow all styles the env might need
        if style_ids is not None:
            env_kwargs["layout_and_style_ids"] = None
            env_kwargs["style_ids"] = style_ids
            env_kwargs.pop("layout_ids", None)

        robot_swapped = env_kwargs.get("robots") != dataset_robots

        if args.verbose:
            print(
                colored(
                    "Initializing environment for {}...".format(env_kwargs["env_name"]),
                    "yellow",
                )
            )

        env = robosuite.make(**env_kwargs)

    f = h5py.File(args.dataset, "r")

    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [
            elem.decode("utf-8")
            for elem in np.array(f["mask/{}".format(args.filter_key)])
        ]
    elif "data" in f.keys():
        demos = list(f["data"].keys())

    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        random.shuffle(demos)
        demos = demos[: args.n]

    # maybe dump video
    # Play back at the rate the demos were recorded at. env_meta is only bound on
    # some branches above, and older datasets predate control_freq being stored in
    # env_args, so read it defensively and fall back to the Kitchen default.
    try:
        video_fps = get_env_metadata_from_dataset(dataset_path=args.dataset)[
            "env_kwargs"
        ].get("control_freq", 20)
    except Exception:
        video_fps = 20
    video_writer = None
    if write_video and style_ids is None:
        video_writer = imageio.get_writer(args.video_path, fps=video_fps)

    for ind in range(len(demos)):
        ep = demos[ind]
        print(colored("\nPlaying back episode: {}".format(ep), "yellow"))

        if args.use_obs:
            playback_trajectory_with_obs(
                traj_grp=f["data/{}".format(ep)],
                video_writer=video_writer,
                video_skip=args.video_skip,
                image_names=args.render_image_names,
                first=args.first,
            )
            continue

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)

        if args.extend_states:
            states = np.concatenate((states, [states[-1]] * 50))

        # supply actions if using open-loop action playback
        actions = None
        assert not (
            args.use_actions and args.use_abs_actions
        )  # cannot use both relative and absolute actions
        if args.use_actions:
            actions = f["data/{}/actions".format(ep)][()]
        elif args.use_abs_actions:
            actions = f["data/{}/actions_abs".format(ep)][()]

        # --- Style override: replay each trajectory with each requested style ---
        if style_ids is not None:
            # Read original style for the label
            orig_ep_meta = (
                json.loads(initial_state["ep_meta"])
                if initial_state.get("ep_meta")
                else {}
            )
            orig_style = orig_ep_meta.get("style_id", "?")
            orig_layout = orig_ep_meta.get("layout_id", "?")
            print(
                colored(
                    f"  Original: layout={orig_layout}, style={orig_style} ({_style_name(orig_style)})",
                    "green",
                )
            )

            # Determine which styles have a compatible coffee machine
            orig_coffee = (
                STYLE_COFFEE_MACHINE.get(int(orig_style)) if orig_style != "?" else None
            )
            compatible_styles = (
                _styles_with_same_coffee_machine(orig_style)
                if orig_style != "?"
                else None
            )

            for sid in style_ids:
                style_label = _style_name(sid)

                # Skip styles with a different coffee machine model
                if compatible_styles is not None and sid not in compatible_styles:
                    target_coffee = STYLE_COFFEE_MACHINE.get(sid, "?")
                    print(
                        colored(
                            f"  Skipping style {sid} ({style_label}): "
                            f"coffee machine mismatch ({target_coffee} vs {orig_coffee})",
                            "yellow",
                        )
                    )
                    continue

                print(
                    colored(
                        f"  Replaying {ep} with style {sid} ({style_label})...",
                        "cyan",
                    )
                )
                overridden_state = _override_style_in_initial_state(initial_state, sid)

                # Per-style video writer
                vw = None
                style_video_path = None
                if write_video:
                    base = args.video_path.rsplit(".", 1)[0]
                    ext = (
                        args.video_path.rsplit(".", 1)[1]
                        if "." in args.video_path
                        else "mp4"
                    )
                    style_video_path = f"{base}_{ep}_style{sid}_{style_label}.{ext}"
                    vw = imageio.get_writer(style_video_path, fps=video_fps)

                playback_trajectory_with_env(
                    env=env,
                    initial_state=overridden_state,
                    states=states,
                    actions=actions,
                    render=args.render,
                    video_writer=vw,
                    video_skip=args.video_skip,
                    camera_names=args.render_image_names,
                    first=args.first,
                    verbose=args.verbose,
                    camera_height=args.camera_height,
                    camera_width=args.camera_width,
                    skip_model_and_state=robot_swapped,
                )

                if vw is not None:
                    vw.close()
                    print(colored(f"  Saved: {style_video_path}", "green"))
        else:
            # Normal playback (no style override)
            playback_trajectory_with_env(
                env=env,
                initial_state=initial_state,
                states=states,
                actions=actions,
                render=args.render,
                video_writer=video_writer,
                video_skip=args.video_skip,
                camera_names=args.render_image_names,
                first=args.first,
                verbose=args.verbose,
                camera_height=args.camera_height,
                camera_width=args.camera_width,
                skip_model_and_state=robot_swapped,
            )

    f.close()
    if write_video and video_writer is not None:
        print(colored(f"Saved video to {args.video_path}", "green"))
        video_writer.close()

    if env is not None:
        env.close()


def get_playback_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )

    # number of trajectories to playback. If omitted, playback all of them.
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are played",
    )

    # Use image observations instead of doing playback using the simulator env.
    parser.add_argument(
        "--use-obs",
        action="store_true",
        help="visualize trajectories with dataset image observations instead of simulator",
    )

    # Playback stored dataset actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-actions",
        action="store_true",
        help="use open-loop action playback instead of loading sim states",
    )

    # Playback stored dataset absolute actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-abs-actions",
        action="store_true",
        help="use open-loop action playback with absolute position actions instead of loading sim states",
    )

    # Whether to render playback to screen
    parser.add_argument(
        "--render",
        action="store_true",
        help="on-screen rendering",
    )

    # Dump a video of the dataset playback to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render trajectories to this video file path",
    )

    # How often to write video frames during the playback
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render, or image observations to use for writing to video
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs="+",
        default=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
        "None, which corresponds to a predefined camera for each env type",
    )

    # Only use the first frame of each episode
    parser.add_argument(
        "--first",
        action="store_true",
        help="use first frame of each episode",
    )

    parser.add_argument(
        "--extend_states",
        action="store_true",
        help="play last step of episodes for 50 extra frames",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="log additional information",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=512,
        help="(optional, for offscreen rendering) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=512,
        help="(optional, for offscreen rendering) width of image observations",
    )

    parser.add_argument(
        "--style_ids",
        type=str,
        nargs="+",
        default=None,
        help="override style(s) to replay trajectories with (0-11 or 'all'). "
        "Each trajectory is replayed once per compatible style. "
        "Styles with a different coffee machine model are automatically skipped. "
        "Available styles: "
        "0=industrial, 1=scandanavian, 2=coastal, 3=modern_1, 4=modern_2, "
        "5=traditional_1, 6=traditional_2, 7=farmhouse, 8=rustic, "
        "9=mediterranean, 10=transitional_1, 11=transitional_2",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_playback_args()
    playback_dataset(args)
