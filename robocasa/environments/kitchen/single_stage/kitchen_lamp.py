import mujoco
import numpy as np
from robosuite.utils.mjcf_utils import array_to_string, string_to_array

from robocasa.environments.kitchen.kitchen import *

# One full turn seats the bulb. Kept in sync with the joint range and the equality
# polycoef in robocasa/models/assets/objects/custom/lamp_assembly/model.xml.
SCREW_FULL_TURN = 2 * np.pi
SCREW_DEPTH = 0.02

# Materials declared in the lamp model.
LAMP_MATERIALS = ("base_mat", "bulb_mat")

# Surface finishes, as (specular, shininess, reflectance) ranges. This is the texture axis
# for these parts: the STLs have no UVs, so an image texture would map to the triangulation
# and read as blotches, whereas these three scalars need no UVs and cover the range that
# actually matters for printed plastic - matte PLA through to a polished/varnished finish.
LAMP_FINISHES = {
    "matte": ((0.02, 0.18), (0.02, 0.15), (0.0, 0.0)),
    "satin": ((0.25, 0.5), (0.2, 0.45), (0.0, 0.05)),
    "glossy": ((0.7, 1.0), (0.6, 0.95), (0.1, 0.3)),
}


def scale_screw_joint(obj):
    """
    Applies the object's scale to the parts of the screw that RoboCasa's scaling misses.

    MujocoXMLObject.set_scale calls scale_mjcf_model with scale_slide_joints=False, so the
    slide's travel keeps its unscaled range, and the joint-to-joint <equality> that defines
    the thread pitch lives outside the object body and is never visited at all. Both are
    lengths. Left alone, a scaled-up lamp would still descend only 20 mm over a turn while
    its socket got deeper, so the bulb would never look seated.

    The hinge range is deliberately untouched: a turn is a turn at any size.

    Args:
        obj (MJCFObject): freshly constructed lamp object, already scaled
    """
    scale = float(np.mean(obj._scale))
    if np.isclose(scale, 1.0):
        return

    slide = obj.worldbody.find(
        ".//joint[@name='{}screw_slide']".format(obj.naming_prefix)
    )
    slide.set("range", array_to_string(string_to_array(slide.get("range")) * scale))

    equality = obj.equality.find("joint")
    polycoef = string_to_array(equality.get("polycoef"))
    polycoef[1] *= scale
    equality.set("polycoef", array_to_string(polycoef))


class ScrewLightbulb(Kitchen):
    """
    Class encapsulating the light bulb screwing task.

    A table lamp stands on the counter with its bulb already seated in the socket but not
    screwed down. The robot has to turn the bulb until it is fully screwed in, which takes
    one full revolution and therefore several regrasps - a parallel jaw gripper cannot spin
    a full turn in one go.

    The lamp is a single articulated object: the bulb rides a hinge coupled to a slide, so
    the only way to drive it down is to rotate it. Note that the gripper has to press down
    while turning; holding the bulb at a fixed height locks the screw, since the bulb has
    to descend through the fingers as it turns.

    Args:
        turns_required (float): fraction of a full turn that counts as screwed in.

        scale_range (2-tuple or None): if set, the lamp is scaled by a factor drawn uniformly
            from this range each episode. Success is measured in turns, which is scale free,
            so the same threshold holds at every size.

        randomize_appearance (bool): if True, the lamp's materials get a fresh colour and
            surface finish (matte / satin / glossy) each episode.
    """

    def __init__(
        self,
        turns_required=0.95,
        scale_range=None,
        randomize_appearance=False,
        *args,
        **kwargs
    ):
        self.turns_required = turns_required
        self.scale_range = scale_range
        self.randomize_appearance = randomize_appearance
        # set before super().__init__, which resets and can ask for ep meta
        self._lamp_appearance = None
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the light bulb task. (Counter to stand the lamp on)
        """
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(0.5, 0.5))
        )
        self.init_robot_base_pos = self.counter

    def get_ep_meta(self):
        """
        Get the episode metadata for the light bulb task.
        This includes the language description of the task.

        Returns:
            dict: Episode metadata.
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "screw the light bulb into the lamp base"
        # record the sampled appearance so playback reproduces it (the object scale rides
        # along in object_cfgs, which RoboCasa already serializes)
        if self._lamp_appearance is not None:
            ep_meta["lamp_appearance"] = {
                name: dict(spec, rgba=list(spec["rgba"]))
                for name, spec in self._lamp_appearance.items()
            }
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the light bulb task. Stands the lamp on the
        counter, upright and with room around it for the gripper to approach and regrasp.

        Returns:
            list: List of object configurations.
        """
        cfg = dict(
            name="lamp",
            obj_groups="lamp_assembly",
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    ref=self.counter,
                ),
                size=(0.40, 0.40),
                pos=(0, -0.6),
                # the screw axis is vertical, so only yaw matters and the bulb is
                # rotationally symmetric - but vary it so policies cannot memorize
                rotation=(-np.pi / 6, np.pi / 6),
            ),
        )
        if self.scale_range is not None:
            # goes in the cfg (not applied here) so it lands in ep_meta and replays
            cfg["object_scale"] = float(
                self.rng.uniform(self.scale_range[0], self.scale_range[1])
            )
        return [cfg]

    def _create_obj(self, cfg):
        """
        Creates the lamp, then repairs the parts of the screw that object scaling misses.
        """
        obj, info = super()._create_obj(cfg)
        if cfg.get("name") == "lamp":
            scale_screw_joint(obj)
        return obj, info

    @property
    def screw_joint(self):
        """
        Name of the bulb's hinge joint. MJCFObject prefixes every joint with the object name.

        Returns:
            str: joint name
        """
        return "lamp_screw_hinge"

    def get_screw_state(self):
        """
        Reads how far the bulb has been screwed in.

        Depth is read straight off the slide joint rather than derived from the angle, so it
        stays correct when the lamp is scaled.

        Returns:
            dict: turns completed (0 to 1) and insertion depth in metres
        """
        model = self.sim.model
        angle = self.sim.data.qpos[
            model.jnt_qposadr[model.joint_name2id(self.screw_joint)]
        ]
        depth = -self.sim.data.qpos[
            model.jnt_qposadr[model.joint_name2id("lamp_screw_slide")]
        ]
        return dict(turns=-angle / SCREW_FULL_TURN, depth=depth)

    def _sample_lamp_appearance(self):
        """
        Draws a colour and a surface finish for each lamp material.

        Returns:
            dict: material name -> dict(rgba, finish, specular, shininess, reflectance)
        """
        appearance = {}
        for name in LAMP_MATERIALS:
            finish = list(LAMP_FINISHES)[self.rng.integers(len(LAMP_FINISHES))]
            if name == "bulb_mat":
                # keep the bulb glassy and near-white: only the base takes saturated colour
                value = self.rng.uniform(0.6, 1.0)
                rgba = [
                    value,
                    value,
                    self.rng.uniform(value, 1.0),
                    self.rng.uniform(0.45, 0.8),
                ]
                finish = "glossy" if finish == "matte" else finish
            else:
                rgba = list(self.rng.uniform(0.1, 0.9, size=3)) + [1.0]
            spec_r, shin_r, refl_r = LAMP_FINISHES[finish]
            appearance[name] = dict(
                rgba=rgba,
                finish=finish,
                specular=float(self.rng.uniform(*spec_r)),
                shininess=float(self.rng.uniform(*shin_r)),
                reflectance=float(self.rng.uniform(*refl_r)),
            )
        return appearance

    def _apply_lamp_appearance(self, appearance):
        """
        Writes an appearance onto the compiled model.

        These are all read out of mjModel each frame, so this takes effect without reloading
        the model or rebuilding the render context.

        Args:
            appearance (dict): material name -> dict(rgba, specular, shininess, reflectance)
        """
        model = self.sim.model._model
        prefix = self.objects["lamp"].naming_prefix
        for name, spec in appearance.items():
            mat_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_MATERIAL, prefix + name
            )
            if mat_id == -1:
                continue
            model.mat_rgba[mat_id] = spec["rgba"]
            model.mat_specular[mat_id] = spec["specular"]
            model.mat_shininess[mat_id] = spec["shininess"]
            model.mat_reflectance[mat_id] = spec["reflectance"]

    def _reset_internal(self):
        """
        Resets simulation internal configurations, then re-skins the lamp if asked.
        """
        super()._reset_internal()

        self._lamp_appearance = None
        if self.randomize_appearance:
            self._lamp_appearance = self._ep_meta.get("lamp_appearance")
            if self._lamp_appearance is None:
                self._lamp_appearance = self._sample_lamp_appearance()
            self._apply_lamp_appearance(self._lamp_appearance)

    def _check_success(self):
        """
        Check if the light bulb task is successful: the bulb is screwed all the way down
        and the gripper has let go of it.
        """
        screwed_in = self.get_screw_state()["turns"] >= self.turns_required
        gripper_obj_far = OU.gripper_obj_far(self, obj_name="lamp", th=0.15)
        return screwed_in and gripper_obj_far
