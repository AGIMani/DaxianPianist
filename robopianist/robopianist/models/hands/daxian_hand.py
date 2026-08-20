# Copyright 2023 The RoboPianist Authors.
# Adapted for Daxian hand.

"""Daxian hand composer class."""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from dm_control import composer, mjcf
from dm_control.composer.observation import observable
from dm_env import specs
from mujoco_utils import mjcf_utils, physics_utils, spec_utils, types

from robopianist.models.hands import base
from robopianist.models.hands import daxian_hand_constants as consts


@dataclass(frozen=True)
class Dof:
    """Forearm degree of freedom."""

    joint_type: str
    axis: Tuple[int, int, int]
    stiffness: float
    joint_range: Tuple[float, float]
    force_limit: float
    reflect: bool = False


# Axes below are expressed in the forearm body frame. After the 180° spin about
# local +Z the forearm maps to the world as:
#   +X -> -Z_world,  +Y -> -Y_world,  +Z -> -X_world
# Palm pitch (PALM_PITCH_UP_DEG about world +Y) is applied after that, so a
# naive local axis is no longer world-aligned. Slide DOFs therefore convert
# world XYZ through `_local_axis_for_world` at build time:
#   forearm_tx -> world +Y (along the keyboard)
#   forearm_ty -> world +Z (vertical; negative lowers the palm onto the keys)
#   forearm_tz -> world +X (toward the player; negative reaches into the keys)
# `reflect` is only used where the two hands should genuinely move as mirror images.
#
# Every actuator carries a force limit. Without one the position servos behave as
# infinitely strong constraints: an untrained policy commanding forearm_tx against a
# key wall draws >200 N and drives the hand ~12 mm through the keyboard, which both
# looks wrong and teaches the policy to exploit contact tunnelling.
_FOREARM_DOFS: Dict[str, Dof] = {
    # Slide along the keyboard (world +Y). Range is overridden per-hand in
    # PianoTask._add_hand to cover the piano width.
    "forearm_tx": Dof(
        joint_type="slide",
        axis=(0, -1, 0),
        stiffness=300,
        joint_range=(-1, 1),
        force_limit=20.0,
    ),
    # Vertical, world +Z after palm pitch. Negative lowers the palm onto the keys.
    "forearm_ty": Dof(
        joint_type="slide",
        axis=(-1, 0, 0),
        stiffness=300,
        joint_range=(-0.04, 0.06),
        force_limit=20.0,
    ),
    # Depth, world +X (toward the player). Negative reaches into the keyboard.
    "forearm_tz": Dof(
        joint_type="slide",
        axis=(0, 0, 1),
        stiffness=1000,
        joint_range=(-0.04, 0.0),
        force_limit=20.0,
    ),
    # Extra Euler-X / rpy-roll after the attach RPY. +ctrl = +rpy roll.
    # With ±180° Z-spin this is a hinge about forearm local -X (not world X
    # after pitch). No reflect: both hands share HAND_BASE_RPY_DEG.
    # Per-hand range is applied in _add_dofs (left [-0.5, 0], right [0, 0.5]).
    "forearm_roll": Dof(
        joint_type="hinge",
        axis=(-1, 0, 0),
        stiffness=300,
        joint_range=consts.LEFT_FOREARM_ROLL_RANGE,
        force_limit=5.0,
        reflect=False,
    ),
    # Pitch about the keyboard axis (world Y): tilts the fingers up/down.
    "forearm_pitch": Dof(
        joint_type="hinge",
        axis=(0, -1, 0),
        stiffness=50,
        joint_range=(-0.15, 0.15),
        force_limit=5.0,
    ),
    # Yaw about the vertical axis (world Z), mirrored so that positive turns each
    # hand outwards. Per-hand range is applied in _add_dofs (both [-0.6, 0];
    # axis reflect already Y-mirrors, so do not flip the right-hand numbers).
    "forearm_yaw": Dof(
        joint_type="hinge",
        axis=(-1, 0, 0),
        stiffness=300,
        joint_range=(-0.6, 0.0),
        force_limit=5.0,
        reflect=True,
    ),
}

# No forearm_ty: vertical press comes from MCP/swing at a lowered attach z.
# forearm_roll is extra rpy-roll (Euler X), policy-commanded.
_DEFAULT_FOREARM_DOFS = ("forearm_tx", "forearm_roll")
_POLICY_EXCLUDED_ACTUATORS: frozenset = frozenset()


class DaxianHand(base.Hand):
    """Daxian anthropomorphic hand for RoboPianist."""

    def _build(
        self,
        name: Optional[str] = None,
        side: base.HandSide = base.HandSide.RIGHT,
        primitive_fingertip_collisions: bool = False,
        restrict_wrist_yaw_range: bool = False,
        reduced_action_space: bool = False,
        unlock_four_finger_pip_dip: bool = False,
        forearm_dofs: Sequence[str] = _DEFAULT_FOREARM_DOFS,
        consts_module=None,
    ) -> None:
        del restrict_wrist_yaw_range  # Daxian has no WRJ2 analogue.

        self._c = consts_module or consts

        if side == base.HandSide.RIGHT:
            self._prefix = "rh_"
            xml_file = self._c.RIGHT_DAXIAN_HAND_XML
        elif side == base.HandSide.LEFT:
            self._prefix = "lh_"
            xml_file = self._c.LEFT_DAXIAN_HAND_XML
        else:
            raise ValueError(f"Unsupported hand side: {side}")

        name = name or self._prefix + "daxian_hand"
        self._hand_side = side
        self._mjcf_root = mjcf.from_path(str(xml_file))
        self._mjcf_root.model = name
        self._n_forearm_dofs = 0
        self._reduce_action_space = reduced_action_space
        self._forearm_dofs = forearm_dofs
        self._pinned_joints = (
            frozenset(self._c.pinned_joints(unlock_four_finger_pip_dip))
            if reduced_action_space
            else frozenset()
        )
        # None = couple every four-finger (preview / no MIDI gate).
        # frozenset = MIDI-assigned fingers only; others idle (PIP rest, DIP 0).
        self._pip_dip_active_fingers: Optional[frozenset] = None
        self._weld_joints()

        # Important: both calls must happen before parsing, and the mass must be set
        # before the DOFs so their critical damping is computed against it.
        self._add_forearm_inertia()
        self._add_dofs()
        self._parse_mjcf_elements()
        self._add_mjcf_elements()
        self._configure_thumb()
        self._disable_palm_collision()

        if primitive_fingertip_collisions:
            self._add_fingertip_collision_spheres()

        self._action_spec = None

    def _build_observables(self) -> "DaxianHandObservables":
        return DaxianHandObservables(self)

    @staticmethod
    def _joint_key(actuator_name: str) -> str:
        name = actuator_name
        for prefix in ("rh_A_", "lh_A_", "rh_", "lh_", "A_"):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        return name

    def _weld_joints(self) -> None:
        """Drop listed hinge joints so the child body is welded at q=0."""
        keys = frozenset(getattr(self._c, "WELD_JOINTS", ()))
        if not keys:
            return
        for joint in list(mjcf_utils.safe_find_all(self._mjcf_root, "joint")):
            if self._joint_key(joint.name) in keys:
                joint.remove()
        for actuator in list(mjcf_utils.safe_find_all(self._mjcf_root, "actuator")):
            if self._joint_key(actuator.name) in keys:
                actuator.remove()

    def _is_policy_actuator(self, actuator) -> bool:
        key = self._joint_key(actuator.name)
        if key in _POLICY_EXCLUDED_ACTUATORS:
            return False
        if self._reduce_action_space and key in self._pinned_joints:
            return False
        return True

    def _parse_mjcf_elements(self) -> None:
        joints = mjcf_utils.safe_find_all(self._mjcf_root, "joint")
        actuators = mjcf_utils.safe_find_all(self._mjcf_root, "actuator")
        self._joints = tuple(joints)
        self._actuators = tuple(a for a in actuators if self._is_policy_actuator(a))
        self._task_actuators = tuple(
            a for a in actuators if not self._is_policy_actuator(a)
        )

        # The generated MJCF leaves the finger servos unlimited; cap them to a torque a
        # hobby servo of this size could plausibly produce.
        for actuator in actuators:
            if actuator.name in self._forearm_dofs:
                continue
            actuator.forcerange = (
                -self._c.FINGER_FORCE_LIMIT,
                self._c.FINGER_FORCE_LIMIT,
            )
        self._clamp_four_finger_mcp()

    def _configure_thumb(self) -> None:
        """Stop the thumb drive housing from tunnelling through the palm.

        The URDF mounts `thumb_rota_block` / `thumb_rotaback` so their collision
        meshes overlap the palm by ~9 mm at q=0. MuJoCo then tries to separate them
        and the thumb looks like it is being driven through the hand. Those two
        links become visual-only and palm contact is excluded. Joint / actuator
        ranges are the MJCF limits in `THUMB_ROTA_BLOCK_RANGE` /
        `THUMB_ROTABACK_RANGE` (must contain `THUMB_REST_CTRL`).
        """
        for body_name in getattr(self._c, "THUMB_HOUSING_BODIES", ()):
            body = mjcf_utils.safe_find(
                self._mjcf_root, "body", self._prefix + body_name
            )
            # Direct geoms only. find_all() would also hit MCP/PIP descendants and
            # strip the fingertip collision sphere.
            for geom in list(body.geom):
                if geom.contype == 0:
                    continue
                geom.contype = 0
                geom.conaffinity = 0

        for body_a, body_b in getattr(self._c, "THUMB_PALM_EXCLUDE", ()):
            self._mjcf_root.contact.add(
                "exclude",
                body1=self._prefix + body_a,
                body2=self._prefix + body_b,
            )

        range_joints = getattr(
            self._c,
            "THUMB_RANGE_JOINTS",
            ("thumb_rota_block_joint", "thumb_rotaback_joint2"),
        )
        for joint_name in range_joints:
            lo_hi = self._c.thumb_policy_range(joint_name)
            joint = mjcf_utils.safe_find(
                self._mjcf_root, "joint", self._prefix + joint_name
            )
            joint.range = lo_hi
            actuator = mjcf_utils.safe_find(
                self._mjcf_root, "actuator", self._prefix + "A_" + joint_name
            )
            actuator.ctrlrange = lo_hi

    def _clamp_four_finger_mcp(self) -> None:
        """Optionally shrink four-finger MCP joint/ctrl range (V2 training)."""
        rng = getattr(self._c, "FOUR_FINGER_MCP_RANGE", None)
        if rng is None:
            return
        lo, hi = float(rng[0]), float(rng[1])
        for joint in self._joints:
            key = self._joint_key(joint.name)
            if key.endswith("MCP_joint") and not key.startswith("thumb_"):
                joint.range = (lo, hi)
        for actuator in (*self._actuators, *self._task_actuators):
            key = self._joint_key(actuator.name)
            if key.endswith("MCP_joint") and not key.startswith("thumb_"):
                actuator.ctrlrange = (lo, hi)

    def _disable_palm_collision(self) -> None:
        """Drop rest-penetrating palm colliders so finger–finger contacts can block."""
        for body_name in getattr(self._c, "PALM_VISUAL_ONLY_BODIES", ()):
            body = mjcf_utils.safe_find(
                self._mjcf_root, "body", self._prefix + body_name
            )
            for geom in list(body.geom):
                if geom.contype == 0:
                    continue
                geom.contype = 0
                geom.conaffinity = 0

    def _mirror_y(
        self, pos: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Constants are given for the right hand; the left hand is its Y-mirror."""
        x, y, z = pos
        return (x, -y, z) if self._hand_side == base.HandSide.LEFT else (x, y, z)

    def _fingertip_contact_pos(self, tip_name: str) -> Tuple[float, float, float]:
        return self._mirror_y(self._c.FINGERTIP_SITE_POS[tip_name])

    def _add_fingertip_collision_spheres(self) -> None:
        """Replace each distal collision mesh with a palmar pad (sphere or capsule)."""
        kind = getattr(self._c, "FINGERTIP_COLLISION_TYPE", "sphere")
        if kind == "mesh":
            # V2 piano-tip meshes already include the distal capsule.
            return
        radius = float(self._c.FINGERTIP_COLLISION_RADIUS)
        half = float(getattr(self._c, "FINGERTIP_CAPSULE_HALF_LENGTH", 0.0))
        solref = getattr(self._c, "FINGERTIP_SOLREF", None)
        solimp = getattr(self._c, "FINGERTIP_SOLIMP", None)
        for tip_name, rgb in zip(self._c.FINGERTIP_BODIES, self._c.FINGERTIP_COLORS):
            body = mjcf_utils.safe_find(
                self._mjcf_root, "body", self._prefix + tip_name
            )
            pos = self._mirror_y(self._c.FINGERTIP_COLLISION_POS[tip_name])
            rgba = rgb + (0.85,)
            geom_kwargs = dict(
                contype=1,
                conaffinity=1,
                group=0,
                rgba=rgba,
                pos=pos,
            )
            if kind == "capsule":
                quat_fn = getattr(self._c, "fingertip_capsule_quat_mujoco", None)
                geom_kwargs.update(
                    type="capsule",
                    size=(radius, half),
                    quat=quat_fn(tip_name) if quat_fn else (1.0, 0.0, 0.0, 0.0),
                )
            else:
                geom_kwargs.update(type="sphere", size=(radius, 0, 0))
            if solref is not None:
                geom_kwargs["solref"] = solref
            if solimp is not None:
                geom_kwargs["solimp"] = solimp

            converted = False
            for geom in list(body.geom):
                # Visual-only mesh (contype=conaffinity=0). Keep it.
                if geom.contype == 0 and geom.conaffinity == 0:
                    continue
                geom.mesh = None
                for key, val in geom_kwargs.items():
                    setattr(geom, key, val)
                converted = True
            if not converted:
                body.add("geom", name=tip_name + "_collision", **geom_kwargs)

    def set_palm_roll(self, physics: mjcf.Physics, angle_rad: float) -> None:
        """Set the task-driven forearm_roll target (positive = thumb side down)."""
        for actuator in self._task_actuators:
            if actuator.name == "forearm_roll":
                lo, hi = np.asarray(physics.bind(actuator).ctrlrange).reshape(-1)[:2]
                physics.bind(actuator).ctrl = float(np.clip(angle_rad, float(lo), float(hi)))
                return

    def _add_mjcf_elements(self) -> None:
        # Drop pre-baked tip sites from the XML so we own naming.
        for tip_name in self._c.FINGERTIP_BODIES:
            tip_elem = mjcf_utils.safe_find(
                self._mjcf_root, "body", self._prefix + tip_name
            )
            for site in list(tip_elem.find_all("site")):
                site.remove()

        fingertip_sites = []
        for tip_name in self._c.FINGERTIP_BODIES:
            tip_elem = mjcf_utils.safe_find(
                self._mjcf_root, "body", self._prefix + tip_name
            )
            tip_site = tip_elem.add(
                "site",
                name=tip_name + "_site",
                pos=self._fingertip_contact_pos(tip_name),
                type="sphere",
                size=(0.004,),
                group=composer.SENSOR_SITES_GROUP,
            )
            fingertip_sites.append(tip_site)
        self._fingertip_sites = tuple(fingertip_sites)

        joint_torque_sensors = []
        for joint_elem in self._joints:
            site_elem = joint_elem.parent.add(
                "site",
                name=joint_elem.name + "_site",
                size=(0.001, 0.001, 0.001),
                type="box",
                rgba=(0, 1, 0, 1),
                group=composer.SENSOR_SITES_GROUP,
            )
            torque_sensor_elem = joint_elem.root.sensor.add(
                "torque",
                site=site_elem,
                name=joint_elem.name + "_torque",
            )
            joint_torque_sensors.append(torque_sensor_elem)
        self._joint_torque_sensors = tuple(joint_torque_sensors)

        actuator_velocity_sensors = []
        actuator_force_sensors = []
        for actuator_elem in self._actuators:
            velocity_sensor_elem = self._mjcf_root.sensor.add(
                "actuatorvel",
                actuator=actuator_elem,
                name=actuator_elem.name + "_velocity",
            )
            actuator_velocity_sensors.append(velocity_sensor_elem)
            force_sensor_elem = self._mjcf_root.sensor.add(
                "actuatorfrc",
                actuator=actuator_elem,
                name=actuator_elem.name + "_force",
            )
            actuator_force_sensors.append(force_sensor_elem)
        self._actuator_velocity_sensors = tuple(actuator_velocity_sensors)
        self._actuator_force_sensors = tuple(actuator_force_sensors)

        fingertip_touch_sensors = []
        for tip_name in self._c.FINGERTIP_BODIES:
            tip_elem = mjcf_utils.safe_find(
                self._mjcf_root, "body", self._prefix + tip_name
            )
            touch_site = tip_elem.add(
                "site",
                name=tip_name + "_touch_site",
                pos=self._fingertip_contact_pos(tip_name),
                type="sphere",
                size=(0.01,),
                group=composer.SENSOR_SITES_GROUP,
                rgba=(0, 1, 0, 0.6),
            )
            touch_sensor = self._mjcf_root.sensor.add(
                "touch",
                site=touch_site,
                name=tip_name + "_touch",
            )
            fingertip_touch_sensors.append(touch_sensor)
        self._fingertip_touch_sensors = tuple(fingertip_touch_sensors)

    def _add_forearm_inertia(self) -> None:
        """Give the virtual forearm mount the inertia of the arm it stands in for.

        The exported URDF weighs only 113 g, and the mount body itself is massless, so
        the forearm servos accelerate the whole hand to nearly 2 m/s within one control
        step. At that speed the wrist advances ~9 mm per physics step and tunnels
        straight through the keys. Loading the mount with a plausible forearm mass keeps
        the same servo gains but bounds the speed to something a hand can actually move
        at.
        """
        inertial = self.root_body.inertial
        if inertial is None:
            inertial = self.root_body.add(
                "inertial",
                pos=(0, 0, 0),
                mass=self._c.FOREARM_MASS,
                diaginertia=(self._c.FOREARM_INERTIA,) * 3,
            )
            return
        inertial.mass = self._c.FOREARM_MASS
        inertial.diaginertia = (self._c.FOREARM_INERTIA,) * 3
        inertial.fullinertia = None

    def _local_axis_for_world(self, world_xyz: Sequence[float]) -> Tuple[float, float, float]:
        """Forearm-local slide axis that moves the hand along ``world_xyz``.

        Matches ``suite.tasks.base._hand_quat``: RPY, then Z-spin, then palm pitch
        about world +Y. Left/right 180° spins are the same rotation, so one axis
        works for both hands.
        """
        pitch = np.radians(float(getattr(self._c, "PALM_PITCH_UP_DEG", 0.0)))
        # Local -> world before palm pitch (see module comment).
        r0 = np.array(
            [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        c, s = np.cos(pitch), np.sin(pitch)
        ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
        axis = (ry @ r0).T @ np.asarray(world_xyz, dtype=np.float64)
        return tuple(float(x) for x in axis)

    def _add_dofs(self) -> None:
        def _maybe_reflect_axis(
            axis: Sequence[float], reflect: bool
        ) -> Sequence[float]:
            if self._hand_side == base.HandSide.LEFT and reflect:
                return tuple([-a for a in axis])
            return axis

        world_axis = {
            "forearm_tx": (0.0, 1.0, 0.0),  # along keyboard
            "forearm_ty": (0.0, 0.0, 1.0),  # vertical
            "forearm_tz": (1.0, 0.0, 0.0),  # toward the player
            "forearm_yaw": (0.0, 0.0, 1.0),  # yaw about world +Z
        }
        # With yaw enabled, roll must not also be world Z (local -X at pitch
        # -90°). Use world +X so pronation and yaw stay independent.
        if "forearm_yaw" in self._forearm_dofs:
            world_axis["forearm_roll"] = (1.0, 0.0, 0.0)
        for dof_name in self._forearm_dofs:
            if dof_name not in _FOREARM_DOFS:
                raise ValueError(
                    f"Invalid forearm DOF: {dof_name}. Valid DOFs are: "
                    f"{tuple(_FOREARM_DOFS)}."
                )
            dof = _FOREARM_DOFS[dof_name]
            joint_range = dof.joint_range
            if dof_name == "forearm_roll":
                joint_range = (
                    self._c.LEFT_FOREARM_ROLL_RANGE
                    if self._hand_side == base.HandSide.LEFT
                    else self._c.RIGHT_FOREARM_ROLL_RANGE
                )
            if dof_name == "forearm_yaw":
                if self._hand_side == base.HandSide.LEFT:
                    joint_range = getattr(
                        self._c,
                        "LEFT_FOREARM_YAW_RANGE",
                        getattr(self._c, "FOREARM_YAW_RANGE", joint_range),
                    )
                else:
                    joint_range = getattr(
                        self._c,
                        "RIGHT_FOREARM_YAW_RANGE",
                        getattr(self._c, "FOREARM_YAW_RANGE", joint_range),
                    )
            tz_range = getattr(self._c, "FOREARM_TZ_RANGE", None)
            if dof_name == "forearm_tz" and tz_range is not None:
                joint_range = tz_range
            ty_range = getattr(self._c, "FOREARM_TY_RANGE", None)
            if dof_name == "forearm_ty" and ty_range is not None:
                joint_range = ty_range
            if dof_name in world_axis:
                axis = self._local_axis_for_world(world_axis[dof_name])
                if dof.reflect:
                    axis = _maybe_reflect_axis(axis, True)
            else:
                axis = _maybe_reflect_axis(dof.axis, dof.reflect)
            joint = self.root_body.add(
                "joint",
                type=dof.joint_type,
                name=dof_name,
                axis=axis,
                range=joint_range,
            )
            joint.damping = physics_utils.get_critical_damping_from_stiffness(
                dof.stiffness, joint.full_identifier, self.mjcf_model
            )
            self._mjcf_root.actuator.add(
                "position",
                name=dof_name,
                joint=joint,
                ctrlrange=joint_range,
                kp=dof.stiffness,
                forcerange=(-dof.force_limit, dof.force_limit),
            )
            self._n_forearm_dofs += 1

    @property
    def hand_side(self) -> base.HandSide:
        return self._hand_side

    @property
    def mjcf_model(self) -> types.MjcfRootElement:
        return self._mjcf_root

    @property
    def name(self) -> str:
        return self._mjcf_root.model

    @property
    def n_forearm_dofs(self) -> int:
        return self._n_forearm_dofs

    @composer.cached_property
    def root_body(self) -> types.MjcfElement:
        return mjcf_utils.safe_find(self._mjcf_root, "body", self._prefix + "forearm")

    @composer.cached_property
    def fingertip_bodies(self) -> Sequence[types.MjcfElement]:
        return tuple(
            mjcf_utils.safe_find(self._mjcf_root, "body", self._prefix + name)
            for name in self._c.FINGERTIP_BODIES
        )

    @property
    def joints(self) -> Sequence[types.MjcfElement]:
        return self._joints

    @property
    def actuators(self) -> Sequence[types.MjcfElement]:
        return self._actuators

    @property
    def joint_torque_sensors(self) -> Sequence[types.MjcfElement]:
        return self._joint_torque_sensors

    @property
    def fingertip_sites(self) -> Sequence[types.MjcfElement]:
        return self._fingertip_sites

    @property
    def actuator_velocity_sensors(self) -> Sequence[types.MjcfElement]:
        return self._actuator_velocity_sensors

    @property
    def actuator_force_sensors(self) -> Sequence[types.MjcfElement]:
        return self._actuator_force_sensors

    @property
    def fingertip_touch_sensors(self) -> Sequence[types.MjcfElement]:
        return self._fingertip_touch_sensors

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        if self._action_spec is None:
            self._action_spec = spec_utils.create_action_spec(
                physics=physics, actuators=self.actuators, prefix=self.name
            )
        return self._action_spec

    def _qpos_name(self, joint_key: str) -> str:
        return f"{self.name}/{self._prefix}{joint_key}"

    def _body_name(self, body_key: str) -> str:
        return f"{self.name}/{self._prefix}{body_key}"

    def _four_finger_mcp_ctrl(self, physics: mjcf.Physics) -> dict:
        mcp = {}
        for actuator in self._actuators:
            key = self._joint_key(actuator.name)
            if key.endswith("MCP_joint") and not key.startswith("thumb_"):
                finger = key[: -len("_MCP_joint")]
                mcp[finger] = float(
                    np.asarray(physics.bind(actuator).ctrl).reshape(-1)[0]
                )
        return mcp

    def _ik_flex_remaining(self, physics: mjcf.Physics, mcp: dict) -> dict:
        """PIP+DIP needed so DIP +Z aligns with world -Z, at the commanded MCP."""
        couple_mod = self._c
        qpos0 = np.array(physics.data.qpos, copy=True)
        remaining = {}
        try:
            for finger, mcp_val in mcp.items():
                physics.named.data.qpos[self._qpos_name(f"{finger}_MCP_joint")] = (
                    mcp_val
                )
                physics.named.data.qpos[self._qpos_name(f"{finger}_PIP_joint")] = 0.0
                physics.named.data.qpos[self._qpos_name(f"{finger}_DIP_joint")] = 0.0
            physics.forward()
            target = getattr(
                couple_mod, "FOUR_FINGER_IK_TARGET_WORLD", (0.0, 0.0, -1.0)
            )
            for finger in mcp:
                rd = np.array(
                    physics.named.data.xmat[self._body_name(f"{finger}_DIP_link")]
                ).reshape(3, 3)
                rm = np.array(
                    physics.named.data.xmat[self._body_name(f"{finger}_MCP_link")]
                ).reshape(3, 3)
                remaining[finger] = float(
                    couple_mod.signed_flex_to_align(rd[:, 2], rm[:, 1], target)
                )
        finally:
            physics.data.qpos[:] = qpos0
            physics.forward()
        return remaining

    def set_pip_dip_active_fingers(self, fingers: Optional[Sequence[str]]) -> None:
        """Restrict PIP/DIP IK coupling to these four-finger names, or all if None."""
        if fingers is None:
            self._pip_dip_active_fingers = None
        else:
            self._pip_dip_active_fingers = frozenset(fingers)

    def _pin_fixed_fingers(self, physics: mjcf.Physics) -> None:
        """Hold pinned joints at rest, or IK PIP/DIP for MIDI-assigned fingers."""
        if not self._reduce_action_space:
            return
        rest = self._c.rest_ctrl()
        coupled = {}
        couple_fn = getattr(self._c, "couple_four_finger_pip_dip", None)
        idle_fn = getattr(self._c, "idle_four_finger_pip_dip", None)
        if (
            couple_fn is not None
            and getattr(self._c, "COUPLE_PIP_DIP_TO_MCP", False)
        ):
            mcp = self._four_finger_mcp_ctrl(physics)
            active = self._pip_dip_active_fingers
            if active is not None:
                mcp = {f: v for f, v in mcp.items() if f in active}
            flex = None
            if hasattr(self._c, "signed_flex_to_align") and mcp:
                flex = self._ik_flex_remaining(physics, mcp)
            coupled = couple_fn(mcp, rest, flex_remaining=flex) if mcp else {}
            if active is not None:
                idle = idle_fn(rest) if idle_fn is not None else {}
                for finger in getattr(self._c, "FOUR_FINGERS", ()):
                    if finger in active:
                        continue
                    pip_key = f"{finger}_PIP_joint"
                    dip_key = f"{finger}_DIP_joint"
                    coupled[pip_key] = float(
                        idle.get(pip_key, rest.get(pip_key, 0.0))
                    )
                    coupled[dip_key] = float(idle.get(dip_key, 0.0))
        for actuator in self._task_actuators:
            key = self._joint_key(actuator.name)
            if key not in self._pinned_joints:
                continue
            if key in coupled:
                physics.bind(actuator).ctrl = float(coupled[key])
            else:
                physics.bind(actuator).ctrl = float(rest.get(key, 0.0))

    def initialize_episode(self, physics, random_state) -> None:
        del random_state
        rest = self._c.rest_ctrl()
        for joint in self.joints:
            for name, val in rest.items():
                if joint.name.endswith(name):
                    physics.bind(joint).qpos = float(val)
                    break
        for actuator in (*self._actuators, *self._task_actuators):
            for name, val in rest.items():
                if actuator.name.endswith(name):
                    physics.bind(actuator).ctrl = float(val)
                    break
        # Four-finger MCP is policy-driven and not a pinned rest joint: idle is
        # URDF straight 0, not the [0, 1.57] midpoint. Thumb MCP is also policy
        # on V2 and is initialized from rest_ctrl() above.
        for joint in self.joints:
            if joint.name.endswith("MCP_joint") and "thumb_" not in joint.name:
                physics.bind(joint).qpos = 0.0
        for actuator in self._actuators:
            if actuator.name.endswith("MCP_joint") and "thumb_" not in actuator.name:
                physics.bind(actuator).ctrl = 0.0
        # Unlocked four-finger PIP/DIP: CanonicalSpec 0 = straight, so init
        # there too. Pinned PIP/DIP are IK-solved from MCP in _pin_fixed_fingers.
        for joint in self.joints:
            key = self._joint_key(joint.name)
            if key in self._pinned_joints:
                continue
            if (key.endswith("PIP_joint") or key.endswith("DIP_joint")) and (
                not key.startswith("thumb_")
            ):
                physics.bind(joint).qpos = 0.0
        for actuator in self._actuators:
            key = self._joint_key(actuator.name)
            if key in self._pinned_joints:
                continue
            if (key.endswith("PIP_joint") or key.endswith("DIP_joint")) and (
                not key.startswith("thumb_")
            ):
                physics.bind(actuator).ctrl = 0.0
        self._pin_fixed_fingers(physics)

    def apply_action(
        self,
        physics: mjcf.Physics,
        action: np.ndarray,
        random_state: np.random.RandomState,
    ) -> None:
        del random_state
        physics.bind(self.actuators).ctrl = action
        self._pin_fixed_fingers(physics)


class DaxianHandObservables(base.HandObservables):
    """DaxianHand observables."""

    _entity: DaxianHand

    @composer.observable
    def actuators_force(self):
        return observable.MJCFFeature("sensordata", self._entity.actuator_force_sensors)

    @composer.observable
    def actuators_velocity(self):
        return observable.MJCFFeature(
            "sensordata", self._entity.actuator_velocity_sensors
        )

    @composer.observable
    def actuators_power(self):
        def _get_actuator_power(physics: mjcf.Physics) -> np.ndarray:
            force = physics.bind(self._entity.actuator_force_sensors).sensordata
            velocity = physics.bind(self._entity.actuator_velocity_sensors).sensordata
            return abs(force) * abs(velocity)

        return observable.Generic(raw_observation_callable=_get_actuator_power)

    @composer.observable
    def fingertip_positions(self):
        def _get_fingertip_positions(physics: mjcf.Physics) -> np.ndarray:
            return physics.bind(self._entity.fingertip_sites).xpos.ravel()

        return observable.Generic(raw_observation_callable=_get_fingertip_positions)

    @composer.observable
    def fingertip_force(self):
        def _get_fingertip_force(physics: mjcf.Physics) -> np.ndarray:
            return physics.bind(self._entity.fingertip_touch_sensors).sensordata

        return observable.Generic(raw_observation_callable=_get_fingertip_force)
