# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base piano composer task."""

from typing import Optional, Sequence

import mujoco
import numpy as np
from dm_control import composer
from mujoco_utils import composer_utils, physics_utils

from robopianist.models.hands import HandSide, daxian_hand, daxian_v2_hand, shadow_hand
from robopianist.models.hands import daxian_hand_constants as daxian_consts
from robopianist.models.hands import daxian_v2_hand_constants as daxian_v2_consts
from robopianist.models.hands import shadow_hand_constants as shadow_consts
from robopianist.models.piano import piano

# Timestep of the physics simulation, in seconds.
_PHYSICS_TIMESTEP = 0.005

# Interval between agent actions, in seconds.
_CONTROL_TIMESTEP = 0.05  # 20 Hz.

# Daxian attach pose lives in daxian_hand_constants so the preview cannot drift.
# Shadow uses the original RoboPianist paper pose.
_SHADOW_LEFT_HAND_POSITION = (0.4, -0.15, 0.13)
_SHADOW_RIGHT_HAND_POSITION = (0.4, 0.15, 0.13)
_SHADOW_HAND_QUATERNION = (-1, -1, 1, 1)


def _euler_rpy_deg_to_quat(rpy_deg) -> np.ndarray:
    """Convert XYZ Euler degrees to a MuJoCo (w, x, y, z) quaternion."""
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_euler2Quat(quat, np.radians(rpy_deg), "XYZ")
    return quat


def _quat_mul(q_left, q_right) -> np.ndarray:
    out = np.zeros(4, dtype=np.float64)
    mujoco.mju_mulQuat(out, np.asarray(q_left, dtype=np.float64), np.asarray(q_right, dtype=np.float64))
    return out


def _axis_angle_quat(axis, angle_rad: float) -> np.ndarray:
    out = np.zeros(4, dtype=np.float64)
    mujoco.mju_axisAngle2Quat(out, np.asarray(axis, dtype=np.float64), angle_rad)
    return out


def _hand_quat(consts_mod, spin_deg: float) -> np.ndarray:
    """Base Euler, then spin about the forearm local +Z (preview blue axis).

    PALM_PITCH_UP_DEG is a world-+Y rotation applied after that, so from the
    player's view the finger edge of the palm sits above the wrist.
    """
    q_base = _euler_rpy_deg_to_quat(consts_mod.HAND_BASE_RPY_DEG)
    q_spin = _axis_angle_quat((0.0, 0.0, 1.0), np.radians(spin_deg))
    q = _quat_mul(q_base, q_spin)
    pitch = float(consts_mod.PALM_PITCH_UP_DEG)
    if pitch:
        q_pitch = _axis_angle_quat((0.0, 1.0, 0.0), np.radians(pitch))
        q = _quat_mul(q_pitch, q)
    return q


_DAXIAN_LEFT_HAND_QUATERNION = _hand_quat(
    daxian_consts, daxian_consts.LEFT_FOREARM_Z_SPIN_DEG
)
_DAXIAN_RIGHT_HAND_QUATERNION = _hand_quat(
    daxian_consts, daxian_consts.RIGHT_FOREARM_Z_SPIN_DEG
)

_ATTACHMENT_YAW = 0  # Degrees.


class PianoOnlyTask(composer.Task):
    """Piano task with no hands."""

    def __init__(
        self,
        arena: composer_utils.Arena,
        change_color_on_activation: bool = False,
        add_piano_actuators: bool = False,
        physics_timestep: float = _PHYSICS_TIMESTEP,
        control_timestep: float = _CONTROL_TIMESTEP,
    ) -> None:
        self._arena = arena
        self._piano = piano.Piano(
            change_color_on_activation=change_color_on_activation,
            add_actuators=add_piano_actuators,
        )
        arena.attach(self._piano)

        # Harden the piano keys.
        # The default solref parameters are (0.02, 1). In particular, the first
        # parameter specifies -stiffness, and so decreasing it makes the contacts
        # harder. The documentation recommends keeping the stiffness at least 2x larger
        # than the physics timestep, see:
        # https://mujoco.readthedocs.io/en/latest/modeling.html?highlight=stiffness#solver-parameters
        self._piano.mjcf_model.default.geom.solref = (physics_timestep * 2, 1)

        self.set_timesteps(
            control_timestep=control_timestep, physics_timestep=physics_timestep
        )

    # Accessors.

    @property
    def root_entity(self):
        return self._arena

    @property
    def arena(self):
        return self._arena

    @property
    def piano(self) -> piano.Piano:
        return self._piano

    # Composer methods.

    def get_reward(self, physics) -> float:
        del physics  # Unused.
        return 0.0


class PianoTask(PianoOnlyTask):
    """Base class for piano tasks."""

    def __init__(
        self,
        arena: composer_utils.Arena,
        gravity_compensation: bool = False,
        change_color_on_activation: bool = False,
        primitive_fingertip_collisions: bool = False,
        reduced_action_space: bool = False,
        attachment_yaw: float = _ATTACHMENT_YAW,
        forearm_dofs: Optional[Sequence[str]] = None,
        physics_timestep: float = _PHYSICS_TIMESTEP,
        control_timestep: float = _CONTROL_TIMESTEP,
        robot: str = "daxian",
        unlock_four_finger_pip_dip: bool = False,
    ) -> None:
        super().__init__(
            arena=arena,
            change_color_on_activation=change_color_on_activation,
            add_piano_actuators=False,
            physics_timestep=physics_timestep,
            control_timestep=control_timestep,
        )

        name = (robot or "daxian").lower().replace("-", "_")
        if name in ("shadow", "shadow_hand"):
            self._robot = "shadow"
            self.hand_consts = shadow_consts
            hand_cls = shadow_hand.ShadowHand
            if forearm_dofs is None:
                forearm_dofs = shadow_hand._DEFAULT_FOREARM_DOFS
            left_pos = _SHADOW_LEFT_HAND_POSITION
            right_pos = _SHADOW_RIGHT_HAND_POSITION
            left_quat = _SHADOW_HAND_QUATERNION
            right_quat = _SHADOW_HAND_QUATERNION
        elif name in ("daxian_v2", "daxian2"):
            self._robot = "daxian_v2"
            self.hand_consts = daxian_v2_consts
            hand_cls = daxian_v2_hand.DaxianV2Hand
            if forearm_dofs is None:
                forearm_dofs = daxian_hand._DEFAULT_FOREARM_DOFS
            left_pos = daxian_v2_consts.LEFT_HAND_POSITION
            right_pos = daxian_v2_consts.RIGHT_HAND_POSITION
            left_quat = _hand_quat(
                daxian_v2_consts, daxian_v2_consts.LEFT_FOREARM_Z_SPIN_DEG
            )
            right_quat = _hand_quat(
                daxian_v2_consts, daxian_v2_consts.RIGHT_FOREARM_Z_SPIN_DEG
            )
        elif name in ("daxian", "daxian_v3") or name.startswith("daxian"):
            self._robot = "daxian"
            self.hand_consts = daxian_consts
            hand_cls = daxian_hand.DaxianHand
            if forearm_dofs is None:
                forearm_dofs = daxian_hand._DEFAULT_FOREARM_DOFS
            left_pos = daxian_consts.LEFT_HAND_POSITION
            right_pos = daxian_consts.RIGHT_HAND_POSITION
            left_quat = _DAXIAN_LEFT_HAND_QUATERNION
            right_quat = _DAXIAN_RIGHT_HAND_QUATERNION
        else:
            raise ValueError(
                f"Unknown robot {robot!r}. Use 'daxian', 'daxian_v2', or 'shadow'."
            )

        self._unlock_four_finger_pip_dip = bool(unlock_four_finger_pip_dip)

        self._right_hand = self._add_hand(
            hand_cls=hand_cls,
            hand_side=HandSide.RIGHT,
            position=right_pos,
            quaternion=right_quat,
            gravity_compensation=gravity_compensation,
            primitive_fingertip_collisions=primitive_fingertip_collisions,
            reduced_action_space=reduced_action_space,
            attachment_yaw=attachment_yaw,
            forearm_dofs=forearm_dofs,
        )
        self._left_hand = self._add_hand(
            hand_cls=hand_cls,
            hand_side=HandSide.LEFT,
            position=left_pos,
            quaternion=left_quat,
            gravity_compensation=gravity_compensation,
            primitive_fingertip_collisions=primitive_fingertip_collisions,
            reduced_action_space=reduced_action_space,
            attachment_yaw=attachment_yaw,
            forearm_dofs=forearm_dofs,
        )

    @property
    def robot(self) -> str:
        return self._robot

    @property
    def left_hand(self):
        return self._left_hand

    @property
    def right_hand(self):
        return self._right_hand

    def _add_hand(
        self,
        hand_cls,
        hand_side: HandSide,
        position,
        quaternion,
        gravity_compensation: bool,
        primitive_fingertip_collisions: bool,
        reduced_action_space: bool,
        attachment_yaw: float,
        forearm_dofs: Sequence[str],
    ):
        joint_range = [-self._piano.size[1], self._piano.size[1]]

        # Offset the joint range by the hand's initial position.
        joint_range[0] -= position[1]
        joint_range[1] -= position[1]

        hand_init = dict(
            side=hand_side,
            primitive_fingertip_collisions=primitive_fingertip_collisions,
            restrict_wrist_yaw_range=False,
            reduced_action_space=reduced_action_space,
            forearm_dofs=forearm_dofs,
        )
        if str(self._robot).startswith("daxian"):
            hand_init["unlock_four_finger_pip_dip"] = self._unlock_four_finger_pip_dip
        hand = hand_cls(**hand_init)
        hand.root_body.pos = position

        # Slightly rotate the forearms inwards (Z-axis) to mimic human posture.
        rotate_axis = np.asarray([0, 0, 1], dtype=np.float64)
        rotate_by = np.zeros(4, dtype=np.float64)
        sign = -1 if hand_side == HandSide.LEFT else 1
        angle = np.radians(sign * attachment_yaw)
        mujoco.mju_axisAngle2Quat(rotate_by, rotate_axis, angle)
        final_quaternion = np.zeros(4, dtype=np.float64)
        mujoco.mju_mulQuat(final_quaternion, rotate_by, quaternion)
        hand.root_body.quat = final_quaternion

        if gravity_compensation:
            physics_utils.compensate_gravity(hand.mjcf_model)

        # Override forearm translation joint range.
        forearm_tx_joint = hand.mjcf_model.find("joint", "forearm_tx")
        if forearm_tx_joint is not None:
            forearm_tx_joint.range = joint_range
        forearm_tx_actuator = hand.mjcf_model.find("actuator", "forearm_tx")
        if forearm_tx_actuator is not None:
            forearm_tx_actuator.ctrlrange = joint_range

        self._arena.attach(hand)
        return hand
