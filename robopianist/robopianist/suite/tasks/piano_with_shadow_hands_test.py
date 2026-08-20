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

"""Tests for piano_with_shadow_hands_test.py."""

import itertools
import math
from typing import Optional

import numpy as np
from absl.testing import absltest, parameterized
from dm_control import composer
from mujoco_utils import spec_utils
from note_seq.protobuf import music_pb2

from robopianist.models.hands import daxian_v2_hand_constants as v2c
from robopianist.music import midi_file
from robopianist.suite.tasks import piano_with_shadow_hands


def _get_test_midi(dt: float = 0.01) -> midi_file.MidiFile:
    seq = music_pb2.NoteSequence()

    # C6 for 2 dts.
    seq.notes.add(
        start_time=0.0,
        end_time=2 * dt,
        velocity=80,
        pitch=midi_file.note_name_to_midi_number("C6"),
        part=1,  # Right hand index.
    )
    # G5 for 1 dt.
    seq.notes.add(
        start_time=2 * dt,
        end_time=3 * dt,
        velocity=80,
        pitch=midi_file.note_name_to_midi_number("G5"),
        part=0,  # Left hand thumb.
    )

    seq.total_time = 3 * dt
    seq.tempos.add(qpm=60)
    return midi_file.MidiFile(seq=seq)


def _get_env(
    control_timestep: float = 0.01,
    n_steps_lookahead: int = 0,
    n_seconds_lookahead: Optional[float] = None,
    wrong_press_termination: bool = False,
    disable_fingering_reward: bool = False,
) -> composer.Environment:
    task = piano_with_shadow_hands.PianoWithShadowHands(
        midi=_get_test_midi(dt=control_timestep),
        n_steps_lookahead=n_steps_lookahead,
        n_seconds_lookahead=n_seconds_lookahead,
        control_timestep=control_timestep,
        wrong_press_termination=wrong_press_termination,
        change_color_on_activation=True,
        disable_fingering_reward=disable_fingering_reward,
    )
    return composer.Environment(task, strip_singleton_obs_buffer_dim=True)


class PianoWithShadowHandsTest(parameterized.TestCase):
    @parameterized.parameters(True, False)
    def test_observables(self, disable_fingering_reward: bool) -> None:
        env = _get_env(disable_fingering_reward=disable_fingering_reward)
        timestep = env.reset()

        # Piano observables.
        self.assertIn("piano/state", timestep.observation)
        self.assertIn("piano/sustain_state", timestep.observation)

        # Goal observables.
        self.assertIn("goal", timestep.observation)
        if disable_fingering_reward:
            self.assertNotIn("fingering", timestep.observation)
        else:
            self.assertIn("fingering", timestep.observation)

        # Hand observables.
        for name in ["rh_shadow_hand", "lh_shadow_hand"]:
            self.assertIn(f"{name}/joints_pos", timestep.observation)
            # self.assertIn(f"{name}/position", timestep.observation)

    def test_action_spec(self) -> None:
        env = _get_env()
        rh_action_spec = env.task.right_hand.action_spec(env.physics)
        lh_action_spec = env.task.left_hand.action_spec(env.physics)
        combined_spec = spec_utils.merge_specs([rh_action_spec, lh_action_spec])
        actual_shape = env.action_spec().shape[0] - 1  # Don't include sustain pedal.
        expected_shape = combined_spec.shape[0]
        self.assertEqual(actual_shape, expected_shape)

        right_action = np.random.uniform(
            low=rh_action_spec.minimum, high=rh_action_spec.maximum
        ).astype(rh_action_spec.dtype)
        left_action = np.random.uniform(
            low=lh_action_spec.minimum, high=lh_action_spec.maximum
        ).astype(lh_action_spec.dtype)
        action = np.concatenate([right_action, left_action, [0]])
        env.task.before_step(env.physics, action, env.random_state)

        actual_rh_action = env.physics.bind(env.task.right_hand.actuators).ctrl
        np.testing.assert_array_equal(actual_rh_action, right_action)
        actual_lh_action = env.physics.bind(env.task.left_hand.actuators).ctrl
        np.testing.assert_array_equal(actual_lh_action, left_action)

    def test_termination_and_discount(self) -> None:
        env = _get_env()
        action_spec = env.action_spec()
        env.reset()

        # With a dt of 0.01 and a 3 dt long midi, the episode should end after 4 steps.
        zero_action = np.zeros(action_spec.shape)
        for _ in range(3):
            timestep = env.step(zero_action)
            self.assertFalse(env.task.should_terminate_episode(env.physics))
            np.testing.assert_array_equal(env.task.get_discount(env.physics), 1.0)

        # 1 more step to terminate.
        timestep = env.step(zero_action)
        self.assertTrue(timestep.last())
        self.assertTrue(env.task.should_terminate_episode(env.physics))
        # No failure, so discount should be 1.0.
        np.testing.assert_array_equal(env.task.get_discount(env.physics), 1.0)

    @parameterized.parameters(itertools.product([0.01, 0.05, 0.1], [0, 0.01, 0.1, 1]))
    def test_n_seconds_lookahead(
        self, control_timestep: float, n_seconds_lookahead: float
    ) -> None:
        env = _get_env(
            control_timestep=control_timestep, n_seconds_lookahead=n_seconds_lookahead
        )

        actual_n_steps_lookahead = env.task._n_steps_lookahead
        expected_n_steps_lookahead = int(
            np.ceil(n_seconds_lookahead / control_timestep)
        )
        self.assertEqual(actual_n_steps_lookahead, expected_n_steps_lookahead)

    @parameterized.parameters(0, 1, 2, 5)
    def test_goal_observable_lookahead(self, n_steps_lookahead: int) -> None:
        env = _get_env(control_timestep=0.01, n_steps_lookahead=n_steps_lookahead)
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape)
        timestep = env.reset()

        midi = _get_test_midi(dt=0.01)
        note_traj = midi_file.NoteTrajectory.from_midi(
            midi, dt=env.task.control_timestep
        )
        notes = note_traj.notes
        sustains = note_traj.sustains
        self.assertLen(notes, 4)

        for i in range(len(notes)):
            expected_goal = np.zeros((n_steps_lookahead + 1, env.task.piano.n_keys + 1))

            t_start = i
            t_end = min(i + n_steps_lookahead + 1, len(notes))
            for j, t in enumerate(range(t_start, t_end)):
                keys = [note.key for note in notes[t]]
                expected_goal[j, keys] = 1.0
                expected_goal[j, -1] = sustains[t]

            actual_goal = timestep.observation["goal"]
            np.testing.assert_array_equal(actual_goal, expected_goal.ravel())

            # Check that the 0th goal is always the goal at the current timestep.
            expected_current = np.zeros((env.task.piano.n_keys + 1,))
            keys = [note.key for note in notes[i]]
            expected_current[keys] = 1.0
            expected_current[-1] = sustains[i]
            actual_current = timestep.observation["goal"][0 : env.task.piano.n_keys + 1]
            np.testing.assert_array_equal(actual_current, expected_current)

            timestep = env.step(zero_action)

            # In the `after_step` method, we cache the goal for the current timestep
            # to compute the reward. Let's check that it matches the expected goal.
            np.testing.assert_array_equal(expected_current, env.task._goal_current)

    def test_fingering_observable(self) -> None:
        env = _get_env(control_timestep=0.01)
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape)
        timestep = env.reset()

        midi = _get_test_midi(dt=0.01)
        note_traj = midi_file.NoteTrajectory.from_midi(
            midi, dt=env.task.control_timestep
        )
        notes = note_traj.notes
        self.assertLen(notes, 4)

        for i in range(len(notes)):
            expected_fingering = np.zeros((2, 5))
            idxs = [note.fingering for note in notes[i]]
            rh_idxs = [idx for idx in idxs if idx < 5]
            lh_idxs = [idx - 5 for idx in idxs if idx >= 5]
            expected_fingering[0, rh_idxs] = 1.0
            expected_fingering[1, lh_idxs] = 1.0

            actual_fingering = timestep.observation["fingering"]
            np.testing.assert_array_equal(actual_fingering, expected_fingering.ravel())

            timestep = env.step(zero_action)

            # In the `after_step` method, we cache the fingering information for the
            # current timestep to compute the reward. Let's check that it matches the
            # expected one.
            actual_rh_current = [r[1] for r in env.task._rh_keys_current]
            np.testing.assert_array_equal(rh_idxs, actual_rh_current)
            actual_lh_current = [r[1] for r in env.task._lh_keys_current]
            np.testing.assert_array_equal(lh_idxs, actual_lh_current)

    def test_failure_termination(self) -> None:
        env = _get_env(wrong_press_termination=True)
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape)
        env.reset()

        # Simulate a wrong press by applying a generalized force on all the keys.
        env.physics.bind(env.task.piano.joints).qfrc_applied = 3.0

        # The episode should terminate in a single step.
        timestep = env.step(zero_action)
        self.assertTrue(timestep.last())
        self.assertTrue(env.task.should_terminate_episode(env.physics))
        # Failure, so discount should be 0.0.
        np.testing.assert_array_equal(env.task.get_discount(env.physics), 0.0)

    @absltest.skip("this observable is disabled")
    def test_steps_left_observable(self) -> None:
        env = _get_env(control_timestep=0.01)
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape)

        timestep = env.reset()
        self.assertEqual(timestep.observation["steps_left"], 1.0)

        for i in range(3):
            timestep = env.step(zero_action)
            self.assertAlmostEqual(
                timestep.observation["steps_left"], 1.0 - (i + 1) / 3
            )

    @parameterized.parameters(True, False)
    def test_fingering_reward_presence(self, disable_fingering_reward: bool) -> None:
        env = _get_env(disable_fingering_reward=disable_fingering_reward)
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape)
        env.reset()

        env.step(zero_action)
        reward_terms = env.task.reward_fn.reward_terms

        if disable_fingering_reward:
            self.assertNotIn("fingering_reward", reward_terms)
        else:
            self.assertIn("fingering_reward", reward_terms)
        self.assertIn("finger_collision_penalty", reward_terms)

    def test_finger_id_from_body_name(self) -> None:
        self.assertEqual(
            piano_with_shadow_hands._finger_id_from_body_name(
                "lh_daxian_hand/lh_index_MCP_link"
            ),
            "L:index",
        )
        self.assertEqual(
            piano_with_shadow_hands._finger_id_from_body_name(
                "rh_daxian_hand/rh_pinky_DIP_link"
            ),
            "R:pinky",
        )
        self.assertEqual(
            piano_with_shadow_hands._finger_id_from_body_name(
                "lh_daxian_hand/lh_mid_PIP_link"
            ),
            "L:middle",
        )
        self.assertIsNone(
            piano_with_shadow_hands._finger_id_from_body_name(
                "lh_daxian_hand/lh_palm_link"
            )
        )

    def test_daxian_v2_rest_has_no_finger_collision_penalty(self) -> None:
        task = piano_with_shadow_hands.PianoWithShadowHands(
            midi=_get_test_midi(dt=0.05),
            robot="daxian_v2",
            reduced_action_space=True,
            n_steps_lookahead=0,
            control_timestep=0.05,
        )
        env = composer.Environment(task, strip_singleton_obs_buffer_dim=True)
        env.reset()
        env.step(np.zeros(env.action_spec().shape))
        self.assertEqual(
            piano_with_shadow_hands._n_finger_collision_pairs(env.physics), 0
        )
        self.assertEqual(task.reward_fn.reward_terms["finger_collision_penalty"], 0.0)

    def test_daxian_v2_four_finger_mcp_range(self) -> None:
        task = piano_with_shadow_hands.PianoWithShadowHands(
            midi=_get_test_midi(dt=0.05),
            robot="daxian_v2",
            reduced_action_space=True,
            n_steps_lookahead=0,
            control_timestep=0.05,
        )
        env = composer.Environment(task, strip_singleton_obs_buffer_dim=True)
        env.reset()
        physics = env.physics
        for name in (
            "lh_daxian_hand/lh_index_MCP_joint",
            "rh_daxian_hand/rh_pinky_MCP_joint",
        ):
            jid = physics.model.name2id(name, "joint")
            lo, hi = physics.model.jnt_range[jid]
            self.assertAlmostEqual(float(lo), 0.0)
            self.assertAlmostEqual(float(hi), 0.6)
        thumb = physics.model.name2id("lh_daxian_hand/lh_thumb_MCP_joint", "joint")
        self.assertGreater(float(physics.model.jnt_range[thumb][1]), 1.0)

    def test_couple_four_finger_pip_dip_ik_vertical_sum(self) -> None:
        from robopianist.models.hands import daxian_v2_hand_constants as C

        at_rest = C.couple_four_finger_pip_dip({"index": 0.0})
        self.assertAlmostEqual(
            at_rest["index_PIP_joint"] + at_rest["index_DIP_joint"],
            C.FOUR_FINGER_IK_FLEX_TARGET,
            places=3,
        )
        pressed = C.couple_four_finger_pip_dip({"index": 0.6})
        self.assertAlmostEqual(
            pressed["index_PIP_joint"] + pressed["index_DIP_joint"] + 0.6,
            C.FOUR_FINGER_IK_FLEX_TARGET,
            places=3,
        )
        self.assertAlmostEqual(
            pressed["index_PIP_joint"]
            / (pressed["index_PIP_joint"] + pressed["index_DIP_joint"]),
            C.FOUR_FINGER_PIP_FRACTION,
            places=3,
        )
        flat = C.couple_four_finger_pip_dip({"index": 2.0})
        self.assertAlmostEqual(flat["index_PIP_joint"], 0.0)
        self.assertAlmostEqual(flat["index_DIP_joint"], 0.0)

    def test_daxian_v2_pinned_pip_dip_follow_mcp_ctrl(self) -> None:
        from robopianist.models.hands import daxian_v2_hand_constants as C

        task = piano_with_shadow_hands.PianoWithShadowHands(
            midi=_get_test_midi(dt=0.05),
            robot="daxian_v2",
            reduced_action_space=True,
            n_steps_lookahead=0,
            control_timestep=0.05,
        )
        env = composer.Environment(task, strip_singleton_obs_buffer_dim=True)
        env.reset()
        spec = env.action_spec()
        names = spec.name.split("\t") if isinstance(spec.name, str) else list(spec.name)
        action = np.zeros(spec.shape)
        mcp = 0.6
        hit = 0
        for i, name in enumerate(names):
            if name.endswith("index_MCP_joint"):
                action[i] = mcp
                hit += 1
        self.assertEqual(hit, 2)
        env.step(action)
        physics = env.physics
        # Test MIDI assigns RH index (part=1) at t=0; LH index is idle.
        pip = float(
            physics.named.data.ctrl["rh_daxian_hand/rh_A_index_PIP_joint"]
        )
        dip = float(
            physics.named.data.ctrl["rh_daxian_hand/rh_A_index_DIP_joint"]
        )
        self.assertAlmostEqual(pip + dip + mcp, math.pi / 2, delta=0.05)
        self.assertGreater(pip, 0.2)
        self.assertGreater(dip, 0.2)
        idle = C.idle_four_finger_pip_dip()
        self.assertAlmostEqual(
            float(physics.named.data.ctrl["lh_daxian_hand/lh_A_index_PIP_joint"]),
            idle["index_PIP_joint"],
            places=4,
        )
        self.assertAlmostEqual(
            float(physics.named.data.ctrl["lh_daxian_hand/lh_A_index_DIP_joint"]),
            0.0,
            places=4,
        )

    def test_daxian_v2_ik_makes_index_tip_vertical(self) -> None:
        task = piano_with_shadow_hands.PianoWithShadowHands(
            midi=_get_test_midi(dt=0.05),
            robot="daxian_v2",
            reduced_action_space=True,
            n_steps_lookahead=0,
            control_timestep=0.05,
        )
        env = composer.Environment(task, strip_singleton_obs_buffer_dim=True)
        env.reset()
        spec = env.action_spec()
        names = spec.name.split("\t") if isinstance(spec.name, str) else list(spec.name)
        action = np.zeros(spec.shape)
        mcp = 0.6
        for i, name in enumerate(names):
            if "rh_" in name and name.endswith("index_MCP_joint"):
                action[i] = mcp
        env.step(action)
        physics = env.physics
        pip = float(physics.named.data.ctrl["rh_daxian_hand/rh_A_index_PIP_joint"])
        dip = float(physics.named.data.ctrl["rh_daxian_hand/rh_A_index_DIP_joint"])
        physics.named.data.qpos["rh_daxian_hand/rh_index_MCP_joint"] = mcp
        physics.named.data.qpos["rh_daxian_hand/rh_index_PIP_joint"] = pip
        physics.named.data.qpos["rh_daxian_hand/rh_index_DIP_joint"] = dip
        physics.forward()
        rd = np.array(
            physics.named.data.xmat["rh_daxian_hand/rh_index_DIP_link"]
        ).reshape(3, 3)
        # Distal +Z should point down onto the keys.
        self.assertGreater(-rd[2, 2], 0.98)

    def test_daxian_v2_unlock_does_not_couple_pip_dip(self) -> None:
        from robopianist.models.hands import daxian_v2_hand_constants as C

        task = piano_with_shadow_hands.PianoWithShadowHands(
            midi=_get_test_midi(dt=0.05),
            robot="daxian_v2",
            reduced_action_space=True,
            unlock_four_finger_pip_dip=True,
            n_steps_lookahead=0,
            control_timestep=0.05,
        )
        env = composer.Environment(task, strip_singleton_obs_buffer_dim=True)
        env.reset()
        spec = env.action_spec()
        names = spec.name.split("\t") if isinstance(spec.name, str) else list(spec.name)
        action = np.zeros(spec.shape)
        for i, name in enumerate(names):
            if name.endswith("index_MCP_joint"):
                action[i] = 0.6
        env.step(action)
        expected = C.couple_four_finger_pip_dip({"index": 0.6})
        physics = env.physics
        pip = float(physics.named.data.ctrl["lh_daxian_hand/lh_A_index_PIP_joint"])
        # Unlocked CanonicalSpec 0 stays at 0, not the coupled rest-follow value.
        self.assertLess(pip, 0.05)
        self.assertGreater(expected["index_PIP_joint"], 0.2)

    def test_daxian_v2_pinky_rota_is_welded(self) -> None:
        task = piano_with_shadow_hands.PianoWithShadowHands(
            midi=_get_test_midi(dt=0.05),
            robot="daxian_v2",
            reduced_action_space=True,
            unlock_four_finger_pip_dip=True,
            n_steps_lookahead=0,
            control_timestep=0.05,
        )
        env = composer.Environment(task, strip_singleton_obs_buffer_dim=True)
        env.reset()
        names = list(env.physics.named.data.qpos.axes.row.names)
        self.assertFalse(any("pinky_rota" in n for n in names))
        spec = env.action_spec()
        act_names = spec.name.split("\t") if isinstance(spec.name, str) else list(spec.name)
        self.assertFalse(any("pinky_rota" in n for n in act_names))

    def test_daxian_v2_overlapping_fingers_are_penalized(self) -> None:
        task = piano_with_shadow_hands.PianoWithShadowHands(
            midi=_get_test_midi(dt=0.05),
            robot="daxian_v2",
            reduced_action_space=True,
            unlock_four_finger_pip_dip=True,
            n_steps_lookahead=0,
            control_timestep=0.05,
        )
        env = composer.Environment(task, strip_singleton_obs_buffer_dim=True)
        env.reset()
        physics = env.physics

        def _setq(name: str, value: float) -> None:
            jid = physics.model.name2id(name, "joint")
            physics.data.qpos[physics.model.jnt_qposadr[jid]] = value

        # Legal training MCP range [0, 0.6]: swing the middle finger into the index.
        _setq("lh_daxian_hand/lh_index_swing_joint", 0.0)
        _setq("lh_daxian_hand/lh_mid_swing_joint", 0.45)
        _setq("lh_daxian_hand/lh_index_MCP_joint", 0.6)
        _setq("lh_daxian_hand/lh_mid_MCP_joint", 0.6)
        _setq("lh_daxian_hand/lh_index_PIP_joint", 0.0)
        _setq("lh_daxian_hand/lh_mid_PIP_joint", 0.0)
        physics.forward()
        n_pairs = piano_with_shadow_hands._n_finger_collision_pairs(physics)
        self.assertGreater(n_pairs, 0)
        penalty = task._compute_finger_collision_penalty(physics)
        self.assertAlmostEqual(penalty, -0.15 * n_pairs)

    def test_daxian_v2_forearm_policy_axes(self) -> None:
        task = piano_with_shadow_hands.PianoWithShadowHands(
            midi=_get_test_midi(dt=0.05),
            robot="daxian_v2",
            reduced_action_space=True,
            n_steps_lookahead=0,
            control_timestep=0.05,
        )
        env = composer.Environment(task, strip_singleton_obs_buffer_dim=True)
        env.reset()
        physics = env.physics
        spec = env.action_spec()
        names = spec.name.split("\t") if isinstance(spec.name, str) else list(spec.name)
        self.assertTrue(any(n.endswith("forearm_yaw") for n in names))
        self.assertTrue(any(n.endswith("forearm_tz") for n in names))
        self.assertTrue(any(n.endswith("forearm_ty") for n in names))
        self.assertFalse(any(n.endswith("forearm_roll") for n in names))

        def _world_axis(joint_name: str) -> np.ndarray:
            jid = physics.model.name2id(joint_name, "joint")
            body = int(physics.model.jnt_bodyid[jid])
            rot = physics.data.xmat[body].reshape(3, 3)
            axis = np.asarray(physics.model.jnt_axis[jid], dtype=np.float64)
            world = rot @ axis
            return world / np.linalg.norm(world)

        yaw = _world_axis("lh_daxian_hand/forearm_yaw")
        tz = _world_axis("lh_daxian_hand/forearm_tz")
        ty = _world_axis("lh_daxian_hand/forearm_ty")
        self.assertLess(abs(float(np.dot(tz, yaw))), 0.25)
        self.assertLess(abs(float(np.dot(tz, ty))), 0.25)
        self.assertGreater(abs(float(yaw[2])), 0.9)
        self.assertGreater(abs(float(tz[0])), 0.9)
        self.assertGreater(abs(float(ty[2])), 0.9)
        lo_hi = physics.named.model.jnt_range
        yaw_lo, yaw_hi = v2c.LEFT_FOREARM_YAW_RANGE
        tz_lo, tz_hi = v2c.FOREARM_TZ_RANGE
        ty_lo, ty_hi = v2c.FOREARM_TY_RANGE
        for side in ("lh_daxian_hand", "rh_daxian_hand"):
            yaw_id = physics.model.name2id(f"{side}/forearm_yaw", "joint")
            tz_id = physics.model.name2id(f"{side}/forearm_tz", "joint")
            ty_id = physics.model.name2id(f"{side}/forearm_ty", "joint")
            self.assertAlmostEqual(float(lo_hi[yaw_id][0]), yaw_lo)
            self.assertAlmostEqual(float(lo_hi[yaw_id][1]), yaw_hi)
            self.assertAlmostEqual(float(lo_hi[tz_id][0]), tz_lo)
            self.assertAlmostEqual(float(lo_hi[tz_id][1]), tz_hi)
            self.assertAlmostEqual(float(lo_hi[ty_id][0]), ty_lo)
            self.assertAlmostEqual(float(lo_hi[ty_id][1]), ty_hi)

    def test_daxian_v2_canonical_zero_is_rest_forearm(self) -> None:
        from dm_env_wrappers import CanonicalSpecWrapper

        from robopianist.wrappers import PipRestAtZeroWrapper

        task = piano_with_shadow_hands.PianoWithShadowHands(
            midi=_get_test_midi(dt=0.05),
            robot="daxian_v2",
            reduced_action_space=True,
            n_steps_lookahead=0,
            control_timestep=0.05,
        )
        env = composer.Environment(task, strip_singleton_obs_buffer_dim=True)
        env = PipRestAtZeroWrapper(env)
        inner = env.action_spec()
        names = inner.name.split("\t") if isinstance(inner.name, str) else list(inner.name)
        yaw_idx = [i for i, n in enumerate(names) if n.endswith("forearm_yaw")]
        tz_idx = [i for i, n in enumerate(names) if n.endswith("forearm_tz")]
        ty_idx = [i for i, n in enumerate(names) if n.endswith("forearm_ty")]
        self.assertEqual(len(yaw_idx), 2)
        self.assertEqual(len(tz_idx), 2)
        self.assertEqual(len(ty_idx), 2)
        yaw_lo, yaw_hi = v2c.LEFT_FOREARM_YAW_RANGE
        tz_lo, tz_hi = v2c.FOREARM_TZ_RANGE
        ty_lo, ty_hi = v2c.FOREARM_TY_RANGE
        yaw_span = max(abs(yaw_lo), abs(yaw_hi))
        tz_span = max(abs(tz_lo), abs(tz_hi))
        ty_span = max(abs(ty_lo), abs(ty_hi))
        for i in yaw_idx:
            self.assertAlmostEqual(float(inner.minimum[i]), -yaw_span)
            self.assertAlmostEqual(float(inner.maximum[i]), yaw_span)
        for i in tz_idx:
            self.assertAlmostEqual(float(inner.minimum[i]), -tz_span)
            self.assertAlmostEqual(float(inner.maximum[i]), tz_span)
        for i in ty_idx:
            self.assertAlmostEqual(float(inner.minimum[i]), -ty_span)
            self.assertAlmostEqual(float(inner.maximum[i]), ty_span)
        env = CanonicalSpecWrapper(env, clip=True)
        env.reset()
        physics = env.physics

        def _ctrl(joint: str) -> tuple[float, float]:
            return (
                float(physics.named.data.ctrl[f"lh_daxian_hand/{joint}"]),
                float(physics.named.data.ctrl[f"rh_daxian_hand/{joint}"]),
            )

        action = np.zeros(env.action_spec().shape, dtype=np.float64)
        env.step(action)
        for q in _ctrl("forearm_yaw") + _ctrl("forearm_tz") + _ctrl("forearm_ty"):
            self.assertAlmostEqual(q, 0.0, places=5)
        plus = np.zeros_like(action)
        plus[yaw_idx + tz_idx + ty_idx] = 1.0
        env.step(plus)
        for q in _ctrl("forearm_yaw"):
            self.assertAlmostEqual(q, yaw_hi, places=5)
        for q in _ctrl("forearm_tz"):
            self.assertAlmostEqual(q, tz_hi, places=5)
        for q in _ctrl("forearm_ty"):
            self.assertAlmostEqual(q, ty_hi, places=5)
        minus = np.zeros_like(action)
        minus[yaw_idx + tz_idx + ty_idx] = -1.0
        env.step(minus)
        for q in _ctrl("forearm_yaw"):
            self.assertAlmostEqual(q, yaw_lo, places=5)
        for q in _ctrl("forearm_tz"):
            self.assertAlmostEqual(q, tz_lo, places=5)
        for q in _ctrl("forearm_ty"):
            self.assertAlmostEqual(q, ty_lo, places=5)

    # TODO(kevin): Add unit tests for individual reward components.
    # TODO(kevin): Add unit tests for augmentation / midi selection.


if __name__ == "__main__":
    absltest.main()
