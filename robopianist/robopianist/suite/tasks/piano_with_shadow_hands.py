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

"""A task where two shadow hands must play a given MIDI file on a piano."""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from dm_control import mjcf
from dm_control.composer import variation as base_variation
from dm_control.composer.observation import observable
from dm_control.mjcf import commit_defaults
from dm_control.utils.rewards import tolerance
from dm_env import specs
from mujoco_utils import collision_utils, spec_utils

from robopianist.models.arenas import stage
from robopianist.music import midi_file
from robopianist.suite import composite_reward
from robopianist.suite.tasks import base

# Distance thresholds for the shaping reward.
_FINGER_CLOSE_ENOUGH_TO_KEY = 0.01
_KEY_CLOSE_ENOUGH_TO_PRESSED = 0.05
# |y_tip - y_key| / key_half_width: on-centre if below 0.25, dead by 1.0.
_KEY_CENTER_IN_BOUNDS = 0.25
_KEY_CENTER_MARGIN = 1.0
# Per wrongly activated (non-MIDI) key. Also gates fingering_reward.
_FALSE_PRESS_PENALTY = 0.15
# Per distinct pair of fingers in contact (same or opposite hand).
_FINGER_COLLISION_PENALTY = 0.15

# Energy penalty coefficient.
_ENERGY_PENALTY_COEF = 5e-3

# Transparency of fingertip geoms.
_FINGERTIP_ALPHA = 1.0

# Bounds for the uniform distribution from which initial hand offset is sampled.
_POSITION_OFFSET = 0.05

_FINGER_NAME_TOKENS: Tuple[Tuple[str, str], ...] = (
    ("pinky", "pinky"),
    ("little", "pinky"),
    ("thumb", "thumb"),
    ("index", "index"),
    ("middle", "middle"),
    ("mid_", "middle"),
    ("ring", "ring"),
    ("_ff", "index"),
    ("_mf", "middle"),
    ("_rf", "ring"),
    ("_lf", "pinky"),
    ("_th", "thumb"),
)


def _hand_side_from_body_name(name: str) -> Optional[str]:
    n = name.lower()
    if "/lh_" in n or n.startswith("lh_"):
        return "L"
    if "/rh_" in n or n.startswith("rh_"):
        return "R"
    return None


def _finger_id_from_body_name(name: str) -> Optional[str]:
    """Map a geom body to ``L:index`` / ``R:thumb`` or None if not a finger."""
    n = name.lower()
    if any(skip in n for skip in ("palm", "forearm", "wrist", "world")):
        return None
    hand = _hand_side_from_body_name(n)
    if hand is None:
        return None
    for token, finger in _FINGER_NAME_TOKENS:
        if token in n:
            return f"{hand}:{finger}"
    return None


def _n_finger_collision_pairs(physics: mjcf.Physics) -> int:
    """Count distinct (finger A, finger B) pairs that currently have a contact."""
    model = physics.model
    data = physics.data
    labels = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = model.id2name(body_id, "body") or ""
        labels.append(_finger_id_from_body_name(body_name))
    pairs = set()
    for i in range(int(data.ncon)):
        contact = data.contact[i]
        a = labels[int(contact.geom1)]
        b = labels[int(contact.geom2)]
        if a is None or b is None or a == b:
            continue
        pairs.add(tuple(sorted((a, b))))
    return len(pairs)


class PianoWithShadowHands(base.PianoTask):
    def __init__(
        self,
        midi: midi_file.MidiFile,
        n_steps_lookahead: int = 1,
        n_seconds_lookahead: Optional[float] = None,
        trim_silence: bool = False,
        wrong_press_termination: bool = False,
        initial_buffer_time: float = 0.0,
        disable_fingering_reward: bool = False,
        disable_forearm_reward: bool = False,
        disable_colorization: bool = False,
        disable_hand_collisions: bool = False,
        augmentations: Optional[Sequence[base_variation.Variation]] = None,
        energy_penalty_coef: float = _ENERGY_PENALTY_COEF,
        randomize_hand_positions: bool = False,
        **kwargs,
    ) -> None:
        """Task constructor.

        Args:
            midi: A `MidiFile` object.
            n_steps_lookahead: Number of timesteps to look ahead when computing the
                goal state.
            n_seconds_lookahead: Number of seconds to look ahead when computing the
                goal state. If specified, this will override `n_steps_lookahead`.
            trim_silence: If True, shifts the MIDI file so that the first note starts
                at time 0.
            wrong_press_termination: If True, terminates the episode if the hands press
                the wrong keys at any timestep.
            initial_buffer_time: Specifies the duration of silence in seconds to add to
                the beginning of the MIDI file. A non-zero value can be useful for
                giving the agent time to place its hands near the first notes.
            disable_fingering_reward: If True, disables the shaping reward for
                fingering. This will also disable the colorization of the fingertips
                and corresponding keys. Note that if the MIDI file does not contain
                any fingering information, the fingering reward will also be disabled.
            disable_forearm_reward: If True, disables the shaping reward for the
                forearms.
            disable_colorization: If True, disables the colorization of the fingertips
                and corresponding keys.
            disable_hand_collisions: If True, disables collisions between the two hands.
            augmentations: A list of `Variation` objects that will be applied to the
                MIDI file at the beginning of each episode. If None, no augmentations
                will be applied.
            energy_penalty_coef: Coefficient for the energy penalty.
            randomize_hand_positions: If True, randomizes the initial position of the
                hands at the beginning of each episode.
        """
        super().__init__(arena=stage.Stage(), **kwargs)

        if trim_silence:
            midi = midi.trim_silence()
        self._midi = midi
        self._initial_midi = midi
        self._n_steps_lookahead = n_steps_lookahead
        if n_seconds_lookahead is not None:
            self._n_steps_lookahead = int(
                np.ceil(n_seconds_lookahead / self.control_timestep)
            )
        self._initial_buffer_time = initial_buffer_time
        self._disable_fingering_reward = (
            disable_fingering_reward or not self._midi.has_fingering()
        )
        self._disable_forearm_reward = disable_forearm_reward
        self._wrong_press_termination = wrong_press_termination
        self._disable_colorization = disable_colorization
        self._disable_hand_collisions = disable_hand_collisions
        self._augmentations = augmentations
        self._energy_penalty_coef = energy_penalty_coef
        self._randomize_hand_positions = randomize_hand_positions

        if not disable_fingering_reward and not disable_colorization:
            self._colorize_fingertips()
        if disable_hand_collisions:
            self._disable_collisions_between_hands()
        self._reset_quantities_at_episode_init()
        self._reset_trajectory()  # Important: call before adding observables.
        self._add_observables()
        self._set_rewards()

    def _set_rewards(self) -> None:
        self._reward_fn = composite_reward.CompositeReward(
            key_press_reward=self._compute_key_press_reward,
            sustain_reward=self._compute_sustain_reward,
            energy_reward=self._compute_energy_reward,
        )
        if not self._disable_fingering_reward:
            self._reward_fn.add("fingering_reward", self._compute_fingering_reward)
        else:
            # use OT based fingering
            print('Fingering is unavailable. OT fingering reward is used.')
            self._reward_fn.add("ot_fingering_reward", self._compute_ot_fingering_reward)

        if not self._disable_forearm_reward:
            self._reward_fn.add("forearm_reward", self._compute_forearm_reward)
        self._reward_fn.add("key_center_reward", self._compute_key_center_reward)
        self._reward_fn.add("false_press_penalty", self._compute_false_press_penalty)
        self._reward_fn.add(
            "finger_collision_penalty", self._compute_finger_collision_penalty
        )

    def _reset_quantities_at_episode_init(self) -> None:
        self._t_idx: int = 0
        self._should_terminate: bool = False
        self._discount: float = 1.0
        self._rh_keys: List[Tuple[int, int]] = []
        self._lh_keys: List[Tuple[int, int]] = []
        self._rh_keys_current: List[Tuple[int, int]] = []
        self._lh_keys_current: List[Tuple[int, int]] = []

    def _maybe_change_midi(self, random_state: np.random.RandomState) -> None:
        if self._augmentations is not None:
            midi = self._initial_midi
            for var in self._augmentations:
                midi = var(initial_value=midi, random_state=random_state)
            self._midi = midi
            self._reset_trajectory()

    def _reset_trajectory(self) -> None:
        note_traj = midi_file.NoteTrajectory.from_midi(
            self._midi, self.control_timestep
        )
        note_traj.add_initial_buffer_time(self._initial_buffer_time)
        self._notes = note_traj.notes
        self._sustains = note_traj.sustains

    # Composer methods.

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        self._maybe_change_midi(random_state)
        self._reset_quantities_at_episode_init()
        self._update_fingering_state()
        for hand in (self.right_hand, self.left_hand):
            pin = getattr(hand, "_pin_fixed_fingers", None)
            if pin is not None:
                pin(physics)
        self._randomize_initial_hand_positions(physics, random_state)

    def before_step(
        self,
        physics: mjcf.Physics,
        action: np.ndarray,
        random_state: np.random.RandomState,
    ) -> None:
        """Applies the control to the hands and the sustain pedal to the piano."""
        self._update_fingering_state()
        action_right, action_left = np.split(action[:-1], 2)
        self.right_hand.apply_action(physics, action_right, random_state)
        self.left_hand.apply_action(physics, action_left, random_state)
        self.piano.apply_sustain(physics, action[-1], random_state)

    def after_step(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        del random_state  # Unused.
        self._t_idx += 1
        self._should_terminate = (self._t_idx - 1) == len(self._notes) - 1

        self._goal_current = self._goal_state[0]

        if not self._disable_fingering_reward:
            self._rh_keys_current = self._rh_keys
            self._lh_keys_current = self._lh_keys
            if not self._disable_colorization:
                self._colorize_keys(physics)

        should_not_be_pressed = np.flatnonzero(1 - self._goal_current[:-1])
        self._failure_termination = self.piano.activation[should_not_be_pressed].any()

    def get_reward(self, physics: mjcf.Physics) -> float:
        return self._reward_fn.compute(physics)

    def get_discount(self, physics: mjcf.Physics) -> float:
        del physics  # Unused.
        return self._discount

    def should_terminate_episode(self, physics: mjcf.Physics) -> bool:
        del physics  # Unused.
        if self._should_terminate:
            return True
        if self._wrong_press_termination and self._failure_termination:
            self._discount = 0.0
            return True
        return False

    @property
    def task_observables(self):
        return self._task_observables

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        right_spec = self.right_hand.action_spec(physics)
        left_spec = self.left_hand.action_spec(physics)
        hands_spec = spec_utils.merge_specs([right_spec, left_spec])
        sustain_spec = specs.BoundedArray(
            shape=(1,),
            dtype=hands_spec.dtype,
            minimum=[0.0],
            maximum=[1.0],
            name="sustain",
        )
        return spec_utils.merge_specs([hands_spec, sustain_spec])

    # Other.

    @property
    def midi(self) -> midi_file.MidiFile:
        return self._midi

    @property
    def reward_fn(self) -> composite_reward.CompositeReward:
        return self._reward_fn

    # Helper methods.

    def _compute_forearm_reward(self, physics: mjcf.Physics) -> float:
        """Reward for not colliding the forearms."""
        if collision_utils.has_collision(
            physics,
            [g.full_identifier for g in self.right_hand.root_body.geom],
            [g.full_identifier for g in self.left_hand.root_body.geom],
        ):
            return 0.0
        return 0.5

    def _compute_sustain_reward(self, physics: mjcf.Physics) -> float:
        """Reward for pressing the sustain pedal at the right time."""
        del physics  # Unused.
        return tolerance(
            self._goal_current[-1] - self.piano.sustain_activation[0],
            bounds=(0, _KEY_CLOSE_ENOUGH_TO_PRESSED),
            margin=(_KEY_CLOSE_ENOUGH_TO_PRESSED * 10),
            sigmoid="gaussian",
        )

    def _compute_energy_reward(self, physics: mjcf.Physics) -> float:
        """Reward for minimizing energy."""
        rew = 0.0
        for hand in [self.right_hand, self.left_hand]:
            power = hand.observables.actuators_power(physics).copy()
            rew -= self._energy_penalty_coef * np.sum(power)
        return rew

    def _compute_key_press_reward(self, physics: mjcf.Physics) -> float:
        """Reward for pressing the right keys at the right time."""
        del physics  # Unused.
        on = np.flatnonzero(self._goal_current[:-1])
        rew = 0.0
        # It's possible we have no keys to press at this timestep, so we need to check
        # that `on` is not empty.
        if on.size > 0:
            actual = np.array(self.piano.state / self.piano._qpos_range[:, 1])
            rews = tolerance(
                self._goal_current[:-1][on] - actual[on],
                bounds=(0, _KEY_CLOSE_ENOUGH_TO_PRESSED),
                margin=(_KEY_CLOSE_ENOUGH_TO_PRESSED * 10),
                sigmoid="gaussian",
            )
            rew += 0.5 * rews.mean()
        # If there are any false positives, the remaining 0.5 reward is lost.
        off = np.flatnonzero(1 - self._goal_current[:-1])
        rew += 0.5 * (1 - float(self.piano.activation[off].any()))
        return rew

    def _compute_fingering_reward(self, physics: mjcf.Physics) -> float:
        """Reward for minimizing the distance between the fingers and the keys."""

        def _distance_finger_to_key(
            hand_keys: List[Tuple[int, int]], hand
        ) -> List[float]:
            distances = []
            for key, mjcf_fingering in hand_keys:
                fingertip_site = hand.fingertip_sites[mjcf_fingering]
                fingertip_pos = physics.bind(fingertip_site).xpos.copy()
                key_geom = self.piano.keys[key].geom[0]
                key_geom_pos = physics.bind(key_geom).xpos.copy()
                key_geom_pos[-1] += 0.5 * physics.bind(key_geom).size[2]
                key_geom_pos[0] += 0.35 * physics.bind(key_geom).size[0]
                diff = key_geom_pos - fingertip_pos
                distances.append(float(np.linalg.norm(diff)))
            return distances

        distances = _distance_finger_to_key(self._rh_keys_current, self.right_hand)
        distances += _distance_finger_to_key(self._lh_keys_current, self.left_hand)

        # Case where there are no keys to press at this timestep.
        if not distances:
            return 0.0

        rews = tolerance(
            np.hstack(distances),
            bounds=(0, _FINGER_CLOSE_ENOUGH_TO_KEY),
            margin=(_FINGER_CLOSE_ENOUGH_TO_KEY * 10),
            sigmoid="gaussian",
        )
        raw = float(np.mean(rews))
        # Sticking on non-target keys must not keep paying fingering.
        return raw / (1.0 + self._n_false_presses())

    def _compute_ot_fingering_reward(self, physics: mjcf.Physics) -> float:
        """ OT reward calculation from RP1M https://arxiv.org/abs/2408.11048 """
        # calcuate fingertip positions
        fingertip_pos = [physics.bind(finger).xpos.copy() for finger in self.left_hand.fingertip_sites]
        fingertip_pos += [physics.bind(finger).xpos.copy() for finger in self.right_hand.fingertip_sites]
        
        # calcuate the positions of piano keys to press.
        keys_to_press = np.flatnonzero(self._goal_current[:-1]) # keys to press
        # if no key is pressed
        if keys_to_press.shape[0] == 0:
            return 1.

        # calculate key pos
        key_pos = []
        for key in keys_to_press:
            key_geom = self.piano.keys[key].geom[0]
            key_geom_pos = physics.bind(key_geom).xpos.copy()
            key_geom_pos[-1] += 0.5 * physics.bind(key_geom).size[2]
            key_geom_pos[0] += 0.35 * physics.bind(key_geom).size[0]
            key_pos.append(key_geom_pos.copy())

        # calcualte the distance between keys and fingers
        dist = np.full((len(fingertip_pos), len(key_pos)), 100.)
        for i, finger in enumerate(fingertip_pos):
            for j, key in enumerate(key_pos):
                dist[i, j] = np.linalg.norm(key - finger)
        
        # calculate the shortest distance
        row_ind, col_ind = linear_sum_assignment(dist)
        dist = dist[row_ind, col_ind]
        rews = tolerance(
            dist,
            bounds=(0, _FINGER_CLOSE_ENOUGH_TO_KEY),
            margin=(_FINGER_CLOSE_ENOUGH_TO_KEY * 10),
            sigmoid="gaussian",
        )
        return float(np.mean(rews)) / (1.0 + self._n_false_presses())

    def _n_false_presses(self) -> int:
        """Keys that are down but not in the current MIDI goal."""
        off = np.flatnonzero(1 - self._goal_current[:-1])
        return int(np.count_nonzero(self.piano.activation[off]))

    def _compute_false_press_penalty(self, physics: mjcf.Physics) -> float:
        """Penalty when a key is activated with no MIDI target this step."""
        del physics
        return -_FALSE_PRESS_PENALTY * float(self._n_false_presses())

    def _compute_finger_collision_penalty(self, physics: mjcf.Physics) -> float:
        """Penalty when two different fingers are in contact."""
        return -_FINGER_COLLISION_PENALTY * float(_n_finger_collision_pairs(physics))

    def _compute_key_center_reward(self, physics: mjcf.Physics) -> float:
        """Higher when the fingertip contact is on the key's centre line (world Y)."""
        offsets: List[float] = []
        for hand, keys in zip(
            [self.right_hand, self.left_hand],
            [self._rh_keys_current, self._lh_keys_current],
        ):
            for key, mjcf_fingering in keys:
                fingertip_site = hand.fingertip_sites[mjcf_fingering]
                tip = physics.bind(fingertip_site).xpos
                geom = physics.bind(self.piano.keys[key].geom[0])
                half_width = float(np.abs(geom.size[1]))
                if half_width < 1e-6:
                    continue
                offsets.append(float(np.abs(tip[1] - geom.xpos[1]) / half_width))
        if not offsets:
            return 0.0
        return float(
            np.mean(
                tolerance(
                    np.asarray(offsets),
                    bounds=(0.0, _KEY_CENTER_IN_BOUNDS),
                    margin=_KEY_CENTER_MARGIN,
                    sigmoid="gaussian",
                )
            )
        ) 

    def _update_goal_state(self) -> None:
        # Observable callables get called after `after_step` but before
        # `should_terminate_episode`. Since we increment `self._t_idx` in `after_step`,
        # we need to guard against out of bounds indexing. Note that the goal state
        # does not matter at this point since we are terminating the episode and this
        # update is usually meant for the next timestep.
        if self._t_idx == len(self._notes):
            return

        self._goal_state = np.zeros(
            (self._n_steps_lookahead + 1, self.piano.n_keys + 1),
            dtype=np.float64,
        )
        t_start = self._t_idx
        t_end = min(t_start + self._n_steps_lookahead + 1, len(self._notes))
        for i, t in enumerate(range(t_start, t_end)):
            keys = [note.key for note in self._notes[t]]
            self._goal_state[i, keys] = 1.0
            self._goal_state[i, -1] = self._sustains[t]

    def _four_fingers_from_keys(self, keys: List[Tuple[int, int]]) -> Tuple[str, ...]:
        lookup = getattr(self.hand_consts, "four_finger_from_mjcf_fingering", None)
        names = []
        for _, mjcf_i in keys:
            if lookup is not None:
                name = lookup(mjcf_i)
            else:
                name = {1: "index", 2: "mid", 3: "ring", 4: "pinky"}.get(int(mjcf_i))
            if name:
                names.append(name)
        return tuple(names)

    def _sync_pip_dip_midi_gate(self) -> None:
        rh = self._four_fingers_from_keys(getattr(self, "_rh_keys", []))
        lh = self._four_fingers_from_keys(getattr(self, "_lh_keys", []))
        setter = getattr(self.right_hand, "set_pip_dip_active_fingers", None)
        if setter is not None:
            setter(rh)
        setter = getattr(self.left_hand, "set_pip_dip_active_fingers", None)
        if setter is not None:
            setter(lh)

    def _update_fingering_state(self) -> None:
        if self._t_idx >= len(self._notes):
            self._rh_keys = []
            self._lh_keys = []
            self._fingering_state = np.zeros((2, 5), dtype=np.float64)
            self._sync_pip_dip_midi_gate()
            return

        fingering = [note.fingering for note in self._notes[self._t_idx]]
        fingering_keys = [note.key for note in self._notes[self._t_idx]]

        # Split fingering into right and left hand.
        self._rh_keys: List[Tuple[int, int]] = []
        self._lh_keys: List[Tuple[int, int]] = []
        for key, finger in enumerate(fingering):
            piano_key = fingering_keys[key]
            if finger < 5:
                self._rh_keys.append((piano_key, finger))
            else:
                self._lh_keys.append((piano_key, finger - 5))

        # For each hand, set the finger to 1 if it is used and 0 otherwise.
        self._fingering_state = np.zeros((2, 5), dtype=np.float64)
        for hand, keys in enumerate([self._rh_keys, self._lh_keys]):
            for key, mjcf_fingering in keys:
                self._fingering_state[hand, mjcf_fingering] = 1.0
        self._sync_pip_dip_midi_gate()

    def _add_observables(self) -> None:
        # Enable hand observables.
        enabled_observables = [
            "joints_pos",
            # NOTE(kevin): This observable was previously enabled but it is redundant
            # since it is encoded in the joint positions, specifically via the forearm
            # slider joints (which are in units of meters).
            # "position",
        ]
        for hand in [self.right_hand, self.left_hand]:
            for obs in enabled_observables:
                getattr(hand.observables, obs).enabled = True

        # This returns the current state of the piano keys.
        self.piano.observables.state.enabled = True
        self.piano.observables.sustain_state.enabled = True

        # This returns the goal state for the current timestep and n steps ahead.
        def _get_goal_state(physics) -> np.ndarray:
            del physics  # Unused.
            self._update_goal_state()
            return self._goal_state.ravel()

        goal_observable = observable.Generic(_get_goal_state)
        goal_observable.enabled = True
        self._task_observables = {"goal": goal_observable}

        # This adds fingering information for the current timestep.
        def _get_fingering_state(physics) -> np.ndarray:
            del physics  # Unused.
            self._update_fingering_state()
            return self._fingering_state.ravel()

        fingering_observable = observable.Generic(_get_fingering_state)
        fingering_observable.enabled = not self._disable_fingering_reward
        self._task_observables["fingering"] = fingering_observable

    def _colorize_fingertips(self) -> None:
        """Colorize the fingertips of the hands."""
        colors = self.hand_consts.FINGERTIP_COLORS
        for hand in [self.right_hand, self.left_hand]:
            for i, body in enumerate(hand.fingertip_bodies):
                color = colors[i] + (_FINGERTIP_ALPHA,)
                for geom in body.find_all("geom"):
                    dclass = geom.dclass
                    # Shadow visual meshes use class="plastic_visual". Daxian has
                    # no dclass: visual meshes are contype=0 / group=1. Collision
                    # spheres on the same body get the matching fingering color.
                    is_shadow_visual = (
                        dclass is not None and dclass.dclass == "plastic_visual"
                    )
                    is_daxian_visual = dclass is None and (
                        geom.contype == 0 or geom.group == 1
                    )
                    is_tip_sphere = geom.type == "sphere"
                    if is_shadow_visual or is_daxian_visual or is_tip_sphere:
                        geom.rgba = color
                # Also color the fingertip sites.
                hand.fingertip_sites[i].rgba = color

    def _colorize_keys(self, physics) -> None:
        """Colorize the keys by the corresponding fingertip color."""
        for hand, keys in zip(
            [self.right_hand, self.left_hand],
            [self._rh_keys_current, self._lh_keys_current],
        ):
            for key, mjcf_fingering in keys:
                key_geom = self.piano.keys[key].geom[0]
                fingertip_site = hand.fingertip_sites[mjcf_fingering]
                if not self.piano.activation[key]:
                    physics.bind(key_geom).rgba = tuple(fingertip_site.rgba[:3]) + (
                        1.0,
                    )

    def _disable_collisions_between_hands(self) -> None:
        """Disable collisions between the hands."""
        for hand in [self.right_hand, self.left_hand]:
            for geom in hand.mjcf_model.find_all("geom"):
                # If both hands have the same contype and conaffinity, then they can't
                # collide. They can still collide with the piano since the piano has
                # contype 0 and conaffinity 1. Lastly, we make sure we're not changing
                # the contype and conaffinity of the hand geoms that are already
                # disabled (i.e., the visual geoms).
                commit_defaults(geom, ["contype", "conaffinity"])
                if geom.contype == 0 and geom.conaffinity == 0:
                    continue
                geom.conaffinity = 0
                geom.contype = 1

    def _randomize_initial_hand_positions(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        """Randomize the initial position of the hands."""
        if not self._randomize_hand_positions:
            return
        offset = random_state.uniform(low=-_POSITION_OFFSET, high=_POSITION_OFFSET)
        for hand in [self.right_hand, self.left_hand]:
            hand.shift_pose(physics, (0, offset, 0))
