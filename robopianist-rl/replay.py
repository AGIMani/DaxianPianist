from typing import NamedTuple, Optional

import numpy as np
import dm_env

try:
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jnp = None


class Transition(NamedTuple):
    state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    discount: np.ndarray
    next_state: np.ndarray


class Buffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_size: int,
        batch_size: int,
        on_gpu: bool = False,
    ) -> None:
        self._max_size = max_size
        self._batch_size = batch_size
        self._on_gpu = bool(on_gpu) and jnp is not None
        zeros = jnp.zeros if self._on_gpu else np.zeros

        self._states = zeros((max_size, state_dim), dtype=np.float32)
        self._actions = zeros((max_size, action_dim), dtype=np.float32)
        self._next_states = zeros((max_size, state_dim), dtype=np.float32)
        self._rewards = zeros((max_size,), dtype=np.float32)
        self._discounts = zeros((max_size,), dtype=np.float32)

        self._ptr: int = 0
        self._size: int = 0
        self._prev: Optional[dm_env.TimeStep] = None
        self._action: Optional[np.ndarray] = None
        self._latest: Optional[dm_env.TimeStep] = None

    def insert(
        self,
        timestep: dm_env.TimeStep,
        action: Optional[np.ndarray],
    ) -> None:
        self._prev = self._latest
        self._action = action
        self._latest = timestep

        if action is not None:
            self.insert_batch(
                np.asarray(self._prev.observation, dtype=np.float32)[None],
                np.asarray(action, dtype=np.float32)[None],
                np.asarray([self._latest.reward], dtype=np.float32),
                np.asarray([self._latest.discount], dtype=np.float32),
                np.asarray(self._latest.observation, dtype=np.float32)[None],
            )

    def insert_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        discounts: np.ndarray,
        next_states: np.ndarray,
    ) -> None:
        n = int(states.shape[0])
        if n == 0:
            return
        end = self._ptr + n
        if end <= self._max_size:
            self._write_slice(slice(self._ptr, end), states, actions, rewards, discounts, next_states)
        else:
            first = self._max_size - self._ptr
            self._write_slice(
                slice(self._ptr, self._max_size),
                states[:first],
                actions[:first],
                rewards[:first],
                discounts[:first],
                next_states[:first],
            )
            self._write_slice(
                slice(0, n - first),
                states[first:],
                actions[first:],
                rewards[first:],
                discounts[first:],
                next_states[first:],
            )
        self._ptr = end % self._max_size
        self._size = min(self._size + n, self._max_size)

    def _write_slice(self, slc: slice, states, actions, rewards, discounts, next_states) -> None:
        if self._on_gpu:
            self._states = self._states.at[slc].set(jnp.asarray(states, dtype=jnp.float32))
            self._actions = self._actions.at[slc].set(jnp.asarray(actions, dtype=jnp.float32))
            self._rewards = self._rewards.at[slc].set(jnp.asarray(rewards, dtype=jnp.float32))
            self._discounts = self._discounts.at[slc].set(jnp.asarray(discounts, dtype=jnp.float32))
            self._next_states = self._next_states.at[slc].set(
                jnp.asarray(next_states, dtype=jnp.float32)
            )
        else:
            self._states[slc] = states
            self._actions[slc] = actions
            self._rewards[slc] = rewards
            self._discounts[slc] = discounts
            self._next_states[slc] = next_states

    def sample(self) -> Transition:
        ind = np.random.randint(0, self._size, size=self._batch_size)
        if self._on_gpu:
            ind_j = jnp.asarray(ind)
            return Transition(
                state=self._states[ind_j],
                action=self._actions[ind_j],
                reward=self._rewards[ind_j],
                discount=self._discounts[ind_j],
                next_state=self._next_states[ind_j],
            )
        return Transition(
            state=self._states[ind],
            action=self._actions[ind],
            reward=self._rewards[ind],
            discount=self._discounts[ind],
            next_state=self._next_states[ind],
        )

    def is_ready(self) -> bool:
        return self._batch_size <= len(self)

    def __len__(self) -> int:
        return self._size
