# Copyright 2023 The RoboPianist Authors.

"""Shift selected action bounds so CanonicalSpec maps 0 to rest, not mid-range."""

from typing import Sequence

import dm_env
import numpy as np
from dm_env import specs
from dm_env_wrappers import EnvironmentWrapper

from robopianist.models.hands import daxian_hand_constants as consts


def _action_names(spec: specs.BoundedArray) -> list[str]:
    raw = spec.name
    if not raw:
        return [f"a{i}" for i in range(int(spec.shape[0]))]
    names = raw.split("\t") if isinstance(raw, str) else list(raw)
    if len(names) != spec.shape[0]:
        return [f"a{i}" for i in range(int(spec.shape[0]))]
    return names


def _rest_for_name(name: str, orig_lo: float) -> float | None:
    rest = consts.rest_ctrl()
    for joint_name, val in rest.items():
        if not name.endswith(joint_name):
            continue
        # Unlocked four-finger PIP/DIP: CanonicalSpec 0 = straight (range low),
        # same as MCP. Thumb PIP stays pinned at rest_ctrl.
        if (
            (name.endswith("PIP_joint") or name.endswith("DIP_joint"))
            and "thumb_" not in name
        ):
            return float(orig_lo)
        return float(val)
    # Four-finger MCP is policy-driven and not in rest_ctrl(). 0 = URDF straight
    # (ctrlrange low), not the [0, 1.57] midpoint. Skip thumb_MCP (pinned).
    if name.endswith("MCP_joint") and "thumb_" not in name:
        return float(orig_lo)
    if name.endswith("forearm_roll"):
        return 0.0
    return None


class PipRestAtZeroWrapper(EnvironmentWrapper):
    """Advertise bounds symmetric about rest so CanonicalSpec 0 is rest.

    Four-finger MCP ctrlrange is ``[0, 1.57]``. Without this, policy 0 became
    ~0.785 rad (already a press). Unlocked thumb ``rota`` / ``rotaback`` use
    ``rest_ctrl()`` the same way. ``forearm_roll`` rest is 0 (no extra Euler X);
    left is ``[0, 0.3]``, right the Y-mirror ``[-0.3, 0]``.

    * policy 0 → rest (MCP: straight 0; roll: 0; thumb: ``THUMB_REST_CTRL``)
    * policy +1 → original maximum (clipped if rest sits at the high end)
    * policy -1 → advertised minimum, clipped to the physical range
    """

    def __init__(
        self,
        environment: dm_env.Environment,
        joint_suffixes: Sequence[str] = ("MCP_joint", "forearm_roll"),
    ) -> None:
        super().__init__(environment)
        inner = environment.action_spec()
        names = _action_names(inner)
        idx = []
        orig_lo = []
        orig_hi = []
        advertised_lo = np.array(inner.minimum, dtype=inner.dtype, copy=True)
        advertised_hi = np.array(inner.maximum, dtype=inner.dtype, copy=True)
        for i, name in enumerate(names):
            lo = float(inner.minimum[i])
            hi = float(inner.maximum[i])
            rest_val = _rest_for_name(name, lo)
            if rest_val is None and not any(name.endswith(s) for s in joint_suffixes):
                continue
            if rest_val is None:
                rest_val = lo
            if rest_val < lo:
                lo = rest_val
            if rest_val > hi:
                hi = rest_val
            idx.append(i)
            orig_lo.append(lo)
            orig_hi.append(hi)
            # Shift the far side of rest so CanonicalSpec 0 is rest. If rest is
            # already the high bound (right forearm_roll), expand max instead of
            # collapsing the advertised range to a point.
            adv_lo = 2.0 * rest_val - hi
            adv_hi = hi
            if adv_lo >= adv_hi - 1e-9:
                adv_lo = lo
                adv_hi = 2.0 * rest_val - lo
            advertised_lo[i] = adv_lo
            advertised_hi[i] = adv_hi
        if not idx:
            raise ValueError(
                "PipRestAtZeroWrapper found no rest/MCP/roll joints in "
                f"{tuple(joint_suffixes)} / rest_ctrl among {names}"
            )
        self._idx = np.asarray(idx, dtype=np.int64)
        self._lo = np.asarray(orig_lo, dtype=np.float64)
        self._hi = np.asarray(orig_hi, dtype=np.float64)
        self._spec = inner.replace(minimum=advertised_lo, maximum=advertised_hi)

    def action_spec(self) -> specs.BoundedArray:
        return self._spec

    def step(self, action) -> dm_env.TimeStep:
        action = np.array(action, copy=True)
        action[self._idx] = np.clip(action[self._idx], self._lo, self._hi)
        return self._environment.step(action)
