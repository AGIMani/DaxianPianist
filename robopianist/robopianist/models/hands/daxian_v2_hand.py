# Copyright 2023 The RoboPianist Authors.
# Daxian V2 composer class. Shares DaxianHand; V3 files stay untouched.

from robopianist.models.hands.daxian_hand import DaxianHand
from robopianist.models.hands import daxian_v2_hand_constants as consts


class DaxianV2Hand(DaxianHand):
    """Daxian V2 anthropomorphic hand (21 finger joints, pinky rota)."""

    def _build(self, **kwargs) -> None:
        kwargs.setdefault("consts_module", consts)
        super()._build(**kwargs)
