# Copyright 2023 The RoboPianist Authors.
# Daxian V2 hand. Isolated from V3 (daxian_hand_constants.py).

from pathlib import Path
from typing import Dict, Tuple

_HERE = Path(__file__).resolve().parent
_DAXIAN_V2_HAND_DIR = _HERE / "third_party" / "daxian_v2"

NQ = 21  # Finger joints in the base MJCF (no forearm). V3 is 20.
NU = 21

JOINT_GROUP: Dict[str, Tuple[str, ...]] = {
    "thumb": (
        "thumb_rota_joint",
        "thumb_swing_joint",
        "thumb_MCP_joint",
        "thumb_PIP_joint",
    ),
    "first": (
        "index_swing_joint",
        "index_MCP_joint",
        "index_PIP_joint",
        "index_DIP_joint",
    ),
    "middle": (
        "mid_swing_joint",
        "mid_MCP_joint",
        "mid_PIP_joint",
        "mid_DIP_joint",
    ),
    "ring": (
        "ring_swing_joint",
        "ring_MCP_joint",
        "ring_PIP_joint",
        "ring_DIP_joint",
    ),
    "little": (
        "pinky_rota_joint",
        "pinky_swing_joint",
        "pinky_MCP_joint",
        "pinky_PIP_joint",
        "pinky_DIP_joint",
    ),
}

FINGERTIP_BODIES: Tuple[str, ...] = (
    "thumb_PIP_link",
    "index_DIP_link",
    "mid_DIP_link",
    "ring_DIP_link",
    "pinky_DIP_link",
)

FINGERTIP_COLLISION_RADIUS = 0.008
FINGER_FORCE_LIMIT = 2.0
FOREARM_MASS = 0.5
FOREARM_INERTIA = 1e-3

# Same attach convention as V3 so the piano task can share forearm DOFs.
HAND_BASE_RPY_DEG = (0.0, -90.0, 0.0)
LEFT_FOREARM_Z_SPIN_DEG = 180.0
RIGHT_FOREARM_Z_SPIN_DEG = -180.0
PALM_PITCH_UP_DEG = 22
THUMB_DOWN_ROLL_DEG = 0.0
LEFT_HAND_POSITION = (0.205, -0.3, 0.045)
RIGHT_HAND_POSITION = (0.205, 0.3, 0.045)
LEFT_HAND_RPY_DEG = HAND_BASE_RPY_DEG
RIGHT_HAND_RPY_DEG = HAND_BASE_RPY_DEG

# V2 has no rota_block / rotaback housing inside the palm.
THUMB_HOUSING_BODIES: Tuple[str, ...] = ()
THUMB_PALM_EXCLUDE: Tuple[Tuple[str, str], ...] = ()
THUMB_RANGE_JOINTS: Tuple[str, ...] = ()

THUMB_REST_CTRL = {
    "thumb_rota_joint": 0.0,
    "thumb_swing_joint": 0.0,
    "thumb_MCP_joint": 0.0,
    "thumb_PIP_joint": 0.0,
}
# Hardware q_zero is all zeros. Pinned PIP/DIP use a light curl so the pads
# hover above the keys (same idea as V3); retune if the V2 mesh sits too low.
FINGER_REST_CTRL = {
    "index_PIP_joint": 0.5490,
    "index_DIP_joint": 0.8740,
    "mid_PIP_joint": 0.5310,
    "mid_DIP_joint": 1.2850,
    "ring_PIP_joint": 0.5310,
    "ring_DIP_joint": 1.0470,
    "pinky_PIP_joint": 0.4720,
    "pinky_DIP_joint": 0.9200,
}

LEFT_FOREARM_ROLL_RANGE = (0.0, 0.3)
RIGHT_FOREARM_ROLL_RANGE = (-0.3, 0.0)


def rest_ctrl() -> Dict[str, float]:
    return {**THUMB_REST_CTRL, **FINGER_REST_CTRL}


def range_containing_rest(
    lo_hi: Tuple[float, float], rest: float
) -> Tuple[float, float]:
    lo, hi = float(lo_hi[0]), float(lo_hi[1])
    rest = float(rest)
    return (min(lo, rest), max(hi, rest))


def thumb_policy_range(joint_name: str) -> Tuple[float, float]:
    raise KeyError(joint_name)


FOUR_FINGER_POLICY_JOINTS: Tuple[str, ...] = (
    "thumb_rota_joint",
    "thumb_swing_joint",
    "index_swing_joint",
    "index_MCP_joint",
    "mid_swing_joint",
    "mid_MCP_joint",
    "ring_swing_joint",
    "ring_MCP_joint",
    "pinky_rota_joint",
    "pinky_swing_joint",
    "pinky_MCP_joint",
)
THUMB_PINNED_JOINTS: Tuple[str, ...] = (
    "thumb_MCP_joint",
    "thumb_PIP_joint",
)
FOUR_FINGER_PIP_DIP_JOINTS: Tuple[str, ...] = (
    "index_PIP_joint",
    "index_DIP_joint",
    "mid_PIP_joint",
    "mid_DIP_joint",
    "ring_PIP_joint",
    "ring_DIP_joint",
    "pinky_PIP_joint",
    "pinky_DIP_joint",
)
FOUR_FINGER_FIXED_JOINTS: Tuple[str, ...] = (
    *FOUR_FINGER_PIP_DIP_JOINTS,
    *THUMB_PINNED_JOINTS,
)


def pinned_joints(unlock_four_finger_pip_dip: bool = False) -> Tuple[str, ...]:
    if unlock_four_finger_pip_dip:
        return THUMB_PINNED_JOINTS
    return FOUR_FINGER_FIXED_JOINTS


SKIP_THUMB_FINGERINGS: Tuple[int, ...] = ()

# AABB centres of the V2 tip collision meshes (right hand). Left is Y-mirrored.
FINGERTIP_COLLISION_POS: Dict[str, Tuple[float, float, float]] = {
    "thumb_PIP_link": (0.0000, 0.0104, 0.0169),
    "index_DIP_link": (-0.0104, 0.0000, 0.0169),
    "mid_DIP_link": (-0.0104, 0.0000, 0.0169),
    "ring_DIP_link": (-0.0104, 0.0000, 0.0169),
    "pinky_DIP_link": (-0.0104, 0.0000, 0.0169),
}

FINGERTIP_SITE_POS: Dict[str, Tuple[float, float, float]] = dict(
    FINGERTIP_COLLISION_POS
)

FINGERTIP_COLORS: Tuple[Tuple[float, float, float], ...] = (
    (0.8, 0.2, 0.8),
    (1.0, 0.22, 0.0),
    (0.2, 0.8, 0.8),
    (0.2, 0.2, 0.8),
    (0.8, 0.8, 0.2),
)

RIGHT_DAXIAN_HAND_XML = _DAXIAN_V2_HAND_DIR / "right_hand.xml"
LEFT_DAXIAN_HAND_XML = _DAXIAN_V2_HAND_DIR / "left_hand.xml"
