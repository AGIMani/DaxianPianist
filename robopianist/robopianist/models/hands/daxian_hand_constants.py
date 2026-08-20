# Copyright 2023 The RoboPianist Authors.
# Adapted for Daxian hand.

from pathlib import Path
from typing import Dict, Tuple

_HERE = Path(__file__).resolve().parent
_DAXIAN_HAND_DIR = _HERE / "third_party" / "daxian"

NQ = 20  # Number of finger/wrist joints in the base MJCF (no forearm).
NU = 20  # Number of finger actuators in the base MJCF.

JOINT_GROUP: Dict[str, Tuple[str, ...]] = {
    "thumb": (
        "thumb_rota_block_joint",
        "thumb_rotaback_joint2",
        "thumb_MCP_joint",
        "thumb_PIP_joint",
    ),
    "first": (
        "index_MCP_joint",
        "index_swing_joint",
        "index_PIP_joint",
        "index_DIP_joint",
    ),
    "middle": (
        "mid_MCP_joint",
        "mid_swing_joint",
        "mid_PIP_joint",
        "mid_DIP_joint",
    ),
    "ring": (
        "ring_MCP_joint",
        "ring_swing_joint",
        "ring_PIP_joint",
        "ring_DIP_joint",
    ),
    "little": (
        "pinky_MCP_joint",
        "pinky_swing_joint",
        "pinky_PIP_joint",
        "pinky_DIP_joint",
    ),
}

# Order must match fingering reward / colorization (thumb → pinky).
FINGERTIP_BODIES: Tuple[str, ...] = (
    "thumb_PIP_link",
    "index_DIP_link",
    "mid_DIP_link",
    "ring_DIP_link",
    "pinky_DIP_link",
)

# Where the fingertip actually touches a key, in each tip body frame: the palmar
# surface of the primitive collision sphere, i.e. its centre plus the radius along the
# direction the tip travels when the finger flexes. Putting the reward site here (rather
# than on the finger's centre axis, ~12 mm above the pad) makes "site reaches the key"
# and "finger touches the key" the same event. Values are for the right hand; the left
# hand is a Y-mirror, so its y component is negated.
FINGERTIP_COLLISION_RADIUS = 0.008

# Peak torque (N-m) allowed on a finger position servo.
FINGER_FORCE_LIMIT = 2.0

# Effective mass (kg) and diagonal inertia (kg*m^2) of the virtual forearm mount, chosen
# to represent the human forearm the hand would be attached to.
FOREARM_MASS = 0.5
FOREARM_INERTIA = 1e-3

# World attachment pose. Single source of truth for the task AND the preview script.
# XYZ Euler degrees, MuJoCo 'XYZ' convention, then a spin about the forearm local +Z
# (the blue gizmo drawn on the hand base). Positive spin is CCW when looking along +Z.
HAND_BASE_RPY_DEG = (0.0, -90.0, 0.0)
# Left: CCW 180° about base blue; right: CW 180°. At 180° these are the same rotation.
LEFT_FOREARM_Z_SPIN_DEG = 180.0
RIGHT_FOREARM_Z_SPIN_DEG = -180.0
# From the player's view: rotate about world +Y so the finger edge of the palm
# sits above the wrist (fingers higher than the forearm attach).
PALM_PITCH_UP_DEG = 22
THUMB_DOWN_ROLL_DEG = 0.0
# World attach: x toward the player (keys occupy x∈[-0.075,0.075]).
# No forearm_ty. z from the left-hand SAPIEN tune; right is a Y-mirror.
LEFT_HAND_POSITION = (0.205, -0.3, 0.045)
RIGHT_HAND_POSITION = (0.205, 0.3, 0.045)
# Kept for the preview printout; the actual body quat is composed in suite/tasks/base.py.
LEFT_HAND_RPY_DEG = HAND_BASE_RPY_DEG
RIGHT_HAND_RPY_DEG = HAND_BASE_RPY_DEG

# Thumb drive. The CAD housing of rota_block / rotaback sits inside the palm at the
# URDF zero (~9 mm overlap). Those two links stay visual-only and palm contact is
# excluded; joint ranges match the MJCF (do not clip below rest).
#
# Reduced space: policy commands rota / rotaback; MCP / PIP stay at rest.
THUMB_HOUSING_BODIES = ("thumb_rota_block_link", "thumb_rotaback_link2")
THUMB_PALM_EXCLUDE = (
    ("palm_link", "thumb_rota_block_link"),
    ("palm_link", "thumb_rotaback_link2"),
)
# V3 palm mesh does not rest-penetrate the fingers; leave collision on.
PALM_VISUAL_ONLY_BODIES = ()
# MJCF ctrlrange. Must contain THUMB_REST_CTRL; never clip rest to fit a
# narrower box (the old rotaback clip (-0.40, 0.30) made rest unreachable).
THUMB_ROTA_BLOCK_RANGE = (0.0, 1.279)
THUMB_ROTABACK_RANGE = (0.0, 0.772)

THUMB_REST_CTRL = {
    "thumb_rota_block_joint": 1.1840,
    "thumb_rotaback_joint2": 0.4880,
    "thumb_MCP_joint": 0.00,
    "thumb_PIP_joint": 0.0,
}
# Four-finger PIP/DIP rest (pinned). MCP is policy-driven: CanonicalSpec 0 =
# straight (ctrlrange low), not these values and not the [0, 1.57] midpoint.
# Tuned on the left hand in SAPIEN; the right hand uses the same joint-space
# values (URDF/MJCF are already Y-mirrors).
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

# Extra Euler-X / rpy-roll on the attach pose. CanonicalSpec 0 is no extra
# roll (the 0 end of the range, not the midpoint). Right is the Y-mirror.
LEFT_FOREARM_ROLL_RANGE = (-0.5, 0.0)
RIGHT_FOREARM_ROLL_RANGE = (0.0, 0.5)
# Same signed range on both hands: yaw axis is reflected, +ctrl = outwards.
LEFT_FOREARM_YAW_RANGE = (-0.6, 0.0)
RIGHT_FOREARM_YAW_RANGE = (-0.6, 0.0)
FOREARM_YAW_RANGE = LEFT_FOREARM_YAW_RANGE


def rest_ctrl() -> Dict[str, float]:
    return {**THUMB_REST_CTRL, **FINGER_REST_CTRL}


def range_containing_rest(
    lo_hi: Tuple[float, float], rest: float
) -> Tuple[float, float]:
    """Widen ``(lo, hi)`` so ``rest`` is always a reachable command."""
    lo, hi = float(lo_hi[0]), float(lo_hi[1])
    rest = float(rest)
    return (min(lo, rest), max(hi, rest))


def thumb_policy_range(joint_name: str) -> Tuple[float, float]:
    """Actuator/joint range for an unlocked thumb drive, including rest."""
    named = {
        "thumb_rota_block_joint": THUMB_ROTA_BLOCK_RANGE,
        "thumb_rotaback_joint2": THUMB_ROTABACK_RANGE,
    }
    if joint_name not in named:
        raise KeyError(joint_name)
    return range_containing_rest(named[joint_name], THUMB_REST_CTRL[joint_name])


# When reduced_action_space=True the policy commands these joints (plus
# forearm_tx / forearm_roll). Four-finger PIP/DIP and thumb MCP/PIP stay at
# rest_ctrl() every step.
FOUR_FINGER_POLICY_JOINTS: Tuple[str, ...] = (
    "thumb_rota_block_joint",
    "thumb_rotaback_joint2",
    "index_swing_joint",
    "index_MCP_joint",
    "mid_swing_joint",
    "mid_MCP_joint",
    "ring_swing_joint",
    "ring_MCP_joint",
    "pinky_swing_joint",
    "pinky_MCP_joint",
)
# Always pinned in reduced space. Four-finger PIP/DIP are pinned unless
# unlock_four_finger_pip_dip=True (comparison experiment).
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

# MIDI fingering ids to ignore in goals / fingering / key-center rewards.
# Empty: thumb notes (RH=0, LH=5) are active now that rota/rotaback are free.
SKIP_THUMB_FINGERINGS: Tuple[int, ...] = ()

# Centre of the primitive collision sphere, in the tip body frame. These are the
# offsets MuJoCo derives from the tip meshes; they must be written out explicitly
# because swapping a mesh geom for a sphere discards the mesh-derived frame and would
# otherwise drop the sphere at the body origin, leaving the distal tip uncollidable.
FINGERTIP_COLLISION_POS: Dict[str, Tuple[float, float, float]] = {
    "thumb_PIP_link": (0.0049, 0.0024, 0.0120),
    "index_DIP_link": (0.0040, 0.0024, 0.0125),
    "mid_DIP_link": (0.0040, 0.0024, 0.0125),
    "ring_DIP_link": (0.0040, 0.0024, 0.0125),
    "pinky_DIP_link": (0.0040, 0.0024, 0.0125),
}

FINGERTIP_SITE_POS: Dict[str, Tuple[float, float, float]] = {
    "thumb_PIP_link": (0.0049, -0.0053, 0.0098),
    "index_DIP_link": (0.0119, 0.0024, 0.0111),
    "mid_DIP_link": (0.0119, 0.0024, 0.0111),
    "ring_DIP_link": (0.0119, 0.0024, 0.0111),
    "pinky_DIP_link": (0.0120, 0.0024, 0.0119),
}

FINGERTIP_COLORS: Tuple[Tuple[float, float, float], ...] = (
    (0.8, 0.2, 0.8),  # Purple: thumb.
    # Orange: index. Keep G low — topdown + key specular clips (0.95, 0.45, 0.05)
    # to the same pale yellow as pinky, so those keys look un-colored / yellow.
    # Wrong-press red is (0.85, 0.05, 0.05); this stays redder-orange.
    (1.0, 0.22, 0.0),
    (0.2, 0.8, 0.8),  # Cyan: middle.
    (0.2, 0.2, 0.8),  # Blue: ring.
    (0.8, 0.8, 0.2),  # Yellow: pinky.
)

RIGHT_DAXIAN_HAND_XML = _DAXIAN_HAND_DIR / "right_hand.xml"
LEFT_DAXIAN_HAND_XML = _DAXIAN_HAND_DIR / "left_hand.xml"
