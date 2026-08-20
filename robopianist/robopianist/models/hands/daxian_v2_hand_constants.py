# Copyright 2023 The RoboPianist Authors.
# Daxian V2 hand. Isolated from V3 (daxian_hand_constants.py).

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import math

_HERE = Path(__file__).resolve().parent
_DAXIAN_V2_HAND_DIR = _HERE / "third_party" / "daxian_v2"

NQ = 20  # Finger joints in the base MJCF (no forearm). pinky_rota is welded.
NU = 20

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

# Distal pad is authored into the tip meshes (assets/piano_tip/*_piano_tip.obj).
# Do not swap those for primitive spheres/capsules.
FINGERTIP_COLLISION_TYPE = "mesh"
FINGERTIP_COLLISION_RADIUS = 0.010
FINGERTIP_CAPSULE_HALF_LENGTH = 0.003
# Stiffer than the compiled-URDF defaults (solref 0.01 / impratio 10).
FINGERTIP_SOLREF = (0.004, 1.0)
FINGERTIP_SOLIMP = (0.99, 0.995, 0.001)
FINGER_FORCE_LIMIT = 2.0
FOREARM_MASS = 0.5
FOREARM_INERTIA = 1e-3

# Same attach convention as V3 so the piano task can share forearm DOFs.
HAND_BASE_RPY_DEG = (0.0, -90.0, 0.0)
LEFT_FOREARM_Z_SPIN_DEG = 180.0
RIGHT_FOREARM_Z_SPIN_DEG = -180.0
PALM_PITCH_UP_DEG = 0
THUMB_DOWN_ROLL_DEG = 0.0
LEFT_HAND_POSITION = (0.205, -0.3, 0.131)
RIGHT_HAND_POSITION = (0.205, 0.3, 0.131)
LEFT_HAND_RPY_DEG = HAND_BASE_RPY_DEG
RIGHT_HAND_RPY_DEG = HAND_BASE_RPY_DEG

# V2 has no rota_block / rotaback housing inside the palm.
THUMB_HOUSING_BODIES: Tuple[str, ...] = ()
THUMB_PALM_EXCLUDE: Tuple[Tuple[str, str], ...] = ()
THUMB_RANGE_JOINTS: Tuple[str, ...] = ()
# Palm collision mesh sits ~1 cm inside four-finger MCP/swing at q=0. Keep the
# palm visual-only so the solver is free to block finger–finger contacts.
PALM_VISUAL_ONLY_BODIES: Tuple[str, ...] = ("palm_link",)

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
# Pinned in reduced space (not policy). pinky_rota is welded at 0, not here.
PINNED_REST_CTRL: Dict[str, float] = {}
# Policy-driven; CanonicalSpec 0 / episode init. Not shown in the rest UI.
FINGER_SWING_REST_CTRL = {
    "index_swing_joint": 0.0,
    "mid_swing_joint": 0.0,
    "ring_swing_joint": 0.0,
    "pinky_swing_joint": 0.0,
}

LEFT_FOREARM_ROLL_RANGE = (-0.5, 0.0)
RIGHT_FOREARM_ROLL_RANGE = (0.0, 0.5)
# World-Z yaw. +ctrl = outwards (axis reflect on the left). Unlike roll, keep
# the same signed range on both hands so the motion Y-mirrors: both inwards
# only. CanonicalSpec 0 is no yaw (the 0 end of the range, not the midpoint).
LEFT_FOREARM_YAW_RANGE = (-0.6, 0.0)
RIGHT_FOREARM_YAW_RANGE = (-0.6, 0.0)
FOREARM_YAW_RANGE = LEFT_FOREARM_YAW_RANGE
# World +X (SAPIEN pos_x). +tz toward the player; negative reaches into the keys.
# CanonicalSpec 0 is attach x (the 0 end of the range). Same for both hands.
FOREARM_TZ_RANGE = (-0.050, 0.0)
# World +Z (SAPIEN pos_z). Negative lowers the palm onto the keys.
# CanonicalSpec 0 is attach z. Same for both hands.
FOREARM_TY_RANGE = (-0.04, 0.06)
# V2 training default. V3 still uses daxian_hand._DEFAULT_FOREARM_DOFS.
DEFAULT_FOREARM_DOFS: Tuple[str, ...] = (
    "forearm_tx",
    "forearm_ty",
    "forearm_tz",
    "forearm_yaw",
)
# Four-finger MCP only (not thumb). Training uses a shallow press range so
# policy  +1 is ~34°, not the full 90° URDF limit.
FOUR_FINGER_MCP_RANGE: Tuple[float, float] = (0.0, 0.6)


def rest_ctrl() -> Dict[str, float]:
    return {
        **THUMB_REST_CTRL,
        **FINGER_REST_CTRL,
        **PINNED_REST_CTRL,
        **FINGER_SWING_REST_CTRL,
    }


def range_containing_rest(
    lo_hi: Tuple[float, float], rest: float
) -> Tuple[float, float]:
    lo, hi = float(lo_hi[0]), float(lo_hi[1])
    rest = float(rest)
    return (min(lo, rest), max(hi, rest))


def thumb_policy_range(joint_name: str) -> Tuple[float, float]:
    raise KeyError(joint_name)


# Four-finger names used by PIP/DIP ↔ MCP coupling (not thumb).
FOUR_FINGERS: Tuple[str, ...] = ("index", "mid", "ring", "pinky")
PIP_DIP_LIMIT: Tuple[float, float] = (0.0, 1.57)
# When True and PIP/DIP are pinned, MIDI-assigned fingers IK-solve PIP/DIP so
# the distal axis (DIP +Z) points at world -Z. Idle fingers keep the
# pre-coupling PIP rest and set DIP = 0.
COUPLE_PIP_DIP_TO_MCP = True
# MIDI fingering 0=thumb … 4=pinky (per hand). Thumb is not PIP/DIP-coupled.
MJCF_FINGERING_TO_FOUR_FINGER: Dict[int, str] = {
    1: "index",
    2: "mid",
    3: "ring",
    4: "pinky",
}
# Closed-form fallback when FK is unavailable (default attach pose).
FOUR_FINGER_IK_FLEX_TARGET = 0.5 * math.pi
FOUR_FINGER_PIP_FRACTION = 0.4
# Distal +Z should align with this world direction (down onto the keys).
FOUR_FINGER_IK_TARGET_WORLD: Tuple[float, float, float] = (0.0, 0.0, -1.0)


def signed_flex_to_align(
    distal: Sequence[float],
    axis: Sequence[float],
    target: Sequence[float] = FOUR_FINGER_IK_TARGET_WORLD,
) -> float:
    """Radians about ``axis`` that rotate ``distal`` toward ``target``.

    Used with DIP-link +Z and MCP-link +Y after setting PIP=DIP=0 at the
    commanded MCP. Positive values are the PIP+DIP sum to apply.
    """
    dx, dy, dz = (float(distal[0]), float(distal[1]), float(distal[2]))
    ax, ay, az = (float(axis[0]), float(axis[1]), float(axis[2]))
    tx, ty, tz = (float(target[0]), float(target[1]), float(target[2]))
    an = math.sqrt(ax * ax + ay * ay + az * az)
    if an < 1e-12:
        return 0.0
    ax, ay, az = ax / an, ay / an, az / an
    da = dx * ax + dy * ay + dz * az
    ta = tx * ax + ty * ay + tz * az
    dpx, dpy, dpz = dx - ax * da, dy - ay * da, dz - az * da
    tpx, tpy, tpz = tx - ax * ta, ty - ay * ta, tz - az * ta
    nd = math.sqrt(dpx * dpx + dpy * dpy + dpz * dpz)
    nt = math.sqrt(tpx * tpx + tpy * tpy + tpz * tpz)
    if nd < 1e-8 or nt < 1e-8:
        return 0.0
    dpx, dpy, dpz = dpx / nd, dpy / nd, dpz / nd
    tpx, tpy, tpz = tpx / nt, tpy / nt, tpz / nt
    cx = dpy * tpz - dpz * tpy
    cy = dpz * tpx - dpx * tpz
    cz = dpx * tpy - dpy * tpx
    sin_a = ax * cx + ay * cy + az * cz
    cos_a = dpx * tpx + dpy * tpy + dpz * tpz
    return math.atan2(sin_a, cos_a)


def _pip_fraction(finger: str, rest: Optional[Dict[str, float]]) -> float:
    if rest is not None:
        pip0 = rest.get(f"{finger}_PIP_joint")
        dip0 = rest.get(f"{finger}_DIP_joint")
        if pip0 is not None and dip0 is not None and (pip0 + dip0) > 1e-8:
            return float(pip0) / float(pip0 + dip0)
    return FOUR_FINGER_PIP_FRACTION


def four_finger_from_mjcf_fingering(idx: int) -> Optional[str]:
    return MJCF_FINGERING_TO_FOUR_FINGER.get(int(idx))


def idle_four_finger_pip_dip(
    rest: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Pre-coupling idle: PIP at ``FINGER_REST_CTRL``, DIP straight (0)."""
    rest = rest if rest is not None else rest_ctrl()
    out: Dict[str, float] = {}
    for finger in FOUR_FINGERS:
        out[f"{finger}_PIP_joint"] = float(rest.get(f"{finger}_PIP_joint", 0.0))
        out[f"{finger}_DIP_joint"] = 0.0
    return out


def split_pip_dip(flex: float, pip_fraction: float) -> Tuple[float, float]:
    lo, hi = PIP_DIP_LIMIT
    s1 = max(0.0, min(float(flex), hi + hi))
    frac = min(1.0, max(0.0, float(pip_fraction)))
    pip = min(hi, max(lo, frac * s1))
    dip = min(hi, max(lo, s1 - pip))
    return pip, dip


def couple_four_finger_pip_dip(
    mcp: Dict[str, float],
    rest: Optional[Dict[str, float]] = None,
    flex_remaining: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """IK PIP/DIP so the fingertip distal axis is vertical onto the keys.

    ``flex_remaining[finger]`` is the FK-solved PIP+DIP (MCP already applied).
    Without it, uses ``pi/2 - MCP``, which matches the default attach pose.
    ``rest`` only supplies the PIP:DIP split; the sum is not taken from rest.
    """
    remaining = flex_remaining or {}
    out: Dict[str, float] = {}
    for finger in FOUR_FINGERS:
        mcp_key = f"{finger}_MCP_joint"
        mcp_val = float(mcp.get(finger, mcp.get(mcp_key, 0.0)))
        if finger in remaining:
            s1 = float(remaining[finger])
        else:
            s1 = FOUR_FINGER_IK_FLEX_TARGET - mcp_val
        pip, dip = split_pip_dip(s1, _pip_fraction(finger, rest))
        out[f"{finger}_PIP_joint"] = pip
        out[f"{finger}_DIP_joint"] = dip
    return out


# When reduced_action_space=True the policy commands these joints (plus
# forearm_tx / forearm_ty / forearm_tz / forearm_yaw). Four-finger PIP/DIP
# stay pinned and, if COUPLE_PIP_DIP_TO_MCP, MIDI-assigned fingers are IK-solved
# so the distal axis is vertical onto the keys (idle: PIP rest, DIP 0).
# unlock_four_finger_pip_dip=True disables both pin and couple. pinky_rota is
# welded at 0 (WELD_JOINTS), not a policy joint. Thumb MCP/PIP are free.
FOUR_FINGER_POLICY_JOINTS: Tuple[str, ...] = (
    "thumb_rota_joint",
    "thumb_swing_joint",
    "thumb_MCP_joint",
    "thumb_PIP_joint",
    "index_swing_joint",
    "index_MCP_joint",
    "mid_swing_joint",
    "mid_MCP_joint",
    "ring_swing_joint",
    "ring_MCP_joint",
    "pinky_swing_joint",
    "pinky_MCP_joint",
)
# Welded at the URDF/MJCF pose (q=0). Not a revolute joint, not actuated.
WELD_JOINTS: Tuple[str, ...] = ("pinky_rota_joint",)
PINNED_EXTRA_JOINTS: Tuple[str, ...] = ()
THUMB_PINNED_JOINTS: Tuple[str, ...] = ()
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
    *PINNED_EXTRA_JOINTS,
)


def pinned_joints(unlock_four_finger_pip_dip: bool = False) -> Tuple[str, ...]:
    if unlock_four_finger_pip_dip:
        return PINNED_EXTRA_JOINTS
    return FOUR_FINGER_FIXED_JOINTS


SKIP_THUMB_FINGERINGS: Tuple[int, ...] = ()

# AABB centres of the piano-tip meshes (right hand). Left is Y-mirrored.
FINGERTIP_COLLISION_POS: Dict[str, Tuple[float, float, float]] = {
    "thumb_PIP_link": (0.0000, 0.0093, 0.0274),
    "index_DIP_link": (-0.0093, 0.0000, 0.0274),
    "mid_DIP_link": (-0.0093, 0.0000, 0.0274),
    "ring_DIP_link": (-0.0093, 0.0000, 0.0274),
    "pinky_DIP_link": (-0.0093, 0.0000, 0.0274),
}

# Capsule long axis in the tip body frame. MuJoCo capsules default to +Z;
# SAPIEN/PhysX capsules default to +X. See fingertip_capsule_quat_*.
FINGERTIP_CAPSULE_AXIS: Dict[str, str] = {
    "thumb_PIP_link": "x",
    "index_DIP_link": "y",
    "mid_DIP_link": "y",
    "ring_DIP_link": "y",
    "pinky_DIP_link": "y",
}

_SQ2 = 0.70710678


def fingertip_capsule_quat_mujoco(tip_name: str) -> Tuple[float, float, float, float]:
    """wxyz rotating MuJoCo's +Z capsule onto the finger-width axis."""
    axis = FINGERTIP_CAPSULE_AXIS[tip_name]
    if axis == "y":
        return (_SQ2, _SQ2, 0.0, 0.0)
    if axis == "x":
        return (_SQ2, 0.0, _SQ2, 0.0)
    return (1.0, 0.0, 0.0, 0.0)


def fingertip_capsule_quat_sapien(tip_name: str) -> Tuple[float, float, float, float]:
    """wxyz rotating PhysX's +X capsule onto the finger-width axis."""
    axis = FINGERTIP_CAPSULE_AXIS[tip_name]
    if axis == "y":
        return (_SQ2, 0.0, 0.0, _SQ2)
    return (1.0, 0.0, 0.0, 0.0)


def mirror_y_pos(pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (float(pos[0]), -float(pos[1]), float(pos[2]))


def palmar_site_pos(
    tip_name: str,
    pos: Tuple[float, float, float],
    radius: float,
    side: str,
) -> Tuple[float, float, float]:
    """Contact site on the palmar surface of the capsule (this-hand frame)."""
    x, y, z = (float(pos[0]), float(pos[1]), float(pos[2]))
    r = float(radius)
    if tip_name.startswith("thumb"):
        sign = 1.0 if side == "right" else -1.0
        return (x, y + sign * r, z)
    return (x - r, y, z)


def fingertip_site_pos_right(
    collision_pos: Dict[str, Tuple[float, float, float]] | None = None,
    radius: float | None = None,
) -> Dict[str, Tuple[float, float, float]]:
    pos = collision_pos or FINGERTIP_COLLISION_POS
    r = FINGERTIP_COLLISION_RADIUS if radius is None else float(radius)
    return {
        tip: palmar_site_pos(tip, p, r, "right") for tip, p in pos.items()
    }


def _fmt_xyz(pos: Tuple[float, float, float]) -> str:
    return f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"


def _fmt_quat(q: Tuple[float, float, float, float]) -> str:
    return f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"


def tip_capsule_mjcf_attribs(
    tip_name: str,
    side: str,
    collision_pos: Dict[str, Tuple[float, float, float]] | None = None,
    radius: float | None = None,
    half_length: float | None = None,
) -> Dict[str, str]:
    """Attribute dict for a distal capsule <geom> (this-hand frame)."""
    src = (collision_pos or FINGERTIP_COLLISION_POS)[tip_name]
    pos = mirror_y_pos(src) if side == "left" else tuple(float(x) for x in src)
    r = FINGERTIP_COLLISION_RADIUS if radius is None else float(radius)
    h = (
        FINGERTIP_CAPSULE_HALF_LENGTH
        if half_length is None
        else float(half_length)
    )
    quat = fingertip_capsule_quat_mujoco(tip_name)
    return {
        "type": "capsule",
        "size": f"{r:.4f} {h:.4f}",
        "pos": _fmt_xyz(pos),
        "quat": _fmt_quat(quat),
        "group": "0",
    }


def apply_tip_capsules_to_mjcf(
    xml_path,
    prefix: str,
    side: str,
    collision_pos: Dict[str, Tuple[float, float, float]] | None = None,
    radius: float | None = None,
    half_length: float | None = None,
    solref: str = "0.004 1",
    solimp: str = "0.99 0.995 0.001",
    impratio: str = "20",
) -> None:
    """Replace distal mesh colliders with capsules and stiffen default contacts."""
    import xml.etree.ElementTree as ET

    path = Path(xml_path)
    tree = ET.parse(path)
    root = tree.getroot()
    option = root.find("option")
    if option is not None:
        option.set("impratio", impratio)
    default_geom = root.find("./default/geom")
    if default_geom is not None:
        default_geom.set("solref", solref)
        default_geom.set("solimp", solimp)

    r = FINGERTIP_COLLISION_RADIUS if radius is None else float(radius)
    for body in root.iter("body"):
        name = body.get("name") or ""
        if not name.startswith(prefix):
            continue
        short = name[len(prefix) :]
        if short not in FINGERTIP_BODIES:
            continue
        attrs = tip_capsule_mjcf_attribs(
            short, side, collision_pos, radius, half_length
        )
        rgb = FINGERTIP_COLORS[FINGERTIP_BODIES.index(short)]
        attrs["rgba"] = f"{rgb[0]:.2f} {rgb[1]:.2f} {rgb[2]:.2f} 0.85"
        attrs["name"] = f"{prefix}{short}_tip_capsule"
        for geom in list(body.findall("geom")):
            if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
                continue
            if geom.get("mesh") or geom.get("type") == "capsule":
                geom.attrib.clear()
                geom.attrib.update(attrs)
        site_pos = palmar_site_pos(short, tuple(float(x) for x in attrs["pos"].split()), r, side)
        for site in body.findall("site"):
            site.set("pos", _fmt_xyz(site_pos))
    ET.indent(root, space="  ")
    tree.write(path, encoding="unicode", xml_declaration=True)


FINGERTIP_SITE_POS: Dict[str, Tuple[float, float, float]] = {
    "thumb_PIP_link": (0.0000, 0.0232, 0.0274),
    "index_DIP_link": (-0.0232, 0.0000, 0.0274),
    "mid_DIP_link": (-0.0232, 0.0000, 0.0274),
    "ring_DIP_link": (-0.0232, 0.0000, 0.0274),
    "pinky_DIP_link": (-0.0232, 0.0000, 0.0274),
}

FINGERTIP_COLORS: Tuple[Tuple[float, float, float], ...] = (
    (0.8, 0.2, 0.8),
    (1.0, 0.22, 0.0),
    (0.2, 0.8, 0.8),
    (0.2, 0.2, 0.8),
    (0.8, 0.8, 0.2),
)

RIGHT_DAXIAN_HAND_XML = _DAXIAN_V2_HAND_DIR / "right_hand.xml"
LEFT_DAXIAN_HAND_XML = _DAXIAN_V2_HAND_DIR / "left_hand.xml"
