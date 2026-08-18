#!/usr/bin/env python3
"""Interactive SAPIEN preview for the current four-finger training setup.

Scene: left Daxian hand + 88-key piano (RoboPianist geometry).
Tune attach pose, palm pitch, wrist roll, MCP (presses keys), and rest joints.
When you are done, print a left→right mirrored snippet for daxian_hand_constants.py.

Usage (conda env handel, needs a display):

  conda activate handel
  cd /home/houjue/pianist_daxian/robopianist
  python examples/sapien_left_hand_tune.py

Un-pause the viewer (Control window) so physics runs; then raise MCP to press keys.
"""

from __future__ import annotations

import importlib.util
import math
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import sapien
from sapien import internal_renderer as R
from sapien.utils.viewer.plugin import Plugin
from sapien.utils.viewer.viewer import (
    ArticulationWindow,
    ContactWindow,
    ControlWindow,
    EntityWindow,
    PathWindow,
    RenderOptionsWindow,
    SceneWindow,
    SettingWindow,
    TransformWindow,
    Viewer,
)
from transforms3d.euler import euler2quat
from transforms3d.quaternions import axangle2quat, qmult

REPO = Path(__file__).resolve().parents[1]


def _load_py(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


C = _load_py(
    "daxian_hand_constants",
    REPO / "robopianist" / "models" / "hands" / "daxian_hand_constants.py",
)
P = _load_py(
    "piano_constants",
    REPO / "robopianist" / "models" / "piano" / "piano_constants.py",
)

_HERE = Path(__file__).resolve()
_V3 = _HERE.parents[2] / "daxian_V3"
_LEFT_URDF = _V3 / "urdf" / "daxian_left.urdf"
_MESH_OBJ = REPO / "robopianist" / "models" / "hands" / "third_party" / "daxian" / "assets"

_FINGERS = ("index", "mid", "ring", "pinky")
_FLEX_MAX = 1.57
_LEFT_ROLL = C.LEFT_FOREARM_ROLL_RANGE
_HAND_GROUP = [1, 2, 0, 0]
_KEY_GROUP = [2, 1, 0, 0]
_Z90 = [0.70710678, 0.0, 0.0, 0.70710678]  # wxyz, +90° about Z → joint X = world Y


def _stem_to_obj(filename: str) -> Path:
    stem = Path(filename).stem
    if stem == "base_link":
        stem = "left_base_link"
    return _MESH_OBJ / f"{stem}.obj"


def rewrite_left_urdf(dst: Path) -> Path:
    text = _LEFT_URDF.read_text()

    def repl(match: re.Match) -> str:
        obj = _stem_to_obj(match.group(1))
        if not obj.is_file():
            raise FileNotFoundError(f"missing mesh for {match.group(1)}: {obj}")
        return f'filename="{obj}"'

    dst.write_text(re.sub(r'filename="([^"]+)"', repl, text))
    return dst


def _q_wxyz(q) -> list[float]:
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def compose_attach_quat(
    rpy_deg, spin_deg: float, pitch_deg: float, roll_rad: float
) -> list[float]:
    """Same composition as suite/tasks/base.py.

    ``roll_rad`` is extra Euler X (rpy roll), not a world-X rotation after pitch.
    +wrist_roll has the same effect as adding the same angle to rpy roll.
    """
    rx, ry, rz = np.radians(rpy_deg)
    rx = rx + float(roll_rad)
    q = euler2quat(rx, ry, rz, axes="sxyz")
    q = qmult(q, axangle2quat((0.0, 0.0, 1.0), math.radians(spin_deg)))
    if pitch_deg:
        q = qmult(axangle2quat((0.0, 1.0, 0.0), math.radians(pitch_deg)), q)
    return _q_wxyz(q)


# MIDI indices of the 52 white keys, same order as piano_mjcf.py.
_WHITE_MIDI = [
    0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 24, 26, 27, 29, 31, 32,
    34, 36, 38, 39, 41, 43, 44, 46, 48, 50, 51, 53, 55, 56, 58, 60, 62, 63,
    65, 67, 68, 70, 72, 74, 75, 77, 79, 80, 82, 84, 86, 87,
]
_BLACK_TWIN_MIDI = [4, 6, 16, 18, 28, 30, 40, 42, 52, 54, 64, 66, 76, 78]
_BLACK_TRIPLET_MIDI = [
    1, 9, 11, 13, 21, 23, 25, 33, 35, 37, 45, 47, 49, 57, 59, 61, 69, 71, 73,
    81, 83, 85,
]


def _key_centers() -> tuple[dict[int, float], dict[int, float]]:
    step = P.WHITE_KEY_WIDTH + P.SPACING_BETWEEN_WHITE_KEYS
    white_ys = {}
    for i, midi in enumerate(_WHITE_MIDI):
        white_ys[midi] = float(
            -P.PIANO_LENGTH * 0.5 + P.WHITE_KEY_WIDTH * 0.5 + i * step
        )
    black_ys: dict[int, float] = {}
    black_ys[_BLACK_TRIPLET_MIDI[0]] = float(
        P.WHITE_KEY_WIDTH + 0.5 * (-P.PIANO_LENGTH + P.SPACING_BETWEEN_WHITE_KEYS)
    )
    n = 0
    for twin_index in range(2, P.NUM_WHITE_KEYS - 1, 7):
        for j in range(2):
            black_ys[_BLACK_TWIN_MIDI[n]] = float(
                -P.PIANO_LENGTH * 0.5 + (j + 1) * step + twin_index * step
            )
            n += 1
    n = 1
    for triplet_index in range(5, P.NUM_WHITE_KEYS - 1, 7):
        for j in range(3):
            black_ys[_BLACK_TRIPLET_MIDI[n]] = float(
                -P.PIANO_LENGTH * 0.5 + (j + 1) * step + triplet_index * step
            )
            n += 1
    return white_ys, black_ys


def build_piano(scene: sapien.Scene) -> list:
    """88 hinged keys + static fallboard, RoboPianist sizes."""
    white_ys, black_ys = _key_centers()
    mat = sapien.physx.PhysxMaterial(0.6, 0.5, 0.0)
    arts = []

    base = scene.create_actor_builder()
    base.add_box_collision(
        sapien.Pose(p=P.BASE_POS), half_size=P.BASE_SIZE, material=mat
    )
    base.add_box_visual(sapien.Pose(p=P.BASE_POS), P.BASE_SIZE, P.BASE_COLOR[:3])
    base.set_physx_body_type("static")
    base.build().name = "piano_base"

    def add_key(midi: int, y: float, white: bool):
        length = P.WHITE_KEY_LENGTH if white else P.BLACK_KEY_LENGTH
        width = P.WHITE_KEY_WIDTH if white else P.BLACK_KEY_WIDTH
        height = P.WHITE_KEY_HEIGHT if white else P.BLACK_KEY_HEIGHT
        x = P.WHITE_KEY_X_OFFSET if white else P.BLACK_KEY_X_OFFSET
        z = P.WHITE_KEY_Z_OFFSET if white else P.BLACK_KEY_Z_OFFSET
        mass = P.WHITE_KEY_MASS if white else P.BLACK_KEY_MASS
        lo_hi = (0.0, P.WHITE_KEY_JOINT_MAX_ANGLE if white else P.BLACK_KEY_JOINT_MAX_ANGLE)
        stiff = P.WHITE_KEY_STIFFNESS if white else P.BLACK_KEY_STIFFNESS
        damp = P.WHITE_JOINT_DAMPING if white else P.BLACK_JOINT_DAMPING
        color = P.WHITE_KEY_COLOR[:3] if white else P.BLACK_KEY_COLOR[:3]
        half = [length / 2, width / 2, height / 2]
        hinge = [x - length / 2, y, z]
        box_in_child = sapien.Pose(p=[length / 2, 0.0, 0.0])

        b = scene.create_articulation_builder()
        root = b.create_link_builder()
        root.set_name(f"key_anchor_{midi}")
        child = b.create_link_builder(root)
        child.set_name(f"{'white' if white else 'black'}_key_{midi}")
        child.set_joint_name(f"key_joint_{midi}")
        child.set_joint_properties(
            "revolute",
            lo_hi,
            sapien.Pose(p=hinge, q=_Z90),
            sapien.Pose(q=_Z90),
            friction=0.01,
            damping=damp,
        )
        child.set_mass_and_inertia(mass, sapien.Pose(), [1e-5, 1e-5, 1e-5])
        child.add_box_collision(box_in_child, half, material=mat, density=mass / (length * width * height))
        child.add_box_visual(box_in_child, half, color)
        child.collision_groups = list(_KEY_GROUP)
        b.set_initial_pose(sapien.Pose())
        art = b.build(fix_root_link=True)
        art.name = f"key_{midi}"
        j = art.get_active_joints()[0]
        j.set_drive_property(stiffness=stiff, damping=damp, force_limit=2.0, mode="force")
        j.set_drive_target(0.0)
        arts.append(art)

    for midi, y in white_ys.items():
        add_key(midi, y, True)
    for midi, y in black_ys.items():
        add_key(midi, y, False)
    return arts


def _bare_joint(name: str) -> str:
    for prefix in ("left_hand_", "Left_hand_", "right_hand_", "Right_hand_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _bare_link(name: str) -> str:
    for prefix in ("left_hand_", "Left_hand_", "right_hand_", "Right_hand_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    if name in ("base_link", "Left_hand_palm_link", "Right_hand_palm_link"):
        return name
    return name


def setup_hand(scene: sapien.Scene, hand):
    """Disable mesh collisions, add fingertip spheres. Visual markers are separate actors."""
    hand.set_name("daxian_left")
    for link in hand.get_links():
        link.disable_gravity = True
        for shape in link.get_collision_shapes():
            shape.set_collision_groups([0, 0, 0, 0])

    mat = sapien.physx.PhysxMaterial(0.8, 0.6, 0.0)
    tip_markers = []
    for body, pos_r in C.FINGERTIP_COLLISION_POS.items():
        link = None
        for cand in hand.get_links():
            if _bare_link(cand.name) == body or cand.name.endswith(body):
                link = cand
                break
        if link is None:
            print(f"[warn] no link for fingertip {body}")
            continue
        pos = (float(pos_r[0]), -float(pos_r[1]), float(pos_r[2]))
        local = sapien.Pose(p=pos)
        sphere = sapien.physx.PhysxCollisionShapeSphere(C.FINGERTIP_COLLISION_RADIUS, mat)
        sphere.set_local_pose(local)
        sphere.set_collision_groups(list(_HAND_GROUP))
        link.attach(sphere)

        # Cannot attach extra visuals to a render body already on an entity.
        builder = scene.create_actor_builder()
        builder.add_sphere_visual(
            radius=C.FINGERTIP_COLLISION_RADIUS, material=[0.1, 0.9, 0.9]
        )
        marker = builder.build_kinematic(name=f"tip_{body}")
        tip_markers.append((link, local, marker))

    joints = {}
    for j in hand.get_active_joints():
        bare = _bare_joint(j.name)
        joints[bare] = j
        lim = np.asarray(j.limit).reshape(-1)
        lo, hi = float(lim[0]), float(lim[1])
        if not np.isfinite(lo):
            lo = -3.14
        if not np.isfinite(hi):
            hi = 3.14
        force = C.FINGER_FORCE_LIMIT
        j.set_drive_property(stiffness=80.0, damping=4.0, force_limit=force, mode="force")
        print(f"  joint {j.name} -> {bare}  range=[{lo:.3f}, {hi:.3f}]")
    return joints, tip_markers


def update_tip_markers(tip_markers) -> None:
    for link, local, marker in tip_markers:
        marker.set_pose(link.entity_pose * local)


class TunePlugin(Plugin):
    def __init__(self, app: "App"):
        self.app = app
        self.ui_window = None

    def get_ui_windows(self):
        self._rebuild()
        return [self.ui_window] if self.ui_window else []

    def _rebuild(self):
        a = self.app
        win = R.UIWindow().Label("Four-finger tune (left)").Pos(10, 10).Size(420, 760)
        win.append(R.UIDisplayText().Text(
            "Drag one joint (this panel or Articulation). Others stay locked on key contact."
        ))

        pose = R.UISection().Label("Attach pose").Expanded(True)
        pose.append(
            R.UISliderFloat().Label("pos x (m)").Min(0.05).Max(0.40).Bind(a, "pos_x"),
            R.UISliderFloat().Label("pos y (m)").Min(-0.70).Max(0.10).Bind(a, "pos_y"),
            R.UISliderFloat().Label("pos z (m)").Min(0.00).Max(0.12).Bind(a, "pos_z"),
            R.UISliderFloat().Label("rpy roll deg").Min(-180).Max(180).Bind(a, "rpy_r"),
            R.UISliderFloat().Label("rpy pitch deg").Min(-180).Max(180).Bind(a, "rpy_p"),
            R.UISliderFloat().Label("rpy yaw deg").Min(-180).Max(180).Bind(a, "rpy_y"),
            R.UISliderFloat().Label("palm pitch_up deg").Min(-20).Max(60).Bind(a, "pitch_up"),
            R.UISliderFloat().Label("forearm z-spin deg").Min(-180).Max(180).Bind(a, "z_spin"),
            R.UISliderFloat()
            .Label("wrist roll rad (= extra rpy roll)")
            .Min(float(_LEFT_ROLL[0]))
            .Max(float(_LEFT_ROLL[1]))
            .Bind(a, "roll"),
        )
        win.append(pose)

        mcp = R.UISection().Label("MCP press (0 = rest/straight, 1.57 = full)").Expanded(True)
        mcp.append(
            R.UISliderFloat().Label("MCP all").Min(0.0).Max(_FLEX_MAX).Bind(a, "mcp_all"),
            R.UISliderFloat().Label("index MCP").Min(0.0).Max(_FLEX_MAX).Bind(a, "mcp_index"),
            R.UISliderFloat().Label("mid MCP").Min(0.0).Max(_FLEX_MAX).Bind(a, "mcp_mid"),
            R.UISliderFloat().Label("ring MCP").Min(0.0).Max(_FLEX_MAX).Bind(a, "mcp_ring"),
            R.UISliderFloat().Label("pinky MCP").Min(0.0).Max(_FLEX_MAX).Bind(a, "mcp_pinky"),
        )
        win.append(mcp)

        rest_f = R.UISection().Label("FINGER_REST_CTRL").Expanded(True)
        for name in C.FINGER_REST_CTRL:
            attr = "rest_" + name
            rest_f.append(
                R.UISliderFloat().Label(name).Min(0.0).Max(_FLEX_MAX).Bind(a, attr)
            )
        win.append(rest_f)

        rest_t = R.UISection().Label("THUMB_REST_CTRL").Expanded(False)
        thumb_lim = {
            "thumb_rota_block_joint": C.thumb_policy_range("thumb_rota_block_joint"),
            "thumb_rotaback_joint2": C.thumb_policy_range("thumb_rotaback_joint2"),
            "thumb_MCP_joint": (0.0, _FLEX_MAX),
            "thumb_PIP_joint": (0.0, _FLEX_MAX),
        }
        for name in C.THUMB_REST_CTRL:
            lo, hi = thumb_lim[name]
            rest_t.append(
                R.UISliderFloat()
                .Label(name)
                .Min(float(lo))
                .Max(float(hi))
                .Bind(a, "rest_" + name)
            )
        win.append(rest_t)

        win.append(
            R.UIButton().Label("Reset to training defaults").Callback(lambda _: a.reset()),
            R.UIButton().Label("Print left + mapped right constants").Callback(
                lambda _: a.print_mapping()
            ),
        )
        self.ui_window = win


class App:
    def __init__(self, hand, joints: dict):
        self.hand = hand
        self.joints = joints
        self._last_mcp_all = 0.0
        self._held_qpos = None
        self._last_ui_q = None
        self._active = set()
        self._active_hold = 0
        self.reset()

    def reset(self):
        x, y, z = C.LEFT_HAND_POSITION
        r, p, yw = C.HAND_BASE_RPY_DEG
        self.pos_x, self.pos_y, self.pos_z = float(x), float(y), float(z)
        self.rpy_r, self.rpy_p, self.rpy_y = float(r), float(p), float(yw)
        self.pitch_up = float(C.PALM_PITCH_UP_DEG)
        self.z_spin = float(C.LEFT_FOREARM_Z_SPIN_DEG)
        self.roll = 0.0
        self.mcp_all = 0.0
        self.mcp_index = self.mcp_mid = self.mcp_ring = self.mcp_pinky = 0.0
        self._last_mcp_all = 0.0
        for name, val in {**C.THUMB_REST_CTRL, **C.FINGER_REST_CTRL}.items():
            setattr(self, "rest_" + name, float(val))
        self._held_qpos = None
        self._last_ui_q = None
        self._active = set()
        self._active_hold = 0

    def _sync_mcp_all(self):
        if abs(self.mcp_all - self._last_mcp_all) > 1e-6:
            self.mcp_index = self.mcp_mid = self.mcp_ring = self.mcp_pinky = float(
                self.mcp_all
            )
            self._last_mcp_all = float(self.mcp_all)

    def _ui_qpos(self) -> np.ndarray:
        """Commanded angles for every active joint (swing held at 0 unless dragged)."""
        self._sync_mcp_all()
        targets = {**C.rest_ctrl()}
        for name in {**C.THUMB_REST_CTRL, **C.FINGER_REST_CTRL}:
            targets[name] = float(getattr(self, "rest_" + name))
        targets["index_MCP_joint"] = float(self.mcp_index)
        targets["mid_MCP_joint"] = float(self.mcp_mid)
        targets["ring_MCP_joint"] = float(self.mcp_ring)
        targets["pinky_MCP_joint"] = float(self.mcp_pinky)
        for finger in _FINGERS:
            targets.setdefault(f"{finger}_swing_joint", 0.0)

        q = np.zeros(self.hand.dof, dtype=np.float32)
        for i, j in enumerate(self.hand.get_active_joints()):
            bare = _bare_joint(j.name)
            lim = np.asarray(j.limit).reshape(-1)
            val = float(targets.get(bare, 0.0))
            if lim.size >= 2:
                lo, hi = float(lim[0]), float(lim[1])
            else:
                lo, hi = val, val
            if bare in C.THUMB_REST_CTRL:
                lo, hi = C.range_containing_rest((lo, hi), val)
            q[i] = float(np.clip(val, lo, hi))
        return q

    def apply(self):
        """Follow sliders, but freeze every joint the user is not dragging.

        Contact with keys therefore cannot fold PIP/DIP/swing or the other fingers.
        """
        q_ui = self._ui_qpos()
        q_now = np.asarray(self.hand.get_qpos(), dtype=np.float32).reshape(-1)
        if self._held_qpos is None:
            self._held_qpos = q_ui.copy()
            self._last_ui_q = q_ui.copy()

        ui_delta = np.abs(q_ui - self._last_ui_q)
        phys_delta = np.abs(q_now - self._held_qpos)
        ui_changed = np.where(ui_delta > 1e-4)[0].tolist()
        phys_changed = np.where(phys_delta > 2e-3)[0].tolist()

        active = set(ui_changed)
        if not active and phys_changed:
            # Articulation-window drag: treat the joint that moved most as the one
            # being edited; lock the rest even if contact moved them too.
            active = {int(np.argmax(phys_delta))}
        if active:
            self._active = active
            self._active_hold = 20
        elif self._active_hold > 0:
            self._active_hold -= 1
        else:
            self._active = set()

        held = self._held_qpos.copy()
        for i in self._active:
            held[i] = q_ui[i] if i in ui_changed else q_now[i]
        if not self._active:
            # No drag: keep the pose the sliders ask for (and lock swings at 0).
            held = q_ui.copy()
        self._held_qpos = held
        self._last_ui_q = q_ui.copy()

        q = compose_attach_quat(
            (self.rpy_r, self.rpy_p, self.rpy_y),
            self.z_spin,
            self.pitch_up,
            self.roll,
        )
        self.hand.set_root_pose(sapien.Pose(p=[self.pos_x, self.pos_y, self.pos_z], q=q))
        for i, j in enumerate(self.hand.get_active_joints()):
            j.set_drive_target(float(held[i]))
        self.hand.set_qpos(held)
        self.hand.set_qvel(np.zeros_like(held))

    def hold_locked(self):
        """Call after scene.step() so contact cannot leave other joints bent."""
        if self._held_qpos is None:
            return
        self.hand.set_qpos(self._held_qpos)
        self.hand.set_qvel(np.zeros_like(self._held_qpos))

    def print_mapping(self):
        left_pos = (self.pos_x, self.pos_y, self.pos_z)
        right_pos = (self.pos_x, -self.pos_y, self.pos_z)
        rpy = (self.rpy_r, self.rpy_p, self.rpy_y)
        thumb = {n: float(getattr(self, "rest_" + n)) for n in C.THUMB_REST_CTRL}
        finger = {n: float(getattr(self, "rest_" + n)) for n in C.FINGER_REST_CTRL}

        def fmt_dict(d):
            lines = ["{"]
            for k, v in d.items():
                lines.append(f'    "{k}": {v:.4f},')
            lines.append("}")
            return "\n".join(lines)

        text = f"""
# --- current LEFT (this preview) ---
LEFT_HAND_POSITION = ({left_pos[0]:.4f}, {left_pos[1]:.4f}, {left_pos[2]:.4f})
LEFT_FOREARM_Z_SPIN_DEG = {self.z_spin:.1f}
forearm_roll (left ctrl) = {self.roll:.4f} rad   # extra rpy roll; training left {C.LEFT_FOREARM_ROLL_RANGE}, right {C.RIGHT_FOREARM_ROLL_RANGE}

# --- mapped RIGHT (y-mirror; same rpy / pitch / rest) ---
# Paste into robopianist/models/hands/daxian_hand_constants.py
HAND_BASE_RPY_DEG = ({rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f})
PALM_PITCH_UP_DEG = {self.pitch_up:.1f}
LEFT_HAND_POSITION = ({left_pos[0]:.4f}, {left_pos[1]:.4f}, {left_pos[2]:.4f})
RIGHT_HAND_POSITION = ({right_pos[0]:.4f}, {right_pos[1]:.4f}, {right_pos[2]:.4f})
LEFT_FOREARM_Z_SPIN_DEG = {self.z_spin:.1f}
RIGHT_FOREARM_Z_SPIN_DEG = {-self.z_spin:.1f}

THUMB_REST_CTRL = {fmt_dict(thumb)}
FINGER_REST_CTRL = {fmt_dict(finger)}

# MCP press in this session (not rest; policy can go to 1.57):
#   index={self.mcp_index:.3f}  mid={self.mcp_mid:.3f}  ring={self.mcp_ring:.3f}  pinky={self.mcp_pinky:.3f}
"""
        print(text)
        out = Path("/tmp/daxian_hand_tune_mapped.py")
        out.write_text(text)
        print(f"wrote {out}")


def main():
    if not _LEFT_URDF.is_file():
        raise SystemExit(f"missing {_LEFT_URDF}")
    if not _MESH_OBJ.is_dir():
        raise SystemExit(f"missing OBJ dir {_MESH_OBJ}")

    sapien.physx.set_default_material(0.5, 0.4, 0.0)
    scene = sapien.Scene()
    scene.set_timestep(0.005)
    scene.add_ground(altitude=-0.02, render_half_size=[1.5, 1.5])
    scene.set_ambient_light([0.45, 0.45, 0.45])
    scene.add_directional_light([0.4, 0.2, -1], [0.8, 0.8, 0.8])
    scene.add_point_light([0.4, -0.4, 0.6], [0.6, 0.6, 0.6])

    print("building piano…")
    build_piano(scene)

    tmp = Path(tempfile.mkdtemp(prefix="daxian_sapien_")) / "left.urdf"
    rewrite_left_urdf(tmp)
    print(f"loading hand from {tmp}")
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = False
    hand = loader.load(str(tmp))
    if hand is None:
        raise SystemExit("URDF load failed")
    print("hand joints:")
    joints, tip_markers = setup_hand(scene, hand)

    app = App(hand, joints)
    app.apply()
    update_tip_markers(tip_markers)

    plugin = TunePlugin(app)
    viewer = Viewer(
        plugins=[
            plugin,
            PathWindow(),
            ContactWindow(),
            SettingWindow(),
            TransformWindow(),
            RenderOptionsWindow(),
            ControlWindow(),
            SceneWindow(),
            EntityWindow(),
            ArticulationWindow(),
        ]
    )
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=0.55, y=-0.30, z=0.28)
    viewer.set_camera_rpy(r=0, p=-0.45, y=3.14)
    viewer.window.set_camera_parameters(near=0.02, far=10, fovy=0.9)
    viewer.paused = False

    print(
        "\nDrag one joint at a time; the rest stay locked when the fingertip hits a key.\n"
        "Raise MCP to press. Pause in Control if the hand jitters while dragging pose.\n"
        "Button 'Print left + mapped right constants' dumps a snippet to paste.\n"
    )

    while not viewer.closed:
        app.apply()
        if not viewer.paused:
            scene.step()
            app.hold_locked()
        update_tip_markers(tip_markers)
        scene.update_render()
        viewer.render()


if __name__ == "__main__":
    os.chdir(REPO)
    main()
