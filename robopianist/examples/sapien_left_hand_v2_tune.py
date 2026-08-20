#!/usr/bin/env python3
"""Interactive SAPIEN preview for the Tsensei 达显 V2 left hand.

Same piano + attach-pose tuner as examples/sapien_left_hand_tune.py (V3),
but loads daxian_V2 URDF / OBJ and daxian_v2_hand_constants.py.

V2 kinematics differ from V3: thumb is rota + swing (no rota_block / rotaback),
and the little finger's pinky_rota is welded at 0 (not a joint).

Training (reduced space) drives four-finger MCP/swing and all thumb joints.
PIP/DIP are pinned and IK-solved so the fingertip distal axis is vertical
onto the keys (capsule approaches the key plane along world -Z). This script
exposes attach pose, MCP preview (same IK), and live PIP/DIP.

Usage (needs the xrdp desktop, DISPLAY=:10). Do not run this inside
the training container `daxianpianist-dev-1` (no X11 socket).

  # on the Linux host, after connecting xrdp :3391:
  bash docker/run_sapien_v2.sh
"""

from __future__ import annotations

import importlib.util
import os
import re
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_ICD = REPO.parent / "docker" / "nvidia_icd.json"
if _ICD.is_file():
    os.environ.setdefault("VK_ICD_FILENAMES", str(_ICD))

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


def _load_py(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


C = _load_py(
    "daxian_v2_hand_constants",
    REPO / "robopianist" / "models" / "hands" / "daxian_v2_hand_constants.py",
)
V3 = _load_py("sapien_left_hand_tune", Path(__file__).with_name("sapien_left_hand_tune.py"))

_HERE = Path(__file__).resolve()
_V2 = _HERE.parents[2] / "daxian_V2"
_LEFT_URDF = _V2 / "urdf" / "daxian__hand_left_v1.urdf"
_MESH_OBJ = REPO / "robopianist" / "models" / "hands" / "third_party" / "daxian_v2" / "assets"

_FINGERS = ("index", "mid", "ring", "pinky")
_FLEX_MAX = 1.57
_YAW = getattr(C, "LEFT_FOREARM_YAW_RANGE", getattr(C, "FOREARM_YAW_RANGE", (-0.6, 0.0)))
_TZ = getattr(C, "FOREARM_TZ_RANGE", (-0.04, 0.0))
_TY = getattr(C, "FOREARM_TY_RANGE", (-0.04, 0.06))
_HAND_GROUP = [1, 2, 0, 0]


def _stem_to_obj(filename: str) -> Path:
    stem = Path(filename).stem
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


def setup_hand(scene: sapien.Scene, hand):
    hand.set_name("daxian_v2_left")
    for link in hand.get_links():
        link.disable_gravity = True
        bare = V3._bare_link(link.name)
        is_tip = bare in C.FINGERTIP_BODIES or any(
            link.name.endswith(b) for b in C.FINGERTIP_BODIES
        )
        for shape in link.get_collision_shapes():
            shape.set_collision_groups(list(_HAND_GROUP) if is_tip else [0, 0, 0, 0])

    joints = {}
    for j in hand.get_active_joints():
        bare = V3._bare_joint(j.name)
        joints[bare] = j
        lim = np.asarray(j.limit).reshape(-1)
        lo, hi = float(lim[0]), float(lim[1])
        if not np.isfinite(lo):
            lo = -3.14
        if not np.isfinite(hi):
            hi = 3.14
        j.set_drive_property(
            stiffness=80.0, damping=4.0, force_limit=C.FINGER_FORCE_LIMIT, mode="force"
        )
        print(f"  joint {j.name} -> {bare}  range=[{lo:.3f}, {hi:.3f}]")
    return joints


class TunePlugin(Plugin):
    def __init__(self, app: "App"):
        self.app = app
        self.ui_window = None

    def get_ui_windows(self):
        self._rebuild()
        return [self.ui_window] if self.ui_window else []

    def _rebuild(self):
        a = self.app
        win = R.UIWindow().Label("达显 V2 left tune").Pos(10, 10).Size(420, 980)
        win.append(
            R.UIDisplayText().Text(
                "Tsensei 达显 V2. Drag one joint; others stay locked on key contact."
            )
        )

        pose = R.UISection().Label("Attach pose").Expanded(True)
        pose.append(
            R.UISliderFloat().Label("pos x (m)").Min(0.05).Max(0.40).Bind(a, "pos_x"),
            R.UISliderFloat().Label("pos y (m)").Min(-0.70).Max(0.10).Bind(a, "pos_y"),
            R.UISliderFloat().Label("pos z (m)").Min(0.00).Max(0.30).Bind(a, "pos_z"),
            R.UISliderFloat().Label("rpy roll deg (world X)").Min(-180).Max(180).Bind(a, "rpy_r"),
            R.UISliderFloat().Label("rpy pitch deg").Min(-180).Max(180).Bind(a, "rpy_p"),
            R.UISliderFloat().Label("rpy yaw deg (world Z)").Min(-180).Max(180).Bind(a, "rpy_y"),
            R.UISliderFloat().Label("palm pitch_up deg").Min(-20).Max(60).Bind(a, "pitch_up"),
            R.UISliderFloat().Label("forearm z-spin deg").Min(-180).Max(180).Bind(a, "z_spin"),
            R.UISliderFloat()
            .Label("forearm_tz m (world +X / pos_x, 0 = attach)")
            .Min(float(_TZ[0]))
            .Max(float(_TZ[1]))
            .Bind(a, "tz"),
            R.UISliderFloat()
            .Label("forearm_ty m (world +Z / pos_z, 0 = attach)")
            .Min(float(_TY[0]))
            .Max(float(_TY[1]))
            .Bind(a, "ty"),
            R.UISliderFloat()
            .Label("forearm_yaw rad (world Z, + = outwards, range inwards-only)")
            .Min(float(_YAW[0]))
            .Max(float(_YAW[1]))
            .Bind(a, "yaw"),
        )
        win.append(pose)

        mcp_hi = float(C.FOUR_FINGER_MCP_RANGE[1])
        mcp = R.UISection().Label(
            f"MCP press (IK PIP/DIP: tip vertical to keys; train max {mcp_hi:.2f})"
        ).Expanded(True)
        mcp.append(
            R.UISliderFloat().Label("MCP all").Min(0.0).Max(_FLEX_MAX).Bind(a, "mcp_all"),
            R.UISliderFloat().Label("index MCP").Min(0.0).Max(_FLEX_MAX).Bind(a, "mcp_index"),
            R.UISliderFloat().Label("mid MCP").Min(0.0).Max(_FLEX_MAX).Bind(a, "mcp_mid"),
            R.UISliderFloat().Label("ring MCP").Min(0.0).Max(_FLEX_MAX).Bind(a, "mcp_ring"),
            R.UISliderFloat().Label("pinky MCP").Min(0.0).Max(_FLEX_MAX).Bind(a, "mcp_pinky"),
        )
        win.append(mcp)

        thumb = R.UISection().Label("Thumb (policy; 0 = THUMB_REST_CTRL)").Expanded(True)
        for name in C.THUMB_REST_CTRL:
            lo, hi = a.thumb_limit(name)
            thumb.append(
                R.UISliderFloat()
                .Label(name)
                .Min(float(lo))
                .Max(float(hi))
                .Bind(a, "rest_" + name)
            )
        win.append(thumb)

        rest_f = R.UISection().Label("PIP/DIP (IK, follows MCP)").Expanded(True)
        for name in C.FINGER_REST_CTRL:
            rest_f.append(
                R.UISliderFloat()
                .Label(name)
                .Min(0.0)
                .Max(_FLEX_MAX)
                .Bind(a, "rest_" + name)
            )
        win.append(rest_f)

        win.append(
            R.UIButton().Label("Reset to V2 training defaults").Callback(lambda _: a.reset()),
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
        self.yaw = 0.0
        self.tz = 0.0
        self.ty = 0.0
        self.mcp_all = 0.0
        self.mcp_index = self.mcp_mid = self.mcp_ring = self.mcp_pinky = 0.0
        self._last_mcp_all = 0.0
        self._links = {V3._bare_link(l.name): l for l in self.hand.get_links()}
        for name, val in {
            **C.THUMB_REST_CTRL,
            **C.FINGER_REST_CTRL,
            **C.PINNED_REST_CTRL,
            **C.FINGER_SWING_REST_CTRL,
        }.items():
            setattr(self, "rest_" + name, float(val))
        self._held_qpos = None
        self._last_ui_q = None
        self._active = set()
        self._active_hold = 0

    def thumb_limit(self, name: str) -> tuple[float, float]:
        j = self.joints.get(name)
        if j is None:
            return (0.0, _FLEX_MAX)
        lim = np.asarray(j.limit).reshape(-1)
        lo = float(lim[0]) if lim.size >= 1 and np.isfinite(lim[0]) else 0.0
        hi = float(lim[1]) if lim.size >= 2 and np.isfinite(lim[1]) else _FLEX_MAX
        rest = float(C.THUMB_REST_CTRL.get(name, 0.0))
        return C.range_containing_rest((lo, hi), rest)

    def _sync_mcp_all(self):
        if abs(self.mcp_all - self._last_mcp_all) > 1e-6:
            self.mcp_index = self.mcp_mid = self.mcp_ring = self.mcp_pinky = float(
                self.mcp_all
            )
            self._last_mcp_all = float(self.mcp_all)

    def _finger_mcp(self, finger: str) -> float:
        return float(getattr(self, f"mcp_{finger}"))

    def _rest_targets(self) -> dict:
        targets = {**C.rest_ctrl()}
        for name in {
            **C.THUMB_REST_CTRL,
            **C.PINNED_REST_CTRL,
            **C.FINGER_SWING_REST_CTRL,
        }:
            attr = "rest_" + name
            if hasattr(self, attr):
                targets[name] = float(getattr(self, attr))
        return targets

    def _set_root_pose(self):
        q = V3.compose_attach_quat(
            (self.rpy_r, self.rpy_p, self.rpy_y),
            self.z_spin,
            self.pitch_up,
            0.0,
            yaw_rad=-float(self.yaw),
            world_roll_yaw=True,
        )
        self.hand.set_root_pose(
            sapien.Pose(
                p=[
                    self.pos_x + float(self.tz),
                    self.pos_y,
                    self.pos_z + float(self.ty),
                ], q=q
            )
        )

    def _q_from_targets(self, targets: dict, zero_pip_dip: bool) -> np.ndarray:
        q = np.zeros(self.hand.dof, dtype=np.float32)
        for i, j in enumerate(self.hand.get_active_joints()):
            bare = V3._bare_joint(j.name)
            val = float(targets.get(bare, 0.0))
            if zero_pip_dip and bare in C.FOUR_FINGER_PIP_DIP_JOINTS:
                val = 0.0
            lim = np.asarray(j.limit).reshape(-1)
            if lim.size >= 2:
                lo, hi = float(lim[0]), float(lim[1])
            else:
                lo, hi = val, val
            if bare in C.THUMB_REST_CTRL:
                lo, hi = C.range_containing_rest((lo, hi), val)
            q[i] = float(np.clip(val, lo, hi))
        return q

    def _link_rot(self, bare: str) -> np.ndarray:
        link = self._links[bare]
        pose = link.get_pose() if hasattr(link, "get_pose") else link.pose
        return np.asarray(pose.to_transformation_matrix()[:3, :3], dtype=np.float64)

    def _ik_pip_dip(self, targets: dict) -> dict:
        mcp = {finger: self._finger_mcp(finger) for finger in _FINGERS}
        try:
            self.hand.set_qpos(self._q_from_targets(targets, zero_pip_dip=True))
            remaining = {}
            for finger in _FINGERS:
                rd = self._link_rot(f"{finger}_DIP_link")
                rm = self._link_rot(f"{finger}_MCP_link")
                remaining[finger] = C.signed_flex_to_align(rd[:, 2], rm[:, 1])
            ik = C.couple_four_finger_pip_dip(mcp, flex_remaining=remaining)
        except KeyError:
            ik = C.couple_four_finger_pip_dip(mcp)
        rest = C.rest_ctrl()
        idle = C.idle_four_finger_pip_dip(rest)
        for finger in _FINGERS:
            if abs(float(mcp.get(finger, 0.0))) > 1e-4:
                continue
            ik[f"{finger}_PIP_joint"] = idle[f"{finger}_PIP_joint"]
            ik[f"{finger}_DIP_joint"] = idle[f"{finger}_DIP_joint"]
        return ik

    def _ui_qpos(self) -> np.ndarray:
        self._sync_mcp_all()
        self._set_root_pose()
        targets = self._rest_targets()
        targets["index_MCP_joint"] = float(self.mcp_index)
        targets["mid_MCP_joint"] = float(self.mcp_mid)
        targets["ring_MCP_joint"] = float(self.mcp_ring)
        targets["pinky_MCP_joint"] = float(self.mcp_pinky)
        if getattr(C, "COUPLE_PIP_DIP_TO_MCP", False):
            ik = self._ik_pip_dip(targets)
            targets.update(ik)
            for name, val in ik.items():
                setattr(self, "rest_" + name, float(val))
        return self._q_from_targets(targets, zero_pip_dip=False)

    def apply(self):
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
            held = q_ui.copy()
        if getattr(C, "COUPLE_PIP_DIP_TO_MCP", False):
            for i, j in enumerate(self.hand.get_active_joints()):
                if V3._bare_joint(j.name) in C.FOUR_FINGER_PIP_DIP_JOINTS:
                    held[i] = q_ui[i]
        self._held_qpos = held
        self._last_ui_q = q_ui.copy()

        self._set_root_pose()
        for i, j in enumerate(self.hand.get_active_joints()):
            j.set_drive_target(float(held[i]))
        self.hand.set_qpos(held)
        self.hand.set_qvel(np.zeros_like(held))

    def hold_locked(self):
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
        pinned = {n: float(getattr(self, "rest_" + n)) for n in C.PINNED_REST_CTRL}

        def fmt_dict(d):
            lines = ["{"]
            for k, v in d.items():
                lines.append(f'    "{k}": {v:.4f},')
            lines.append("}")
            return "\n".join(lines)

        text = f"""
# --- Tsensei 达显 V2 LEFT (this preview) ---
LEFT_HAND_POSITION = ({left_pos[0]:.4f}, {left_pos[1]:.4f}, {left_pos[2]:.4f})
LEFT_FOREARM_Z_SPIN_DEG = {self.z_spin:.1f}
forearm_yaw (left ctrl, + = outwards) = {self.yaw:.4f} rad
forearm_tz (world +X / pos_x, 0 = attach) = {self.tz:.4f} m
forearm_ty (world +Z / pos_z, 0 = attach) = {self.ty:.4f} m

# --- mapped RIGHT (y-mirror; same rpy / pitch / rest) ---
# Paste into robopianist/models/hands/daxian_v2_hand_constants.py
HAND_BASE_RPY_DEG = ({rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f})
PALM_PITCH_UP_DEG = {self.pitch_up:.1f}
LEFT_HAND_POSITION = ({left_pos[0]:.4f}, {left_pos[1]:.4f}, {left_pos[2]:.4f})
RIGHT_HAND_POSITION = ({right_pos[0]:.4f}, {right_pos[1]:.4f}, {right_pos[2]:.4f})
LEFT_FOREARM_Z_SPIN_DEG = {self.z_spin:.1f}
RIGHT_FOREARM_Z_SPIN_DEG = {-self.z_spin:.1f}

PINNED_REST_CTRL = {fmt_dict(pinned)}
FINGER_REST_CTRL = {fmt_dict(finger)}
# Thumb joints are policy-driven; rest is episode init / CanonicalSpec 0:
THUMB_REST_CTRL = {fmt_dict(thumb)}

# MCP press in this session:
#   index={self.mcp_index:.3f}  mid={self.mcp_mid:.3f}  ring={self.mcp_ring:.3f}  pinky={self.mcp_pinky:.3f}
"""
        print(text)
        out = Path("/tmp/daxian_v2_hand_tune_mapped.py")
        out.write_text(text)
        print(f"wrote {out}")


def _require_x11() -> None:
    display = os.environ.get("DISPLAY", "")
    x11 = Path("/tmp/.X11-unix")
    sockets = sorted(p.name for p in x11.glob("X*")) if x11.is_dir() else []
    if display and sockets:
        return
    host = os.uname().nodename
    raise SystemExit(
        "SAPIEN viewer needs X11. "
        f"hostname={host} DISPLAY={display!r} sockets={sockets or 'none'}.\n"
        "This is the training container (daxianpianist-dev-1): no /tmp/.X11-unix.\n"
        "Do not recreate it to add the mount (that stops training).\n"
        "\n"
        "1. Connect remote desktop: xrdp port 3391 (XFCE is DISPLAY=:10).\n"
        "2. Exit this exec, then on the Linux host:\n"
        "     bash docker/run_sapien_v2.sh\n"
        "The window opens on the XFCE desktop, not in Cursor."
    )


def main():
    _require_x11()
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
    V3.build_piano(scene)

    tmp = Path(tempfile.mkdtemp(prefix="daxian_v2_sapien_")) / "left.urdf"
    rewrite_left_urdf(tmp)
    print(f"loading 达显 V2 hand from {tmp}")
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = False
    hand = loader.load(str(tmp))
    if hand is None:
        raise SystemExit("URDF load failed")
    print("hand joints:")
    joints = setup_hand(scene, hand)

    app = App(hand, joints)
    app.apply()

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
        "\n达显 V2 reduced space: policy = 四指 MCP/swing + 拇指全部关节 "
        "+ forearm_tx/ty/tz/yaw.\n"
        "PIP/DIP are IK-solved so the fingertip points vertically onto the keys.\n"
        "rpy roll = world X (pronation); rpy yaw = world Z (key-plane turn).\n"
    )

    while not viewer.closed:
        app.apply()
        if not viewer.paused:
            scene.step()
            app.hold_locked()
        scene.update_render()
        viewer.render()


if __name__ == "__main__":
    os.chdir(REPO)
    main()
