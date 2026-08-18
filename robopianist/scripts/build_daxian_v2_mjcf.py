#!/usr/bin/env python3
"""Compile daxian_V2 URDFs into isolated RoboPianist MJCF (does not touch V3).

Writes:
  robopianist/models/hands/third_party/daxian_v2/{left,right}_hand.xml
  robopianist/models/hands/third_party/daxian_v2/assets/*.obj
  robopianist/models/hands/third_party/daxian_v2/_raw_{left,right}.xml
"""

from __future__ import annotations

import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[2]
URDF_DIR = WORKSPACE / "daxian_V2" / "urdf"
MESH_DIR = WORKSPACE / "daxian_V2" / "meshes"
OUT_DIR = PACKAGE_ROOT / "robopianist" / "models" / "hands" / "third_party" / "daxian_v2"
ASSETS = OUT_DIR / "assets"

HANDS = (
    ("left", "lh_", URDF_DIR / "daxian__hand_left_v1.urdf"),
    ("right", "rh_", URDF_DIR / "daxian__hand_right_v1.urdf"),
)

FINGER_JOINTS = (
    "thumb_rota_joint",
    "thumb_swing_joint",
    "thumb_MCP_joint",
    "thumb_PIP_joint",
    "index_swing_joint",
    "index_MCP_joint",
    "index_PIP_joint",
    "index_DIP_joint",
    "mid_swing_joint",
    "mid_MCP_joint",
    "mid_PIP_joint",
    "mid_DIP_joint",
    "ring_swing_joint",
    "ring_MCP_joint",
    "ring_PIP_joint",
    "ring_DIP_joint",
    "pinky_rota_joint",
    "pinky_swing_joint",
    "pinky_MCP_joint",
    "pinky_PIP_joint",
    "pinky_DIP_joint",
)

TIP_BODIES = (
    "thumb_PIP_link",
    "index_DIP_link",
    "mid_DIP_link",
    "ring_DIP_link",
    "pinky_DIP_link",
)

STRIP_PREFIXES = (
    "left_hand_",
    "Left_hand_",
    "right_hand_",
    "Right_hand_",
)


def _strip_side(name: str) -> str:
    for prefix in STRIP_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    if name == "base_link":
        return "forearm"
    return name


# MuJoCo rejects meshes with >200k faces. V2 palms are ~224k; V3 palms are ~80k.
_MAX_FACES = 180_000
_TARGET_FACES = 80_000


def _stl_to_obj(src: Path, dst: Path) -> None:
    import trimesh

    mesh = trimesh.load(str(src), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    n_faces = len(mesh.faces)
    if n_faces > _MAX_FACES:
        mesh = mesh.simplify_quadric_decimation(1.0 - _TARGET_FACES / n_faces)
        print(f"  decimate {src.name}: {n_faces} -> {len(mesh.faces)} faces")
    dst.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(dst))


def _convert_meshes() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for src in sorted(MESH_DIR.iterdir()):
        if src.suffix.lower() != ".stl":
            continue
        dst = ASSETS / (src.stem + ".obj")
        if dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        print(f"convert {src.name}")
        _stl_to_obj(src, dst)


def _urdf_with_local_meshes(urdf: Path) -> Path:
    """Copy the URDF and converted OBJs into one temp dir for MuJoCo."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="daxian_v2_urdf_"))
    text = (
        urdf.read_text()
        .replace("../meshes/", "")
        .replace(".STL", ".obj")
        .replace(".stl", ".obj")
    )
    dst = tmp_dir / urdf.name
    dst.write_text(text)
    for mesh in ASSETS.glob("*.obj"):
        shutil.copy2(mesh, tmp_dir / mesh.name)
    return dst


def _compile_urdf(urdf: Path) -> Path:
    if not urdf.is_file():
        raise FileNotFoundError(urdf)
    patched = _urdf_with_local_meshes(urdf)
    model = mujoco.MjModel.from_xml_path(str(patched))
    tmp = Path(tempfile.mkdtemp(prefix="daxian_v2_mjcf_")) / "raw.xml"
    mujoco.mj_saveLastXML(str(tmp), model)
    return tmp


def _rename_attr(elem: ET.Element, attr: str, prefix: str) -> None:
    val = elem.get(attr)
    if not val:
        return
    stripped = _strip_side(val)
    if stripped == val and val not in ("base_link", "forearm"):
        # mesh names often equal the original link/file stem
        stripped = _strip_side(val)
    elem.set(attr, prefix + stripped if attr != "file" else val)


def _postprocess(raw_xml: Path, prefix: str, side: str) -> tuple[str, dict[str, tuple[float, float]]]:
    tree = ET.parse(raw_xml)
    root = tree.getroot()
    root.set("model", f"{prefix}daxian_v2_hand")

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("angle", "radian")
    compiler.set("meshdir", "assets")
    compiler.set("autolimits", "true")
    compiler.attrib.pop("boundmass", None)
    compiler.attrib.pop("boundinertia", None)

    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.set("impratio", "10")
    option.set("timestep", "0.005")

    default = root.find("default")
    if default is None:
        default = ET.Element("default")
        # After compiler + option
        root.insert(2, default)
        ET.SubElement(default, "joint", damping="0.05", armature="0.001")
        ET.SubElement(
            default,
            "geom",
            friction="1 0.005 0.001",
            condim="3",
            solimp="0.99 0.99 0.01",
            solref="0.01 1",
        )
        ET.SubElement(default, "position", kp="20.0")

    # Prefix mesh assets; keep file names as copied from CAD.
    joint_ranges: dict[str, tuple[float, float]] = {}
    for mesh in root.findall("./asset/mesh"):
        name = mesh.get("name") or ""
        mesh.set("name", prefix + "mesh_" + _strip_side(name))
        fname = mesh.get("file") or ""
        mesh.set("file", Path(fname).name)

    for geom in root.iter("geom"):
        mesh = geom.get("mesh")
        if mesh:
            geom.set("mesh", prefix + "mesh_" + _strip_side(mesh))

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("compiled MJCF has no worldbody")

    for body in worldbody.iter("body"):
        name = body.get("name") or ""
        body.set("name", prefix + _strip_side(name))

    for joint in worldbody.iter("joint"):
        name = joint.get("name") or ""
        if not name:
            continue
        key = _strip_side(name)
        joint.set("name", prefix + key)
        rng = joint.get("range")
        if rng:
            lo, hi = (float(x) for x in rng.split())
            joint_ranges[key] = (lo, hi)
        joint.set("damping", "0.05")
        joint.set("armature", "0.001")

    for site in list(worldbody.iter("site")):
        site.set("name", prefix + _strip_side(site.get("name") or "site"))

    # Drop MuJoCo's freejoint on the URDF root if present.
    for body in list(worldbody):
        if body.tag != "body":
            continue
        for child in list(body):
            if child.tag == "freejoint" or (
                child.tag == "joint" and child.get("type") == "free"
            ):
                body.remove(child)

    # URDF compile dumps root link geoms onto worldbody. Wrap them in a
    # forearm body so DaxianHand can attach the same way as V3.
    forearm = ET.Element("body", name=f"{prefix}forearm")
    ET.SubElement(
        forearm,
        "inertial",
        pos="0 0 0",
        mass="0.5",
        diaginertia="0.001 0.001 0.001",
    )
    for child in list(worldbody):
        worldbody.remove(child)
        forearm.append(child)
    worldbody.append(forearm)

    for geom in list(forearm.findall("geom")):
        mesh = geom.get("mesh") or ""
        if "palm" not in mesh.lower():
            continue
        pos = geom.get("pos", "0 0 0")
        geom.attrib.pop("pos", None)
        palm = ET.Element("body", name=f"{prefix}palm_link", pos=pos)
        palm.append(geom)
        forearm.remove(geom)
        # Keep palm next to the forearm geoms, before finger bodies.
        insert_at = 0
        for i, child in enumerate(list(forearm)):
            if child.tag == "body":
                insert_at = i
                break
            insert_at = i + 1
        forearm.insert(insert_at, palm)

    def _split_visual_collision(parent: ET.Element) -> None:
        for geom in list(parent.findall("geom")):
            if not geom.get("mesh"):
                continue
            vis = ET.Element("geom", geom.attrib)
            vis.set("contype", "0")
            vis.set("conaffinity", "0")
            vis.set("group", "1")
            vis.set("density", "0")
            parent.insert(list(parent).index(geom), vis)

    _split_visual_collision(forearm)
    for body in forearm.findall(".//body"):
        _split_visual_collision(body)

    actuator = root.find("actuator")
    if actuator is not None:
        root.remove(actuator)
    actuator = ET.SubElement(root, "actuator")
    for key in FINGER_JOINTS:
        lo, hi = joint_ranges.get(key, (0.0, 1.57))
        ET.SubElement(
            actuator,
            "position",
            name=f"{prefix}A_{key}",
            joint=f"{prefix}{key}",
            ctrlrange=f"{lo} {hi}",
            kp="20.0",
        )

    # Placeholder distal sites; DaxianHand replaces them from constants.
    for body in root.iter("body"):
        short = (body.get("name") or "")[len(prefix) :]
        if short not in TIP_BODIES:
            continue
        for site in list(body.findall("site")):
            body.remove(site)
        ET.SubElement(
            body,
            "site",
            name=f"{prefix}{short.replace('_link', '_site')}",
            pos="0 0 0.015",
            type="sphere",
            size="0.004",
            group="3",
            rgba="0.9 0.1 0.1 0.8",
        )

    ET.indent(root, space="  ")
    out = OUT_DIR / f"{side}_hand.xml"
    tree.write(out, encoding="unicode", xml_declaration=True)
    return str(out), joint_ranges


def _tip_offsets(xml_path: Path, prefix: str) -> dict[str, tuple[float, float, float]]:
    """AABB centre of each tip collision mesh, in the tip body frame."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    out: dict[str, tuple[float, float, float]] = {}
    for tip in TIP_BODIES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + tip)
        if body_id < 0:
            continue
        pts: list[np.ndarray] = []
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) != body_id:
                continue
            if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            if int(model.geom_contype[geom_id]) == 0:
                continue
            mesh_id = int(model.geom_dataid[geom_id])
            vertadr = int(model.mesh_vertadr[mesh_id])
            vertnum = int(model.mesh_vertnum[mesh_id])
            verts = model.mesh_vert[vertadr : vertadr + vertnum]
            gpos = model.geom_pos[geom_id]
            gquat = model.geom_quat[geom_id]
            mat = np.zeros(9, dtype=np.float64)
            mujoco.mju_quat2Mat(mat, gquat)
            rot = mat.reshape(3, 3)
            pts.append(verts @ rot.T + gpos)
        if not pts:
            out[tip] = (0.0, 0.0, 0.012)
            continue
        cloud = np.concatenate(pts, axis=0)
        lo = cloud.min(axis=0)
        hi = cloud.max(axis=0)
        center = 0.5 * (lo + hi)
        out[tip] = tuple(float(x) for x in center)
    return out


def main() -> None:
    if not URDF_DIR.is_dir():
        raise FileNotFoundError(f"V2 URDF dir missing: {URDF_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _convert_meshes()

    print("FINGERTIP_COLLISION_POS (AABB centres; paste into daxian_v2_hand_constants.py)")
    print("{")
    for side, prefix, urdf in HANDS:
        raw = _compile_urdf(urdf)
        shutil.copy2(raw, OUT_DIR / f"_raw_{side}.xml")
        xml_path, ranges = _postprocess(raw, prefix, side)
        print(f"  # {side} joint ranges: {ranges}")
        offsets = _tip_offsets(Path(xml_path), prefix)
        if side == "right":
            for tip, pos in offsets.items():
                print(f'    "{tip}": ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}),')
        print(f"wrote {xml_path}")
    print("}")


if __name__ == "__main__":
    main()
