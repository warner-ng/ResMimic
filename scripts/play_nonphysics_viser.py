#!/usr/bin/env python3
"""Non-physics web renderer for retargeted robot + object motions.

This script intentionally follows OmniRetarget's playback style in
`holosoma_retargeting/src/viser_utils.py`:
- qpos layout with robot + object in one frame vector
- SLERP interpolation for quaternions
- slider + play/pause + FPS + interpolation multiplier

No physics stepping is performed; this is render-only playback.
"""

from __future__ import annotations

import argparse
import pickle
import tempfile
import threading
import time
import traceback
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import viser
    from viser.extras import ViserUrdf
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "viser is required for this script. Install it first (e.g. pip install viser).\n"
        f"Import error: {type(e).__name__}: {e}"
    )
try:
    import trimesh
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "trimesh is required for runtime pair leveling mesh support.\n"
        f"Install: pip install trimesh\nImport error: {type(e).__name__}: {e}"
    )


def _log_urdf_mesh_inventory(urdf_path: Path) -> None:
    """Print mesh inventory and per-mesh loading status for diagnosis."""
    print(f"[URDF-DIAG] inspecting meshes: {urdf_path}")
    try:
        root = ET.parse(str(urdf_path)).getroot()
    except Exception as e:
        print(f"[URDF-DIAG][ERROR] cannot parse URDF XML: {type(e).__name__}: {e}")
        return

    mesh_nodes = root.findall(".//mesh")
    if not mesh_nodes:
        print("[URDF-DIAG] no <mesh> tags found (only primitive geometries).")
        return

    base_dir = urdf_path.parent
    for i, mesh_node in enumerate(mesh_nodes, start=1):
        fn = mesh_node.get("filename", "")
        if not fn:
            print(f"[URDF-DIAG] #{i:02d} empty mesh filename")
            continue
        if fn.startswith("package://"):
            print(f"[URDF-DIAG][WARN] package URI not auto-resolved by URDF parser: {fn}")
        if fn.startswith("file://"):
            print(f"[URDF-DIAG][WARN] file URI detected: {fn}")
        if fn.startswith("file://"):
            from urllib.parse import urlparse

            parsed = urlparse(fn)
            mesh_path = Path(parsed.path)
        else:
            mesh_path = Path(fn)
        if not mesh_path.is_absolute():
            mesh_path = (base_dir / mesh_path).resolve()
        exists = mesh_path.exists()
        scale = mesh_node.get("scale", "<none>")
        print(f"[URDF-DIAG] #{i:02d} filename={fn} resolved={mesh_path} scale={scale} exists={exists}")
        if not exists:
            print(f"[URDF-DIAG][ERROR] missing mesh file: {mesh_path}")
            continue
        try:
            geom = trimesh.load_mesh(str(mesh_path), force="mesh", process=False)
            if geom.is_empty:
                print(f"[URDF-DIAG][WARN] {mesh_path.name} loaded as empty mesh")
            else:
                bmin = np.asarray(geom.bounds[0]).round(4).tolist()
                bmax = np.asarray(geom.bounds[1]).round(4).tolist()
                print(
                    f"[URDF-DIAG] {mesh_path.name}: verts={len(geom.vertices)} faces={len(geom.faces)} "
                    f"bbox_min={bmin} bbox_max={bmax}"
                )
        except Exception as e:  # pragma: no cover
            print(f"[URDF-DIAG][ERROR] failed to load mesh {mesh_path}: {type(e).__name__}: {e}")


def _summarize_urdf_graph(urdf_path: Path) -> None:
    """Print URDF link/joint counts and visual/collision geometry breakdown."""
    try:
        root = ET.parse(str(urdf_path)).getroot()
    except Exception as e:
        print(f"[URDF-DIAG][ERROR] cannot parse URDF XML for summary: {type(e).__name__}: {e}")
        return

    links = root.findall("link")
    joints = root.findall("joint")
    mesh_tags = root.findall(".//visual/geometry/mesh") + root.findall(".//collision/geometry/mesh")
    primitive_tags = (
        root.findall(".//visual/geometry/*")
        + root.findall(".//collision/geometry/*")
    )
    primitive_count = len([node for node in primitive_tags if node.tag.split("}")[-1] != "mesh"])
    unique_filenames = []
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        if filename and filename not in unique_filenames:
            unique_filenames.append(filename)
    print(f"[URDF-SUM] links={len(links)} joints={len(joints)}")
    print(f"[URDF-SUM] unique mesh refs={len(unique_filenames)} total mesh tags={len(mesh_tags)} primitive geometry tags={primitive_count}")


def _log_urdf_object_snapshot(urdf_obj: object, tag: str) -> None:
    n_meshes = len(getattr(urdf_obj, "_meshes", []))
    n_joints = len(getattr(urdf_obj, "_joint_frames", []))
    print(f"[URDF-SUM] {tag}: meshes={n_meshes} joints={n_joints}")
    if n_meshes == 0:
        print(f"[URDF-WARN] {tag}: no mesh geometry reported by ViserUrdf, this usually means visual geometry failed to load.")

    urdf = getattr(urdf_obj, "_urdf", None)
    if urdf is None:
        print(f"[URDF-SUM] {tag}: _urdf object not available.")
        return
    try:
        scene = getattr(urdf, "scene", None)
        cscene = getattr(urdf, "collision_scene", None)
        print(
            f"[URDF-SUM] {tag}: yourdfpy scene visual={scene is not None} "
            f"collision={cscene is not None} "
            f"visual_nodes={len(scene.geometry) if scene is not None and getattr(scene, 'geometry', None) is not None else 0} "
            f"collision_nodes={len(cscene.geometry) if cscene is not None and getattr(cscene, 'geometry', None) is not None else 0}"
        )
        errors = getattr(urdf, "errors", None)
        if errors:
            for err in errors:
                print(f"[URDF-ERR] {tag}: {err}")
    except Exception as e:  # pragma: no cover
        print(f"[URDF-SUM][ERROR] reading yourdfpy scene info failed: {type(e).__name__}: {e}")


def xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    q_xyzw = np.asarray(q_xyzw, dtype=np.float64)
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    q_wxyz = np.asarray(q_wxyz, dtype=np.float64)
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)


def axis_angle_to_wxyz(axis: str, degrees: float) -> np.ndarray:
    axis_vec = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}[axis]
    q_xyzw = R.from_rotvec(np.deg2rad(degrees) * np.asarray(axis_vec, dtype=np.float64)).as_quat()
    return xyzw_to_wxyz(q_xyzw)


def rpy_deg_to_wxyz(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    q_xyzw = R.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_quat()
    return xyzw_to_wxyz(q_xyzw)


def _build_urdf(server: viser.ViserServer, urdf_path: str, root_node_name: str):
    """Handle minor API differences across viser versions."""
    urdf_path_obj = Path(urdf_path)
    if not urdf_path_obj.exists():
        raise RuntimeError(f"URDF file does not exist: {urdf_path_obj}")
    _log_urdf_mesh_inventory(urdf_path_obj)
    _summarize_urdf_graph(urdf_path_obj)
    try:
        # Newer viser expects Path (or yourdfpy.URDF), not plain str.
        print(f"[URDF] load: {urdf_path}")
        print(f"[URDF] root_node_name: {root_node_name}")
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            urdf_obj = ViserUrdf(server, urdf_path_obj, root_node_name=root_node_name)
            for w in caught_warnings:
                print(f"[URDF-WARN] {w.category.__name__}: {w.message}")
        _log_urdf_object_snapshot(urdf_obj, "object")
        print(f"[URDF] loaded. meshes={len(urdf_obj._meshes)} joints={len(urdf_obj._joint_frames)}")
        return urdf_obj
    except TypeError:
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                urdf_obj = ViserUrdf(server, urdf_path_obj, root_node_name)
                for w in caught_warnings:
                    print(f"[URDF-WARN] {w.category.__name__}: {w.message}")
            _log_urdf_object_snapshot(urdf_obj, "object-alt")
            print(f"[URDF] loaded (alt signature). meshes={len(urdf_obj._meshes)} joints={len(urdf_obj._joint_frames)}")
            return urdf_obj
        except TypeError:
            try:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")
                    urdf_obj = ViserUrdf(server, urdf_path_obj)
                    for w in caught_warnings:
                        print(f"[URDF-WARN] {w.category.__name__}: {w.message}")
                _log_urdf_object_snapshot(urdf_obj, "object-no-root")
                print(f"[URDF] loaded (no root_node_name). meshes={len(urdf_obj._meshes)} joints={len(urdf_obj._joint_frames)}")
                return urdf_obj
            except AssertionError:
                # Older/newer API mismatch: load explicit URDF object and pass it in.
                import yourdfpy

                urdf = yourdfpy.URDF.load(urdf_path_obj)
                try:
                    with warnings.catch_warnings(record=True) as caught_warnings:
                            warnings.simplefilter("always")
                            urdf_obj = ViserUrdf(server, urdf, root_node_name=root_node_name)
                            for w in caught_warnings:
                                print(f"[URDF-WARN] {w.category.__name__}: {w.message}")
                    _log_urdf_object_snapshot(urdf_obj, "object-via-urdf")
                    print(f"[URDF] loaded via URDF obj. meshes={len(urdf_obj._meshes)} joints={len(urdf_obj._joint_frames)}")
                    return urdf_obj
                except TypeError:
                    try:
                        with warnings.catch_warnings(record=True) as caught_warnings:
                            warnings.simplefilter("always")
                            urdf_obj = ViserUrdf(server, urdf, root_node_name)
                            for w in caught_warnings:
                                print(f"[URDF-WARN] {w.category.__name__}: {w.message}")
                        _log_urdf_object_snapshot(urdf_obj, "object-via-urdf-pos")
                        print(
                            f"[URDF] loaded via URDF obj positional. meshes={len(urdf_obj._meshes)} joints={len(urdf_obj._joint_frames)}"
                        )
                        return urdf_obj
                    except TypeError:
                        with warnings.catch_warnings(record=True) as caught_warnings:
                            warnings.simplefilter("always")
                            urdf_obj = ViserUrdf(server, urdf)
                            for w in caught_warnings:
                                print(f"[URDF-WARN] {w.category.__name__}: {w.message}")
                        _log_urdf_object_snapshot(urdf_obj, "object-via-urdf-default")
                        print(f"[URDF] loaded via URDF obj default root. meshes={len(urdf_obj._meshes)} joints={len(urdf_obj._joint_frames)}")
                        return urdf_obj
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to load URDF via ViserUrdf: {urdf_path_obj} ({type(e).__name__}: {e})") from e


def build_scaled_urdf_copy(urdf_path: str, scale: float) -> str:
    src = Path(urdf_path).resolve()
    src_dir = src.parent
    tree = ET.parse(src)
    root = tree.getroot()

    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename:
            p = Path(filename)
            if not p.is_absolute():
                mesh.set("filename", str((src_dir / p).resolve()))

        s = mesh.get("scale")
        if s:
            vals = [float(x) for x in s.split()]
            if len(vals) == 1:
                vals = [vals[0], vals[0], vals[0]]
            elif len(vals) == 2:
                vals = [vals[0], vals[1], vals[1]]
        else:
            vals = [1.0, 1.0, 1.0]
        vals = [v * float(scale) for v in vals[:3]]
        mesh.set("scale", f"{vals[0]:.8g} {vals[1]:.8g} {vals[2]:.8g}")

    tf = tempfile.NamedTemporaryFile(prefix="object_scaled_", suffix=".urdf", delete=False)
    tree.write(tf.name, encoding="utf-8", xml_declaration=True)
    return tf.name


def load_human_motion(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)

    required = ["root_pos", "root_rot", "dof_pos"]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in human motion: {path}")

    fps = int(data.get("fps", 30))
    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    root_rot = np.asarray(data["root_rot"], dtype=np.float64)  # xyzw
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)
    return fps, root_pos, root_rot, dof_pos


def load_object_motion(path: str):
    data = np.load(path)
    if "trans" not in data or "rot" not in data:
        raise KeyError(f"Object npz must contain trans/rot: {path}")
    trans = np.asarray(data["trans"], dtype=np.float64)
    rot = np.asarray(data["rot"], dtype=np.float64)  # xyzw
    return trans, rot


def quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def root_rot_offset_quat_xyzw(offset_deg: Tuple[float, float, float]) -> np.ndarray:
    roll_deg, pitch_deg, yaw_deg = offset_deg
    qx = R.from_rotvec(np.deg2rad(roll_deg) * np.array([1.0, 0.0, 0.0])).as_quat()
    qy = R.from_rotvec(np.deg2rad(pitch_deg) * np.array([0.0, 1.0, 0.0])).as_quat()
    qz = R.from_rotvec(np.deg2rad(yaw_deg) * np.array([0.0, 0.0, 1.0])).as_quat()
    return quat_mul_xyzw(qx, quat_mul_xyzw(qy, qz))


def apply_root_rot_offset_xyzw(root_rot_xyzw: np.ndarray, offset_deg: Tuple[float, float, float]) -> np.ndarray:
    if np.allclose(np.asarray(offset_deg, dtype=np.float64), 0.0):
        return root_rot_xyzw.copy()
    offset_q = root_rot_offset_quat_xyzw(offset_deg)
    return np.stack([quat_mul_xyzw(offset_q, q) for q in root_rot_xyzw], axis=0)


def apply_root_local_rot_offset_xyzw(root_rot_xyzw: np.ndarray, offset_deg: Tuple[float, float, float]) -> np.ndarray:
    if np.allclose(np.asarray(offset_deg, dtype=np.float64), 0.0):
        return root_rot_xyzw.copy()
    offset_q = root_rot_offset_quat_xyzw(offset_deg)
    return np.stack([quat_mul_xyzw(q, offset_q) for q in root_rot_xyzw], axis=0)


def compute_pair_level_transform(
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    local_body_pos: np.ndarray,
    foot_ids: List[int],
    object_pos: np.ndarray,
    object_rot_xyzw: np.ndarray,
    object_points: np.ndarray,
    target_z: float,
) -> Tuple[np.ndarray, np.ndarray]:
    world_feet = R.from_quat(root_rot_xyzw[0]).apply(local_body_pos[0, foot_ids]) + root_pos[0]
    human_support = world_feet[np.argmin(world_feet[:, 2])]
    world_obj = R.from_quat(object_rot_xyzw[0]).apply(object_points) + object_pos[0]
    object_support = world_obj[np.argmin(world_obj[:, 2])]

    d = human_support - object_support
    h = d.copy()
    h[2] = 0.0
    if np.linalg.norm(h) > 1e-8 and np.linalg.norm(d) > 1e-8 and abs(d[2]) > 1e-8:
        axis = np.cross(d, h)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-8:
            angle = np.arccos(np.clip(np.dot(d, h) / (np.linalg.norm(d) * np.linalg.norm(h)), -1.0, 1.0))
            level_rot = R.from_rotvec(axis / axis_norm * angle)
        else:
            level_rot = R.identity()
    else:
        level_rot = R.identity()

    midpoint = 0.5 * (human_support + object_support)
    trans = midpoint - level_rot.apply(midpoint)
    human_support_after = level_rot.apply(human_support) + trans
    object_support_after = level_rot.apply(object_support) + trans
    trans[2] += target_z - 0.5 * (human_support_after[2] + object_support_after[2])
    return level_rot.as_quat(), trans


def apply_pair_level_transform(
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    object_pos: np.ndarray,
    object_rot_xyzw: np.ndarray,
    level_rot_xyzw: np.ndarray,
    level_trans: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    level_rot = R.from_quat(level_rot_xyzw)
    root_pos_out = level_rot.apply(root_pos) + level_trans[None, :]
    object_pos_out = level_rot.apply(object_pos) + level_trans[None, :]
    root_rot_out = np.stack([quat_mul_xyzw(level_rot_xyzw, q) for q in root_rot_xyzw], axis=0)
    object_rot_out = np.stack([quat_mul_xyzw(level_rot_xyzw, q) for q in object_rot_xyzw], axis=0)
    return root_pos_out, root_rot_out, object_pos_out, object_rot_out


def build_qpos_sequence(
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    object_pos: np.ndarray,
    object_rot_xyzw: np.ndarray,
) -> np.ndarray:
    """Build MuJoCo-style qpos frames used by OmniRetarget viewer utility.

    Layout per frame:
      [0:3]   robot base pos (xyz)
      [3:7]   robot base quat (wxyz)
      [7:7+R] robot joints
      [-7:-4] object pos (xyz)
      [-4:]   object quat (wxyz)
    """
    n = int(min(len(root_pos), len(root_rot_xyzw), len(dof_pos), len(object_pos), len(object_rot_xyzw)))
    if n <= 0:
        raise ValueError("No valid frames found in inputs.")

    robot_dof = int(dof_pos.shape[1])
    qpos = np.zeros((n, 7 + robot_dof + 7), dtype=np.float64)
    qpos[:, 0:3] = root_pos[:n]
    qpos[:, 3:7] = np.stack([xyzw_to_wxyz(q) for q in root_rot_xyzw[:n]], axis=0)
    qpos[:, 7 : 7 + robot_dof] = dof_pos[:n]
    qpos[:, -7:-4] = object_pos[:n]
    qpos[:, -4:] = np.stack([xyzw_to_wxyz(q) for q in object_rot_xyzw[:n]], axis=0)
    return qpos


def create_motion_control_sliders(
    server: viser.ViserServer,
    viser_robot: ViserUrdf,
    robot_base_frame,
    motion_sequence: np.ndarray,
    *,
    robot_dof: int,
    viser_object: ViserUrdf | None = None,
    object_base_frame=None,
    contains_object_in_qpos: bool = True,
    initial_fps: int = 30,
    initial_interp_mult: int = 2,
    loop: bool = True,
) -> Tuple[List[object], List[float]]:
    """Adapted from OmniRetarget's viser_utils.py playback utility."""
    qpos = motion_sequence
    n_frames = int(qpos.shape[0])
    if n_frames == 0:
        raise ValueError("motion_sequence is empty.")

    has_object_input = (
        viser_object is not None
        and object_base_frame is not None
        and contains_object_in_qpos
        and qpos.shape[1] >= (7 + robot_dof + 7)
    )

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider("Frame", min=0, max=max(0, n_frames - 1), step=1, initial_value=0)
        play_btn = server.gui.add_button("Play / Pause")
        fps_in = server.gui.add_number("FPS", initial_value=int(initial_fps), min=1, max=240, step=1)
    with server.gui.add_folder("Smoothing"):
        interp_mult_in = server.gui.add_number(
            "Visual FPS multiplier", initial_value=int(initial_interp_mult), min=1, max=8, step=1
        )

    def _quat_normalize(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, float)
        n = float(np.linalg.norm(q))
        return q if n == 0.0 else q / n

    def _quat_continuous(prev_q: np.ndarray | None, curr_q: np.ndarray) -> np.ndarray:
        q = _quat_normalize(curr_q)
        if prev_q is None:
            return q
        return -q if float(np.dot(prev_q, q)) < 0.0 else q

    def _slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
        q0 = _quat_normalize(q0)
        q1 = _quat_normalize(q1)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            q = q0 + u * (q1 - q0)
            return _quat_normalize(q)
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        s = np.sin(theta)
        return (np.sin((1.0 - u) * theta) * q0 + np.sin(u * theta) * q1) / s

    def _interp_frame(qpos_arr: np.ndarray, i0: int, i1: int, u: float) -> np.ndarray:
        q0 = qpos_arr[i0]
        q1 = qpos_arr[i1]
        out = q0.copy()

        out[0:3] = (1.0 - u) * q0[0:3] + u * q1[0:3]
        out[3:7] = _slerp(q0[3:7], q1[3:7], u)

        j0 = q0[7 : 7 + robot_dof]
        j1 = q1[7 : 7 + robot_dof]
        out[7 : 7 + robot_dof] = (1.0 - u) * j0 + u * j1

        if has_object_input:
            out[-7:-4] = (1.0 - u) * q0[-7:-4] + u * q1[-7:-4]
            out[-4:] = _slerp(q0[-4:], q1[-4:], u)
        return out

    playing = {"flag": False}
    tick = {"next": time.perf_counter()}
    prev = {"robot_q": None, "obj_q": None}
    nonlocal_f = {"f": float(frame_slider.value)}
    updating_programmatically = {"flag": False}

    def _apply_frame_from_q(q: np.ndarray) -> None:
        joints = q[7 : 7 + robot_dof]
        if joints.shape[0] != robot_dof:
            joints = joints[:robot_dof] if joints.shape[0] > robot_dof else np.pad(joints, (0, robot_dof - joints.shape[0]))
        viser_robot.update_cfg(joints)

        robot_base_frame.position = q[0:3]
        r_q = _quat_continuous(prev["robot_q"], q[3:7])
        prev["robot_q"] = r_q
        robot_base_frame.wxyz = r_q

        if has_object_input and object_base_frame is not None:
            object_base_frame.position = q[-7:-4]
            o_q = _quat_continuous(prev["obj_q"], q[-4:])
            prev["obj_q"] = o_q
            object_base_frame.wxyz = o_q

    def _apply_discrete_frame(i: int) -> None:
        i = int(np.clip(i, 0, n_frames - 1))
        _apply_frame_from_q(qpos[i])

    @play_btn.on_click
    def _(_evt) -> None:
        playing["flag"] = not playing["flag"]
        tick["next"] = time.perf_counter()
        prev["robot_q"] = None
        prev["obj_q"] = None
        nonlocal_f["f"] = float(frame_slider.value)

    @fps_in.on_update
    def _(_evt) -> None:
        tick["next"] = time.perf_counter()

    @interp_mult_in.on_update
    def _(_evt) -> None:
        tick["next"] = time.perf_counter()

    @frame_slider.on_update
    def _(_evt) -> None:
        if not updating_programmatically["flag"]:
            playing["flag"] = False
            tick["next"] = time.perf_counter()
            frame_val = int(frame_slider.value)
            _apply_discrete_frame(frame_val)
            prev["robot_q"] = None
            prev["obj_q"] = None
            nonlocal_f["f"] = float(frame_val)

    def _player_loop() -> None:
        if n_frames <= 1:
            return
        while True:
            if playing["flag"]:
                now = time.perf_counter()
                fps_val = max(1, int(fps_in.value))
                mult = max(1, int(interp_mult_in.value))
                dt = 1.0 / (fps_val * mult)

                if now >= tick["next"]:
                    f = nonlocal_f["f"] + 1.0 / mult
                    if loop:
                        f = f % max(1, n_frames)
                    else:
                        f = min(f, float(n_frames - 1))
                    nonlocal_f["f"] = f

                    k0 = int(np.floor(f))
                    k1 = (k0 + 1) % max(1, n_frames) if loop else min(k0 + 1, n_frames - 1)
                    u = float(f - k0)

                    q_interp = _interp_frame(qpos, k0, k1, u)
                    _apply_frame_from_q(q_interp)

                    updating_programmatically["flag"] = True
                    frame_slider.value = k0
                    updating_programmatically["flag"] = False

                    tick["next"] = now + dt
                else:
                    time.sleep(min(0.002, max(0.0, tick["next"] - now)))
            else:
                time.sleep(0.02)

    threading.Thread(target=_player_loop, daemon=True).start()
    _apply_discrete_frame(0)
    return [frame_slider], [0.0]


def main():
    parser = argparse.ArgumentParser(description="Non-physics web playback for robot+object trajectories")
    parser.add_argument(
        "--human",
        default="/home/warner/_projects/ResMimic/legged_gym/assets/motions/Date03_Sub03_chairblack_lift_it1_input_human_upright_suitcasez.pkl",
        help="Path to human motion .pkl",
    )
    parser.add_argument(
        "--object",
        default="/home/warner/_projects/ResMimic/legged_gym/assets/motions/Date03_Sub03_chairblack_lift_it1_input_object_upright_suitcasez.npz",
        help="Path to object motion .npz",
    )
    parser.add_argument(
        "--robot-urdf",
        default="/home/warner/_projects/ResMimic/assets/g1/g1_custom_collision_29dof.urdf",
        help="Robot URDF path",
    )
    parser.add_argument(
        "--object-urdf",
        default="/home/warner/_projects/ResMimic/assets/bicycle_top_tube/bikered.urdf",
        help="Optional object URDF path (if missing, only object frame is shown)",
    )
    parser.add_argument("--object-mesh", required=True, help="Object mesh path used for runtime pair leveling.")
    parser.add_argument("--human-root-rot-roll-deg", type=float, default=0.0)
    parser.add_argument("--human-root-rot-pitch-deg", type=float, default=0.0)
    parser.add_argument("--human-root-rot-yaw-deg", type=float, default=0.0)
    parser.add_argument("--object-root-rot-roll-deg", type=float, default=0.0)
    parser.add_argument("--object-root-rot-pitch-deg", type=float, default=0.0)
    parser.add_argument("--object-root-rot-yaw-deg", type=float, default=0.0)
    parser.add_argument("--object-root-local-rot-roll-deg", type=float, default=0.0)
    parser.add_argument("--object-root-local-rot-pitch-deg", type=float, default=0.0)
    parser.add_argument("--object-root-local-rot-yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--object-root-node",
        default="/world/object_base/object_visual",
        help="Scene node path used as root for object URDF loading.",
    )
    parser.add_argument("--object-root-pos-offset-x", type=float, default=0.0)
    parser.add_argument("--object-root-pos-offset-y", type=float, default=0.0)
    parser.add_argument("--object-root-pos-offset-z", type=float, default=0.0)
    parser.add_argument("--object-mesh-scale", type=float, default=1.0, help="Scale object mesh and URDF mesh scale at runtime")
    parser.add_argument("--enable-runtime-pair-leveling", action="store_true")
    parser.add_argument("--runtime-pair-level-target-z", type=float, default=0.0)
    parser.add_argument("--human-root-z-bias", type=float, default=0.05)
    parser.add_argument("--object-root-z-bias", type=float, default=0.03)
    parser.add_argument("--host", default="0.0.0.0", help="Viser host")
    parser.add_argument("--port", type=int, default=8080, help="Viser port")
    parser.add_argument("--show-ground", action="store_true", help="Enable ground grid rendering in viser.")
    parser.add_argument("--ground-size", type=float, default=8.0, help="Ground grid size (meters).")
    parser.add_argument("--ground-cell-size", type=float, default=0.5, help="Ground grid cell size (meters).")
    parser.add_argument("--ground-plane-z", type=float, default=0.0, help="Ground grid plane height (z offset).")
    parser.add_argument("--pair-root-pos-offset-x", type=float, default=0.0, help="Pair-level translation offset x for both human/object.")
    parser.add_argument("--pair-root-pos-offset-y", type=float, default=0.0, help="Pair-level translation offset y for both human/object.")
    parser.add_argument("--pair-root-pos-offset-z", type=float, default=0.0, help="Pair-level translation offset z for both human/object.")
    parser.add_argument("--fps", type=int, default=0, help="Override playback FPS (0 means use file fps)")
    args = parser.parse_args()

    fps_file, root_pos, root_rot_xyzw, dof_pos = load_human_motion(args.human)
    obj_pos, obj_rot_xyzw = load_object_motion(args.object)
    with open(args.human, "rb") as f:
        human_data = pickle.load(f)
    local_body_pos = np.asarray(human_data["local_body_pos"], dtype=np.float64)
    link_body_list = human_data["link_body_list"]
    foot_ids = [i for i, name in enumerate(link_body_list) if any(key in name.lower() for key in ("ankle", "toe", "foot"))]
    if not foot_ids:
        raise ValueError("No ankle/toe/foot links found in human motion for runtime pair leveling.")
    mesh = trimesh.load(args.object_mesh, force="mesh", process=False)
    object_points = np.asarray(mesh.vertices, dtype=np.float64)
    object_points = object_points * float(args.object_mesh_scale)
    object_points = object_points - object_points.mean(axis=0, keepdims=True)

    root_rot_xyzw = apply_root_rot_offset_xyzw(
        root_rot_xyzw,
        (args.human_root_rot_roll_deg, args.human_root_rot_pitch_deg, args.human_root_rot_yaw_deg),
    )
    obj_rot_xyzw = apply_root_rot_offset_xyzw(
        obj_rot_xyzw,
        (args.object_root_rot_roll_deg, args.object_root_rot_pitch_deg, args.object_root_rot_yaw_deg),
    )
    obj_rot_xyzw = apply_root_local_rot_offset_xyzw(
        obj_rot_xyzw,
        (
            args.object_root_local_rot_roll_deg,
            args.object_root_local_rot_pitch_deg,
            args.object_root_local_rot_yaw_deg,
        ),
    )
    object_local_pos_offset = np.array(
        [args.object_root_pos_offset_x, args.object_root_pos_offset_y, args.object_root_pos_offset_z],
        dtype=np.float64,
    )
    obj_pos = obj_pos + R.from_quat(obj_rot_xyzw).apply(object_local_pos_offset)
    pair_trans = np.array(
        [args.pair_root_pos_offset_x, args.pair_root_pos_offset_y, args.pair_root_pos_offset_z],
        dtype=np.float64,
    )
    root_pos = root_pos + pair_trans[None, :]
    obj_pos = obj_pos + pair_trans[None, :]
    if args.enable_runtime_pair_leveling:
        level_rot_xyzw, level_trans = compute_pair_level_transform(
            root_pos,
            root_rot_xyzw,
            local_body_pos,
            foot_ids,
            obj_pos,
            obj_rot_xyzw,
            object_points,
            args.runtime_pair_level_target_z,
        )
        root_pos, root_rot_xyzw, obj_pos, obj_rot_xyzw = apply_pair_level_transform(
            root_pos, root_rot_xyzw, obj_pos, obj_rot_xyzw, level_rot_xyzw, level_trans
        )

    root_pos = root_pos.copy()
    obj_pos = obj_pos.copy()
    root_pos[:, 2] += float(args.human_root_z_bias)
    obj_pos[:, 2] += float(args.object_root_z_bias)

    n = int(min(len(root_pos), len(obj_pos), len(dof_pos), len(root_rot_xyzw), len(obj_rot_xyzw)))
    if n <= 0:
        raise ValueError("No valid frames found in motion data.")

    fps = int(args.fps) if int(args.fps) > 0 else max(1, fps_file)
    dt = 1.0 / float(fps)

    server = viser.ViserServer(host=args.host, port=args.port)

    world = server.scene.add_frame("/world", wxyz=np.array([1.0, 0.0, 0.0, 0.0]), position=np.zeros(3))
    _ = world  # keep handle referenced

    if args.show_ground:
        server.scene.add_grid(
            "/world/ground",
            width=args.ground_size,
            height=args.ground_size,
            plane="xy",
            cell_color=(180, 180, 180),
            section_color=(120, 120, 120),
            cell_size=args.ground_cell_size,
            section_size=max(args.ground_cell_size, 1.0),
            position=(0.0, 0.0, args.ground_plane_z),
            plane_opacity=0.15,
            plane_color=(220, 220, 220),
        )

    robot_base = server.scene.add_frame("/world/robot_base")
    robot = _build_urdf(server, args.robot_urdf, root_node_name="/world/robot_base/robot")

    object_base = server.scene.add_frame("/world/object_base")
    object_visual = server.scene.add_frame("/world/object_base/object_visual")
    object_urdf = None
    object_urdf_path = Path(args.object_urdf)
    if object_urdf_path.exists():
        if abs(float(args.object_mesh_scale) - 1.0) > 1e-8:
            scaled_object_urdf_path = build_scaled_urdf_copy(str(object_urdf_path), float(args.object_mesh_scale))
            print(f"[INFO] using scaled object URDF copy: {scaled_object_urdf_path}")
            object_urdf = _build_urdf(server, str(scaled_object_urdf_path), root_node_name=args.object_root_node)
        else:
            object_urdf = _build_urdf(server, str(object_urdf_path), root_node_name=args.object_root_node)
    else:
        print(f"[WARN] object urdf not found, showing only object frame: {object_urdf_path}")

    qpos_sequence = build_qpos_sequence(root_pos, root_rot_xyzw, dof_pos, obj_pos, obj_rot_xyzw)
    _controls, _vals = create_motion_control_sliders(
        server,
        robot,
        robot_base,
        qpos_sequence,
        robot_dof=int(dof_pos.shape[1]),
        viser_object=object_urdf,
        object_base_frame=object_base,
        contains_object_in_qpos=True,
        initial_fps=fps,
        initial_interp_mult=2,
        loop=True,
    )

    print(f"[READY] Non-physics viewer running at http://{args.host}:{args.port}")
    print(f"[INFO] frames={n}, fps={fps}, dof={dof_pos.shape[1]}")
    print(
        f"[INFO] human_root_rot_offset_deg=({args.human_root_rot_roll_deg:.3f}, "
        f"{args.human_root_rot_pitch_deg:.3f}, {args.human_root_rot_yaw_deg:.3f})"
    )
    print(
        f"[INFO] object_root_rot_offset_deg=({args.object_root_rot_roll_deg:.3f}, "
        f"{args.object_root_rot_pitch_deg:.3f}, {args.object_root_rot_yaw_deg:.3f})"
    )
    print(
        f"[INFO] object_root_local_rot_offset_deg=({args.object_root_local_rot_roll_deg:.3f}, "
        f"{args.object_root_local_rot_pitch_deg:.3f}, {args.object_root_local_rot_yaw_deg:.3f})"
    )
    print(
        f"[INFO] object_root_pos_offset=({args.object_root_pos_offset_x:.3f}, "
        f"{args.object_root_pos_offset_y:.3f}, {args.object_root_pos_offset_z:.3f})"
    )
    print(
        f"[INFO] runtime_pair_leveling={'on' if args.enable_runtime_pair_leveling else 'off'}, "
        f"target_z={args.runtime_pair_level_target_z:.3f}"
    )
    print(
        f"[INFO] pair_root_pos_offset=({args.pair_root_pos_offset_x:.3f}, "
        f"{args.pair_root_pos_offset_y:.3f}, {args.pair_root_pos_offset_z:.3f})"
    )
    print(f"[INFO] object_mesh_scale={args.object_mesh_scale:.3f}")
    print(f"[INFO] root_z_bias(human, object)=({args.human_root_z_bias:.3f}, {args.object_root_z_bias:.3f})")
    print(
        f"[INFO] ground_show={args.show_ground}, ground_size={args.ground_size:.3f}, "
        f"ground_cell_size={args.ground_cell_size:.3f}, ground_plane_z={args.ground_plane_z:.3f}"
    )

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
