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
import threading
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import viser
    from viser.extras import ViserUrdf
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "viser is required for this script. Install it first (e.g. pip install viser).\n"
        f"Import error: {type(e).__name__}: {e}"
    )


def xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    q_xyzw = np.asarray(q_xyzw, dtype=np.float64)
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    q_wxyz = np.asarray(q_wxyz, dtype=np.float64)
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)


def _build_urdf(server: viser.ViserServer, urdf_path: str, root_node_name: str):
    """Handle minor API differences across viser versions."""
    urdf_path_obj = Path(urdf_path)
    try:
        # Newer viser expects Path (or yourdfpy.URDF), not plain str.
        return ViserUrdf(server, urdf_path_obj, root_node_name=root_node_name)
    except TypeError:
        try:
            return ViserUrdf(server, urdf_path_obj, root_node_name)
        except TypeError:
            try:
                return ViserUrdf(server, urdf_path_obj)
            except AssertionError:
                # Older/newer API mismatch: load explicit URDF object and pass it in.
                import yourdfpy

                urdf = yourdfpy.URDF.load(urdf_path_obj)
                try:
                    return ViserUrdf(server, urdf, root_node_name=root_node_name)
                except TypeError:
                    try:
                        return ViserUrdf(server, urdf, root_node_name)
                    except TypeError:
                        return ViserUrdf(server, urdf)


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
        default="/home/warner/_projects/ResMimic/legged_gym/assets/chairblack_cari4d/chairblack_cari4d.urdf",
        help="Optional object URDF path (if missing, only object frame is shown)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Viser host")
    parser.add_argument("--port", type=int, default=8080, help="Viser port")
    parser.add_argument("--fps", type=int, default=0, help="Override playback FPS (0 means use file fps)")
    args = parser.parse_args()

    fps_file, root_pos, root_rot_xyzw, dof_pos = load_human_motion(args.human)
    obj_pos, obj_rot_xyzw = load_object_motion(args.object)

    n = int(min(len(root_pos), len(obj_pos), len(dof_pos), len(root_rot_xyzw), len(obj_rot_xyzw)))
    if n <= 0:
        raise ValueError("No valid frames found in motion data.")

    fps = int(args.fps) if int(args.fps) > 0 else max(1, fps_file)
    dt = 1.0 / float(fps)

    server = viser.ViserServer(host=args.host, port=args.port)

    world = server.scene.add_frame("/world", wxyz=np.array([1.0, 0.0, 0.0, 0.0]), position=np.zeros(3))
    _ = world  # keep handle referenced

    robot_base = server.scene.add_frame("/world/robot_base")
    robot = _build_urdf(server, args.robot_urdf, root_node_name="/world/robot_base/robot")

    object_base = server.scene.add_frame("/world/object_base")
    object_urdf = None
    object_urdf_path = Path(args.object_urdf)
    if object_urdf_path.exists():
        object_urdf = _build_urdf(server, str(object_urdf_path), root_node_name="/world/object_base/object")
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

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
