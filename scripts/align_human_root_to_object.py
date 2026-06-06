#!/usr/bin/env python3
"""Apply a rigid alignment to post-GMR human root motion.

Default behavior:
- estimate one global XY translation from object_root_xy - human_root_xy
- apply that same translation to every human root frame

Optional behavior:
- estimate one global yaw rotation from object motion direction vs human motion direction
- apply the same yaw to root_pos and root_rot for every frame

This script never scales or deforms the motion.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


try:
    import numpy.core as _np_core
    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
except Exception:
    pass


def load_human_motion(path: Path) -> dict:
    with open(path, "rb") as f:
        data = pickle.load(f)
    required = ("root_pos", "root_rot", "dof_pos")
    for key in required:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in human motion: {path}")
    return data


def load_object_motion(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    required = ("trans", "rot")
    for key in required:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in object motion: {path}")
    return np.asarray(data["trans"], dtype=np.float64), np.asarray(data["rot"], dtype=np.float64)


def trim_pair(human_data: dict, object_pos: np.ndarray) -> tuple[dict, np.ndarray]:
    n = min(human_data["root_pos"].shape[0], object_pos.shape[0])
    trimmed = dict(human_data)
    for key in ("root_pos", "root_rot", "dof_pos", "local_body_pos"):
        if key in trimmed:
            trimmed[key] = np.asarray(trimmed[key])[:n].copy()
    return trimmed, object_pos[:n].copy()


def fit_global_xy_translation(human_root_pos: np.ndarray, object_root_pos: np.ndarray) -> np.ndarray:
    delta_xy = object_root_pos[:, :2] - human_root_pos[:, :2]
    return delta_xy.mean(axis=0)


def fit_global_yaw_deg(human_root_pos: np.ndarray, object_root_pos: np.ndarray) -> float:
    human_disp = human_root_pos[-1, :2] - human_root_pos[0, :2]
    object_disp = object_root_pos[-1, :2] - object_root_pos[0, :2]
    human_norm = np.linalg.norm(human_disp)
    object_norm = np.linalg.norm(object_disp)
    if human_norm < 1e-8 or object_norm < 1e-8:
        return 0.0
    human_yaw = np.arctan2(human_disp[1], human_disp[0])
    object_yaw = np.arctan2(object_disp[1], object_disp[0])
    return float(np.rad2deg(object_yaw - human_yaw))


def apply_global_rigid_transform(
    human_root_pos: np.ndarray,
    human_root_rot_xyzw: np.ndarray,
    yaw_deg: float,
    translation_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    yaw_rot = R.from_euler("z", yaw_deg, degrees=True)

    out_pos = human_root_pos.copy()
    out_rot = human_root_rot_xyzw.copy()

    out_pos = yaw_rot.apply(out_pos)
    out_pos[:, 0] += float(translation_xy[0])
    out_pos[:, 1] += float(translation_xy[1])

    root_rot = R.from_quat(out_rot)
    out_rot = (yaw_rot * root_rot).as_quat()
    return out_pos, out_rot


def summarize_xy_distance(human_root_pos: np.ndarray, object_root_pos: np.ndarray) -> tuple[float, float, float]:
    delta_xy = human_root_pos[:, :2] - object_root_pos[:, :2]
    dist = np.linalg.norm(delta_xy, axis=1)
    return float(dist.mean()), float(np.percentile(dist, 95)), float(dist.max())


def main() -> None:
    parser = argparse.ArgumentParser(description="Rigidly align post-GMR human root to object root.")
    parser.add_argument("--human", type=Path, required=True, help="Input post-GMR human pkl.")
    parser.add_argument("--object", type=Path, required=True, help="Reference object npz.")
    parser.add_argument("--output", type=Path, required=True, help="Output aligned human pkl.")
    parser.add_argument(
        "--align-yaw",
        action="store_true",
        help="Estimate and apply one global yaw rotation before solving translation.",
    )
    args = parser.parse_args()

    human_data = load_human_motion(args.human)
    object_root_pos, _ = load_object_motion(args.object)
    human_data, object_root_pos = trim_pair(human_data, object_root_pos)

    human_root_pos = np.asarray(human_data["root_pos"], dtype=np.float64)
    human_root_rot = np.asarray(human_data["root_rot"], dtype=np.float64)

    before_mean, before_p95, before_max = summarize_xy_distance(human_root_pos, object_root_pos)

    yaw_deg = 0.0
    if args.align_yaw:
        yaw_deg = fit_global_yaw_deg(human_root_pos, object_root_pos)

    rotated_pos, rotated_rot = apply_global_rigid_transform(
        human_root_pos=human_root_pos,
        human_root_rot_xyzw=human_root_rot,
        yaw_deg=yaw_deg,
        translation_xy=np.zeros(2, dtype=np.float64),
    )
    translation_xy = fit_global_xy_translation(rotated_pos, object_root_pos)
    aligned_pos, aligned_rot = apply_global_rigid_transform(
        human_root_pos=human_root_pos,
        human_root_rot_xyzw=human_root_rot,
        yaw_deg=yaw_deg,
        translation_xy=translation_xy,
    )

    aligned = dict(human_data)
    aligned["root_pos"] = aligned_pos.astype(np.float32)
    aligned["root_rot"] = aligned_rot.astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(aligned, f)

    after_mean, after_p95, after_max = summarize_xy_distance(aligned_pos, object_root_pos)

    print("Rigid alignment finished.")
    print(f"Input human: {args.human}")
    print(f"Reference object: {args.object}")
    print(f"Output human: {args.output}")
    print(f"Frames used: {aligned_pos.shape[0]}")
    print(f"Applied yaw deg: {yaw_deg:.6f}")
    print(f"Applied translation xy: [{translation_xy[0]:.6f}, {translation_xy[1]:.6f}]")
    print(
        f"XY distance before: mean={before_mean:.6f}, p95={before_p95:.6f}, max={before_max:.6f}"
    )
    print(
        f"XY distance after:  mean={after_mean:.6f}, p95={after_p95:.6f}, max={after_max:.6f}"
    )


if __name__ == "__main__":
    main()
