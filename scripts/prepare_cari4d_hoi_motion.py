#!/usr/bin/env python3
import argparse
import os
import os.path as osp
import pickle
import re
import sys
import shutil

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


# Compatibility for pickles created with newer NumPy layouts (numpy._core.*)
try:
    import numpy.core as _np_core
    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
except Exception:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ResMimic HOI motion from CARI4D output pth")
    parser.add_argument(
        "--cari4d_pth",
        type=str,
        required=True,
        help="Path to CARI4D pth (e.g., output/coconet/.../Date03_Sub03_... .pth)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="in",
        choices=["in", "pr", "gt"],
        help="Which branch in the CARI4D pth to use",
    )
    parser.add_argument("--robot", type=str, default="unitree_g1", help="GMR target robot")
    parser.add_argument("--gender", type=str, default="male", choices=["male", "female", "neutral"])
    parser.add_argument(
        "--tag",
        type=str,
        default="Date03_Sub03_chairblack_lift_it1_input",
        help="Output tag, used in filenames",
    )
    parser.add_argument(
        "--resmimic_root",
        type=str,
        default="/home/warner/_projects/ResMimic",
        help="Path to ResMimic repo root",
    )
    parser.add_argument(
        "--gmr_root",
        type=str,
        default="/home/warner/_projects/GMR",
        help="Path to GMR repo root",
    )
    parser.add_argument(
        "--cari4d_root",
        type=str,
        default="",
        help="Path to CARI4D repo root (needed when torch.load requires modules like 'learning')",
    )
    parser.add_argument(
        "--tgt_fps",
        type=int,
        default=30,
        help="Target FPS for retargeted motion",
    )
    parser.add_argument(
        "--motion_dir",
        type=str,
        default="",
        help="Output motion directory. If empty, auto-pick <resmimic_root>/assets/motions when present, else <resmimic_root>/legged_gym/assets/motions",
    )
    parser.add_argument(
        "--pair_suffix",
        type=str,
        default="",
        help="If set, also emit *_human_upright_<pair_suffix>.pkl and *_object_upright_<pair_suffix>.npz",
    )
    parser.add_argument(
        "--auto_pair_suffixes",
        type=str,
        default="bikez,chairz,suitcasez",
        help="Comma-separated fallback suffixes for upright pair outputs when --pair_suffix is empty",
    )
    parser.add_argument(
        "--object_offset_x",
        type=float,
        default=0.0,
        help="Object translation offset on X axis (meters), applied during object npz export",
    )
    parser.add_argument(
        "--object_offset_y",
        type=float,
        default=0.0,
        help="Object translation offset on Y axis (meters), applied during object npz export",
    )
    parser.add_argument(
        "--object_offset_z",
        type=float,
        default=0.0,
        help="Object translation offset on Z axis (meters), applied during object npz export",
    )
    parser.add_argument("--object_rot_roll_deg", type=float, default=0.0, help="Object rotation offset roll in degrees.")
    parser.add_argument("--object_rot_pitch_deg", type=float, default=0.0, help="Object rotation offset pitch in degrees.")
    parser.add_argument("--object_rot_yaw_deg", type=float, default=0.0, help="Object rotation offset yaw in degrees.")
    return parser.parse_args()


def _frame_time(frame_name: str) -> float:
    frame_name = frame_name.strip()
    t_token = frame_name.split("/")[-1]
    # Some CARI4D exports use numeric frame indices like "000000" instead of "t0.000".
    if t_token.isdigit():
        return float(t_token)
    if not t_token.startswith("t"):
        raise ValueError(f"Unexpected frame token: {t_token}")
    return float(t_token[1:])


def infer_fps(frames):
    # Numeric frame indices usually do not encode wall-clock time.
    # Fall back to 30 FPS for robust downstream retargeting.
    if len(frames) > 0:
        sample_token = str(frames[0]).strip().split("/")[-1]
        if sample_token.isdigit():
            return 30

    ts = np.array([_frame_time(x) for x in frames], dtype=np.float64)
    if len(ts) < 2:
        return 30
    dt = np.diff(ts)
    dt = dt[dt > 1e-6]
    if len(dt) == 0:
        return 30
    return int(np.round(1.0 / np.median(dt)))


def ensure_import_paths(resmimic_root, gmr_root):
    if resmimic_root not in sys.path:
        sys.path.insert(0, resmimic_root)
    if gmr_root not in sys.path:
        sys.path.insert(0, gmr_root)


def infer_cari4d_root(cari4d_pth: str):
    p = osp.abspath(cari4d_pth)
    cur = osp.dirname(p)
    while True:
        if osp.isdir(osp.join(cur, "learning")):
            return cur
        parent = osp.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def _collect_pair_suffixes(pair_suffix: str, auto_pair_suffixes: str):
    suffixes = []
    if pair_suffix and pair_suffix.strip():
        suffixes.append(pair_suffix.strip())

    if auto_pair_suffixes:
        for s in auto_pair_suffixes.split(","):
            s = s.strip()
            if s:
                suffixes.append(s)

    # de-duplicate while preserving order
    uniq = []
    seen = set()
    for s in suffixes:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


def main():
    args = parse_args()

    resmimic_root = osp.abspath(args.resmimic_root)
    gmr_root = osp.abspath(args.gmr_root)
    cari4d_root = osp.abspath(args.cari4d_root) if args.cari4d_root else None
    if cari4d_root is None:
        cari4d_root = infer_cari4d_root(args.cari4d_pth)

    if args.motion_dir:
        motion_dir = osp.abspath(args.motion_dir)
    else:
        root_assets_motion = osp.join(resmimic_root, "assets", "motions")
        legged_assets_motion = osp.join(resmimic_root, "legged_gym", "assets", "motions")
        motion_dir = root_assets_motion if osp.isdir(root_assets_motion) else legged_assets_motion
    os.makedirs(motion_dir, exist_ok=True)

    ensure_import_paths(resmimic_root, gmr_root)
    if cari4d_root and cari4d_root not in sys.path:
        sys.path.insert(0, cari4d_root)

    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
    import mujoco as mj

    # 1) Load CARI4D motion
    # CARI4D exports store structured training-state objects, not plain tensor weights.
    # PyTorch 2.6+ defaults torch.load(..., weights_only=True), which rejects these files.
    data = torch.load(args.cari4d_pth, map_location="cpu", weights_only=False)
    if args.split not in data:
        raise KeyError(f"Split '{args.split}' not found in {args.cari4d_pth}; available: {list(data.keys())}")
    split_data = data[args.split]

    smpl_pose = split_data["smpl_pose"].detach().cpu().numpy()  # (N, 156)
    smpl_t = split_data["smpl_t"].detach().cpu().numpy()        # (N, 3)
    betas = split_data["betas"].detach().cpu().numpy()          # (N, 10)
    frames = split_data["frames"]
    pose_abs = split_data["pose_abs"].detach().cpu().numpy()    # (N, 4, 4)

    src_fps = infer_fps(frames)

    # 2) Build SMPL-X NPZ expected by GMR
    # CARI4D stores SMPLH axis-angle in 156 dims (global 3 + body 63 + hands ...)
    # GMR's smplx loader only needs root_orient(3), pose_body(63), trans(3), betas, mocap_frame_rate, gender.
    root_orient = smpl_pose[:, :3].astype(np.float32)
    pose_body = smpl_pose[:, 3:66].astype(np.float32)
    smpl_trans = smpl_t.astype(np.float32)

    # Use shared shape (10 betas) to match common SMPL-X body model shape channels.
    betas_10 = np.mean(betas, axis=0).astype(np.float32)
    betas_shape = betas_10

    smplx_npz_path = osp.join(motion_dir, f"{args.tag}_smplx_input.npz")
    np.savez(
        smplx_npz_path,
        pose_body=pose_body,
        betas=betas_shape,
        expression=np.zeros((root_orient.shape[0], 10), dtype=np.float32),
        root_orient=root_orient,
        trans=smpl_trans,
        mocap_frame_rate=np.array(src_fps, dtype=np.float32),
        gender=np.array(args.gender),
    )

    # 3) Run GMR retargeting and collect qpos + body global positions
    smplx_body_model_path = osp.join(gmr_root, "assets", "body_models")
    smplx_data, body_model, smplx_output, human_height = load_smplx_file(smplx_npz_path, smplx_body_model_path)
    smplx_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=args.tgt_fps
    )

    retarget = GMR(
        actual_human_height=human_height,
        src_human="smplx",
        tgt_robot=args.robot,
        verbose=False,
        use_velocity_limit=False,
    )

    # Canonical body list used by ResMimic motions (38 links for fullbody)
    carry_pkl = osp.join(resmimic_root, "legged_gym", "assets", "motions", "carry.pkl")
    with open(carry_pkl, "rb") as f:
        carry_data = pickle.load(f)
    link_body_list = carry_data["link_body_list"]

    body_ids = []
    for name in link_body_list:
        bid = mj.mj_name2id(retarget.model, mj.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body '{name}' from carry.pkl not found in GMR robot model {args.robot}")
        body_ids.append(bid)

    qpos_list = []
    local_body_pos_list = []

    for frame_data in smplx_frames:
        qpos = retarget.retarget(frame_data)
        qpos_list.append(qpos.copy())

        # Ensure FK state is up-to-date for body positions
        retarget.configuration.data.qpos[:] = qpos
        mj.mj_forward(retarget.model, retarget.configuration.data)

        root_pos = qpos[:3]
        root_quat_xyzw = np.array([qpos[4], qpos[5], qpos[6], qpos[3]], dtype=np.float64)  # wxyz -> xyzw
        rot_inv = R.from_quat(root_quat_xyzw).inv()

        body_global = retarget.configuration.data.xpos[body_ids]  # (38, 3)
        body_local = rot_inv.apply(body_global - root_pos[None, :])
        local_body_pos_list.append(body_local.astype(np.float32))

    qpos_arr = np.asarray(qpos_list, dtype=np.float32)
    local_body_pos_arr = np.asarray(local_body_pos_list, dtype=np.float32)

    # 4) Build human motion pkl for ResMimic
    root_pos = qpos_arr[:, :3]
    root_rot = np.stack([
        qpos_arr[:, 4],
        qpos_arr[:, 5],
        qpos_arr[:, 6],
        qpos_arr[:, 3],
    ], axis=-1).astype(np.float32)  # wxyz -> xyzw
    dof_pos = qpos_arr[:, 7:]

    human_motion = {
        "fps": int(np.round(aligned_fps)),
        "root_pos": root_pos.astype(np.float32),
        "root_rot": root_rot.astype(np.float32),
        "dof_pos": dof_pos.astype(np.float32),
        "local_body_pos": local_body_pos_arr.astype(np.float32),
        "link_body_list": link_body_list,
    }

    human_pkl_path = osp.join(motion_dir, f"{args.tag}_human.pkl")
    with open(human_pkl_path, "wb") as f:
        pickle.dump(human_motion, f)

    # 5) Build object motion npz from CARI4D pose_abs
    object_trans = pose_abs[:, :3, 3].astype(np.float32)
    object_offset = np.array([args.object_offset_x, args.object_offset_y, args.object_offset_z], dtype=np.float32)
    object_trans = object_trans + object_offset[None, :]
    object_rot_m = R.from_matrix(pose_abs[:, :3, :3])
    object_rot_offset = R.from_euler(
        "xyz",
        [args.object_rot_roll_deg, args.object_rot_pitch_deg, args.object_rot_yaw_deg],
        degrees=True,
    )
    object_rot = (object_rot_m * object_rot_offset).as_quat().astype(np.float32)  # xyzw

    # Align lengths with human frames after fps conversion
    n = min(len(object_trans), len(root_pos))
    object_npz_path = osp.join(motion_dir, f"{args.tag}_object.npz")
    np.savez(
        object_npz_path,
        trans=object_trans[:n],
        rot=object_rot[:n],
    )

    # Also trim human motion if needed
    if len(root_pos) != n:
        human_motion["root_pos"] = human_motion["root_pos"][:n]
        human_motion["root_rot"] = human_motion["root_rot"][:n]
        human_motion["dof_pos"] = human_motion["dof_pos"][:n]
        human_motion["local_body_pos"] = human_motion["local_body_pos"][:n]
        with open(human_pkl_path, "wb") as f:
            pickle.dump(human_motion, f)

    # 6) Emit pair-named files expected by strict launcher scripts
    pair_suffixes = _collect_pair_suffixes(args.pair_suffix, args.auto_pair_suffixes)
    emitted_pair_files = []
    for suffix in pair_suffixes:
        human_pair_path = osp.join(motion_dir, f"{args.tag}_human_upright_{suffix}.pkl")
        object_pair_path = osp.join(motion_dir, f"{args.tag}_object_upright_{suffix}.npz")
        shutil.copyfile(human_pkl_path, human_pair_path)
        shutil.copyfile(object_npz_path, object_pair_path)
        emitted_pair_files.append((human_pair_path, object_pair_path))

    print("[Done] Generated files:")
    print("  SMPLX input:", smplx_npz_path)
    print("  Human motion:", human_pkl_path)
    print("  Object motion:", object_npz_path)
    print("  Object translation offset:", object_offset.tolist())
    print(
        "  Object rotation offset deg:",
        [float(args.object_rot_roll_deg), float(args.object_rot_pitch_deg), float(args.object_rot_yaw_deg)],
    )
    if emitted_pair_files:
        print("  Upright pair files:")
        for h, o in emitted_pair_files:
            print("   -", h)
            print("   -", o)
    print("  Frames:", n)
    print("  FPS:", int(np.round(aligned_fps)))


if __name__ == "__main__":
    main()
