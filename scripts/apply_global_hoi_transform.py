#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R


def parse_args():
    p = argparse.ArgumentParser(description="Apply the same global rigid transform to human/object HOI motions.")
    p.add_argument("--human_in", required=True, type=str)
    p.add_argument("--object_in", required=True, type=str)
    p.add_argument("--human_out", required=True, type=str)
    p.add_argument("--object_out", required=True, type=str)
    p.add_argument("--yaw_deg", type=float, default=0.0, help="Global yaw rotation in degrees (around +Z).")
    p.add_argument("--pitch_deg", type=float, default=0.0, help="Global pitch rotation in degrees (around +Y).")
    p.add_argument("--roll_deg", type=float, default=0.0, help="Global roll rotation in degrees (around +X).")
    p.add_argument("--x_offset", type=float, default=0.0, help="Global X translation in meters.")
    p.add_argument("--y_offset", type=float, default=0.0, help="Global Y translation in meters.")
    p.add_argument("--z_offset", type=float, default=0.0, help="Global Z translation in meters.")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.human_in, "rb") as f:
        human = pickle.load(f)
    obj = np.load(args.object_in)

    for key in ["root_pos", "root_rot"]:
        if key not in human:
            raise KeyError(f"Human motion missing key: {key}")
    for key in ["trans", "rot"]:
        if key not in obj:
            raise KeyError(f"Object motion missing key: {key}")

    rc = R.from_euler("xyz", [args.roll_deg, args.pitch_deg, args.yaw_deg], degrees=True)
    t = np.array([args.x_offset, args.y_offset, args.z_offset], dtype=np.float64)

    # Human
    root_pos = np.asarray(human["root_pos"], dtype=np.float64)
    root_rot = np.asarray(human["root_rot"], dtype=np.float64)  # xyzw

    root_pos_new = rc.apply(root_pos) + t[None, :]
    root_rot_new = (rc * R.from_quat(root_rot)).as_quat()

    human["root_pos"] = root_pos_new.astype(np.float32)
    human["root_rot"] = root_rot_new.astype(np.float32)

    # Object
    trans = np.asarray(obj["trans"], dtype=np.float64)
    rot = np.asarray(obj["rot"], dtype=np.float64)  # xyzw

    trans_new = rc.apply(trans) + t[None, :]
    rot_new = (rc * R.from_quat(rot)).as_quat()

    with open(args.human_out, "wb") as f:
        pickle.dump(human, f)

    np.savez(
        args.object_out,
        trans=trans_new.astype(np.float32),
        rot=rot_new.astype(np.float32),
    )

    print("[Done] Applied global transform:")
    print(f"  yaw/pitch/roll(deg): {args.yaw_deg}, {args.pitch_deg}, {args.roll_deg}")
    print(f"  xyz offset(m): {args.x_offset}, {args.y_offset}, {args.z_offset}")
    print(f"  human_out: {args.human_out}")
    print(f"  object_out: {args.object_out}")


if __name__ == "__main__":
    main()
