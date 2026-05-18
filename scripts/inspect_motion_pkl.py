#!/usr/bin/env python3
import argparse
import pickle
import numpy as np

def inspect_motion(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)

    print("motion_path:", path)
    print("keys:", list(data.keys()))

    def show(key):
        if key in data:
            v = data[key]
            shape = getattr(v, "shape", None)
            print(f"{key}: type={type(v)}, shape={shape}")

    for key in ["fps", "root_pos", "root_rot", "dof_pos", "dof_vel", "local_body_pos", "link_body_list"]:
        show(key)

    link_body_list = data.get("link_body_list", [])
    if link_body_list:
        print("link_body_list len:", len(link_body_list))
        print("link_body_list sample:", link_body_list[:15])

    if "root_rot" in data:
        root_rot = data["root_rot"]
        if hasattr(root_rot, "shape"):
            print("root_rot first row:", root_rot[0])

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("motion_path", type=str, help="Path to a .pkl motion file")
    args = parser.parse_args()
    inspect_motion(args.motion_path)


if __name__ == "__main__":
    main()
