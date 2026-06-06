#!/usr/bin/env python3
import argparse
import pickle
import sys
from pathlib import Path
from typing import List

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R


try:
    import numpy.core as _np_core
    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
except Exception:
    pass


def load_human(path: Path) -> dict:
    with open(path, "rb") as f:
        data = pickle.load(f)
    required = ("root_pos", "root_rot", "local_body_pos", "link_body_list")
    for key in required:
        if key not in data:
            raise KeyError(f"Missing {key} in human motion: {path}")
    return data


def load_object(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    if "trans" not in data or "rot" not in data:
        raise KeyError(f"Missing trans/rot in object motion: {path}")
    return {"trans": np.asarray(data["trans"]), "rot": np.asarray(data["rot"])}


def get_foot_body_ids(link_body_list: List[str]) -> List[int]:
    ids = []
    for i, name in enumerate(link_body_list):
        lname = name.lower()
        if "ankle" in lname or "toe" in lname or "foot" in lname:
            ids.append(i)
    if not ids:
        raise ValueError("No foot/ankle/toe links found in link_body_list.")
    return ids


def compute_human_foot_bottom_z(root_pos: np.ndarray, root_rot: np.ndarray, local_body_pos: np.ndarray, foot_ids: List[int]) -> np.ndarray:
    mins = []
    for p, q, bodies in zip(root_pos, root_rot, local_body_pos):
        world = R.from_quat(q).apply(bodies[foot_ids]) + p
        mins.append(world[:, 2].min())
    return np.asarray(mins, dtype=np.float64)


def compute_object_mesh_bottom_z(object_trans: np.ndarray, object_rot: np.ndarray, mesh_vertices: np.ndarray) -> np.ndarray:
    mins = []
    for p, q in zip(object_trans, object_rot):
        world = R.from_quat(q).apply(mesh_vertices) + p
        mins.append(world[:, 2].min())
    return np.asarray(mins, dtype=np.float64)


def select_reference(value_series: np.ndarray, stat: str) -> float:
    if stat == "mean":
        return float(value_series.mean())
    if stat == "median":
        return float(np.median(value_series))
    return float(value_series[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply one shared z shift to a human/object pair using pair geometry bottoms.")
    parser.add_argument("--human", type=Path, required=True)
    parser.add_argument("--object", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, required=True)
    parser.add_argument("--output-human", type=Path, required=True)
    parser.add_argument("--output-object", type=Path, required=True)
    parser.add_argument("--target-ground-z", type=float, default=0.0, help="Target z for the chosen pair bottom reference.")
    parser.add_argument("--stat", choices=["first", "mean", "median"], default="first")
    args = parser.parse_args()

    human = load_human(args.human)
    obj = load_object(args.object)
    mesh = trimesh.load(args.object_mesh, force="mesh")
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)

    human_root_pos = np.asarray(human["root_pos"], dtype=np.float64).copy()
    human_root_rot = np.asarray(human["root_rot"], dtype=np.float64)
    human_local_body_pos = np.asarray(human["local_body_pos"], dtype=np.float64)
    object_trans = np.asarray(obj["trans"], dtype=np.float64).copy()
    object_rot = np.asarray(obj["rot"], dtype=np.float64)

    foot_ids = get_foot_body_ids(human["link_body_list"])
    human_bottom_z = compute_human_foot_bottom_z(human_root_pos, human_root_rot, human_local_body_pos, foot_ids)
    object_bottom_z = compute_object_mesh_bottom_z(object_trans, object_rot, mesh_vertices)
    pair_bottom_z = np.minimum(human_bottom_z, object_bottom_z)

    ref_pair_bottom_z = select_reference(pair_bottom_z, args.stat)
    delta_z = float(args.target_ground_z - ref_pair_bottom_z)

    human_root_pos[:, 2] += delta_z
    object_trans[:, 2] += delta_z
    human_bottom_z_after = human_bottom_z + delta_z
    object_bottom_z_after = object_bottom_z + delta_z
    pair_bottom_z_after = pair_bottom_z + delta_z

    grounded_human = dict(human)
    grounded_human["root_pos"] = human_root_pos.astype(np.float32)

    args.output_human.parent.mkdir(parents=True, exist_ok=True)
    args.output_object.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_human, "wb") as f:
        pickle.dump(grounded_human, f)
    np.savez(args.output_object, trans=object_trans.astype(np.float32), rot=np.asarray(obj["rot"], dtype=np.float32))

    print("Grounding finished.")
    print(f"Input human: {args.human}")
    print(f"Input object: {args.object}")
    print(f"Object mesh: {args.object_mesh}")
    print(f"Output human: {args.output_human}")
    print(f"Output object: {args.output_object}")
    print(f"Reference pair bottom z ({args.stat}): {ref_pair_bottom_z:.6f}")
    print(f"Target ground z: {args.target_ground_z:.6f}")
    print(f"Applied shared delta_z: {delta_z:.6f}")
    print(
        f"Human foot bottom after: min={human_bottom_z_after.min():.6f}, "
        f"mean={human_bottom_z_after.mean():.6f}, max={human_bottom_z_after.max():.6f}"
    )
    print(
        f"Object mesh bottom after: min={object_bottom_z_after.min():.6f}, "
        f"mean={object_bottom_z_after.mean():.6f}, max={object_bottom_z_after.max():.6f}"
    )
    print(
        f"Pair bottom after: min={pair_bottom_z_after.min():.6f}, "
        f"mean={pair_bottom_z_after.mean():.6f}, max={pair_bottom_z_after.max():.6f}"
    )


if __name__ == "__main__":
    main()
