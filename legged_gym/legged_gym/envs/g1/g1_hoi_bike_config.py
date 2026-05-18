from legged_gym.envs.g1.g1_hoi_config import G1HOICfg, G1HOICfgDAgger
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

REPO_ROOT_DIR = os.path.abspath(os.path.join(LEGGED_GYM_ROOT_DIR, ".."))


class G1HOIBikeCfg(G1HOICfg):
    class env(G1HOICfg.env):
        object_asset_root = f"{REPO_ROOT_DIR}/assets"
        object_urdf_file = "bicycle_top_tube/bike_top_tube_vscode.urdf"
        object_obj_file = "bicycle_top_tube/meshes/bike_top_tube_merged.obj"

    class motion(G1HOICfg.motion):
        motion_file = f"{REPO_ROOT_DIR}/assets/motions/carrying_bike_rack_g1_fullbody.pkl"
        object_motion_file = f"{REPO_ROOT_DIR}/assets/motions/carrying_bike_rack_g1_object.npz"

    class rewards(G1HOICfg.rewards):
        termination_when_object_far = False
        class scales(G1HOICfg.rewards.scales):
            tracking_object_point_cloud = 0.0


class G1HOIBikeCfgDAgger(G1HOICfgDAgger):
    class env(G1HOICfgDAgger.env):
        object_asset_root = f"{REPO_ROOT_DIR}/assets"
        object_urdf_file = "bicycle_top_tube/bike_top_tube_single.urdf"
        object_obj_file = "bicycle_top_tube/meshes/bike_top_tube_merged.obj"

    class motion(G1HOICfgDAgger.motion):
        motion_file = f"{REPO_ROOT_DIR}/assets/motions/carrying_bike_rack_g1_fullbody.pkl"
        object_motion_file = f"{REPO_ROOT_DIR}/assets/motions/carrying_bike_rack_g1_object.npz"

    class rewards(G1HOICfgDAgger.rewards):
        termination_when_object_far = False
        class scales(G1HOICfgDAgger.rewards.scales):
            tracking_object_point_cloud = 0.0
