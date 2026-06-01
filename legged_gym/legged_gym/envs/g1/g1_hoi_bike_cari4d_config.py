from legged_gym.envs.g1.g1_hoi_config import G1HOICfg, G1HOICfgDAgger
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

REPO_ROOT_DIR = os.path.abspath(os.path.join(LEGGED_GYM_ROOT_DIR, ".."))


class G1HOIBikeCari4DCfg(G1HOICfg):
    class env(G1HOICfg.env):
        object_asset_root = f"{REPO_ROOT_DIR}/assets"
        object_urdf_file = "hy3d_bike_cari4d/hy3d_bike_cari4d.urdf"
        object_obj_file = "hy3d_bike_cari4d/Date03_Sub01_bike_wild001_000_align_centered_x20.obj"
        object_motion_rot_offset_deg = [0.0, 0.0, 0.0]
        motion_global_rot_offset_deg = [0.000000, 180.000000, 90.000000]
        motion_global_pos_offset = [0.000000, 0.000000, 0.500000]
    class motion(G1HOICfg.motion):
        motion_file = f"{REPO_ROOT_DIR}/assets/motions/Date03_Sub01_bike_on_may_29_21_17_human_upright_bikez.pkl"
        object_motion_file = f"{REPO_ROOT_DIR}/assets/motions/Date03_Sub01_bike_on_may_29_21_17_object_upright_bikez.npz"
    class rewards(G1HOICfg.rewards):
        termination_when_object_far = False
        class scales(G1HOICfg.rewards.scales):
            tracking_object_point_cloud = 0.0


class G1HOIBikeCari4DCfgDAgger(G1HOICfgDAgger):
    class env(G1HOICfgDAgger.env):
        object_asset_root = f"{REPO_ROOT_DIR}/assets"
        object_urdf_file = "hy3d_bike_cari4d/hy3d_bike_cari4d.urdf"
        object_obj_file = "hy3d_bike_cari4d/Date03_Sub01_bike_wild001_000_align_centered_x20.obj"
        object_motion_rot_offset_deg = [0.0, 0.0, 0.0]
        motion_global_rot_offset_deg = [0.000000, 180.000000, 90.000000]
        motion_global_pos_offset = [0.000000, 0.000000, 1.000000]
    class motion(G1HOICfgDAgger.motion):
        motion_file = f"{REPO_ROOT_DIR}/assets/motions/Date03_Sub01_bike_on_may_29_21_17_human_upright_bikez.pkl"
        object_motion_file = f"{REPO_ROOT_DIR}/assets/motions/Date03_Sub01_bike_on_may_29_21_17_object_upright_bikez.npz"
    class rewards(G1HOICfgDAgger.rewards):
        termination_when_object_far = False
        class scales(G1HOICfgDAgger.rewards.scales):
            tracking_object_point_cloud = 0.0