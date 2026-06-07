from legged_gym.envs.g1.g1_hoi_config import G1HOICfg, G1HOICfgDAgger
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

REPO_ROOT_DIR = os.path.abspath(os.path.join(LEGGED_GYM_ROOT_DIR, ".."))


class G1HOIBikeCari4DCfg(G1HOICfg):
    class env(G1HOICfg.env):
        object_asset_root = f"{REPO_ROOT_DIR}/assets"
        object_urdf_file = "bicycle_top_tube/bikered.urdf"
        object_obj_file = "bikered.stl"
        object_scale = 0.450
        enable_runtime_pair_leveling = False
        runtime_pair_level_target_z = 0.000
        human_root_z_bias = 0.000
        object_root_z_bias = 0.000
        object_root_pos_offset = [-0.600, 0.000, 0.070]
        human_root_rot_offset_deg = [0.0, 0.0, 0.0]
        object_root_rot_offset_deg = [-90.0, 0.0, -60.0]
        object_root_local_rot_offset_deg = [0.0, 0.0, 0.0]
        object_motion_rot_offset_deg = [0.0, 0.0, 0.0]
        motion_global_rot_offset_deg = [-90.0, 0.0, 0.0]
        motion_global_pos_offset = [0.000, 0.000, 1.100]
    class motion(G1HOICfg.motion):
        motion_file = f"{REPO_ROOT_DIR}/assets/motions/Date03_Sub01_bike_May_31_19_34_human_upright_bikez_aligned.pkl"
        object_motion_file = f"{REPO_ROOT_DIR}/assets/motions/Date03_Sub01_bike_May_31_19_34_object_upright_bikez_aligned.npz"
    class rewards(G1HOICfg.rewards):
        termination_when_object_far = False
        class scales(G1HOICfg.rewards.scales):
            tracking_object_point_cloud = 0.0


class G1HOIBikeCari4DCfgDAgger(G1HOICfgDAgger):
    class env(G1HOICfgDAgger.env):
        object_asset_root = f"{REPO_ROOT_DIR}/assets"
        object_urdf_file = "bicycle_top_tube/bikered.urdf"
        object_obj_file = "bikered.stl"
        object_scale = 0.450
        enable_runtime_pair_leveling = False
        runtime_pair_level_target_z = 0.000
        human_root_z_bias = 0.000
        object_root_z_bias = 0.000
        object_root_pos_offset = [-0.600, 0.000, 0.070]
        human_root_rot_offset_deg = [0.0, 0.0, 0.0]
        object_root_rot_offset_deg = [-90.0, 0.0, -60.0]
        object_root_local_rot_offset_deg = [0.0, 0.0, 0.0]
        object_motion_rot_offset_deg = [0.0, 0.0, 0.0]
        motion_global_rot_offset_deg = [-90.0, 0.0, 0.0]
        motion_global_pos_offset = [0.000, 0.000, 1.100]
    class motion(G1HOICfgDAgger.motion):
        motion_file = f"{REPO_ROOT_DIR}/assets/motions/Date03_Sub01_bike_May_31_19_34_human_upright_bikez_aligned.pkl"
        object_motion_file = f"{REPO_ROOT_DIR}/assets/motions/Date03_Sub01_bike_May_31_19_34_object_upright_bikez_aligned.npz"
    class rewards(G1HOICfgDAgger.rewards):
        termination_when_object_far = False
        class scales(G1HOICfgDAgger.rewards.scales):
            tracking_object_point_cloud = 0.0
