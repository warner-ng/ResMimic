from legged_gym.envs.g1.g1_hoi_config import G1HOICfg, G1HOICfgDAgger
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

REPO_ROOT_DIR = os.path.abspath(os.path.join(LEGGED_GYM_ROOT_DIR, ".."))

# NOTE: 下面DAgger指的是把base_policy（GMT）继续train的加载
# 实际写reward和domain randomization的时候，直接去上面基础cfg就好

class G1HOIBikeCari4DCfg(G1HOICfg):
    class env(G1HOICfg.env):
        object_asset_root = f"{REPO_ROOT_DIR}/assets"
        object_urdf_file = "bicycle_top_tube/bikered.urdf"  # overwritten by run_cari4d_bike_resmimic.sh
        object_obj_file = "bikered.stl"  # overwritten by run_cari4d_bike_resmimic.sh
        object_scale = 0.400
        enable_runtime_pair_leveling = False  # overwritten by run_cari4d_bike_resmimic.sh
        runtime_pair_level_target_z = 0.000  # overwritten by run_cari4d_bike_resmimic.sh
        human_root_z_bias = 0.000  # overwritten by run_cari4d_bike_resmimic.sh
        object_root_z_bias = 0.100  # overwritten by run_cari4d_bike_resmimic.sh
        object_root_pos_offset = [-0.600, 0.000, 0.100]  # overwritten by run_cari4d_bike_resmimic.sh
        human_root_rot_offset_deg = [0.0, 0.0, 0.0]  # overwritten by run_cari4d_bike_resmimic.sh
        object_root_rot_offset_deg = [-90.0, 0.0, -60.0]  # overwritten by run_cari4d_bike_resmimic.sh
        object_root_local_rot_offset_deg = [0.0, 0.0, 0.0]  # overwritten by run_cari4d_bike_resmimic.sh
        object_motion_rot_offset_deg = [0.0, 0.0, 0.0]  # overwritten by run_cari4d_bike_resmimic.sh
        motion_global_rot_offset_deg = [-90.0, 0.0, 0.0]  # overwritten by run_cari4d_bike_resmimic.sh
        motion_global_pos_offset = [0.000, 0.000, 1.000]  # overwritten by run_cari4d_bike_resmimic.sh
    class motion(G1HOICfg.motion):
        motion_file = f"{REPO_ROOT_DIR}/assets/motions/Date03_Sub01_bike_May_31_19_34_human_upright_bikez_aligned.pkl"  # overwritten by run_cari4d_bike_resmimic.sh
        object_motion_file = f"{REPO_ROOT_DIR}/assets/motions/Date03_Sub01_bike_May_31_19_34_object_upright_bikez_aligned.npz"  # overwritten by run_cari4d_bike_resmimic.sh
    class rewards(G1HOICfg.rewards):
        # 原来是 True；bike 当前物体点云对齐误差会过早触发 termination，先关闭。
        # termination_when_object_far = True
        termination_when_object_far = False
        # inherited original:
        # termination_roll = 1.0
        # termination_pitch = 1.0
        termination_roll = 4.0
        termination_pitch = 4.0
        class scales(G1HOICfg.rewards.scales):
            tracking_object_point_cloud = 2.0


class G1HOIBikeCari4DCfgDAgger(G1HOICfgDAgger):
    class env(G1HOICfgDAgger.env):
        object_asset_root = f"{REPO_ROOT_DIR}/assets"
        object_urdf_file = "bicycle_top_tube/bikered.urdf"  # overwritten by run_cari4d_bike_resmimic.sh
        object_obj_file = "bikered.stl"  # overwritten by run_cari4d_bike_resmimic.sh
        object_scale = 0.400
        enable_runtime_pair_leveling = False  # overwritten by run_cari4d_bike_resmimic.sh
        runtime_pair_level_target_z = 0.000  # overwritten by run_cari4d_bike_resmimic.sh
        human_root_z_bias = 0.000  # overwritten by run_cari4d_bike_resmimic.sh
        object_root_z_bias = 0.100  # overwritten by run_cari4d_bike_resmimic.sh
        object_root_pos_offset = [-0.600, 0.000, 0.100]  # overwritten by run_cari4d_bike_resmimic.sh
        human_root_rot_offset_deg = [0.0, 0.0, 0.0]  # overwritten by run_cari4d_bike_resmimic.sh
        object_root_rot_offset_deg = [-90.0, 0.0, -60.0]  # overwritten by run_cari4d_bike_resmimic.sh
        object_root_local_rot_offset_deg = [0.0, 0.0, 0.0]  # overwritten by run_cari4d_bike_resmimic.sh
        object_motion_rot_offset_deg = [0.0, 0.0, 0.0]  # overwritten by run_cari4d_bike_resmimic.sh
        motion_global_rot_offset_deg = [-90.0, 0.0, 0.0]  # overwritten by run_cari4d_bike_resmimic.sh
        motion_global_pos_offset = [0.000, 0.000, 1.000]  # overwritten by run_cari4d_bike_resmimic.sh
    class motion(G1HOICfgDAgger.motion):
        motion_file = f"{REPO_ROOT_DIR}/assets/motions/Date03_Sub01_bike_May_31_19_34_human_upright_bikez_aligned.pkl"  # overwritten by run_cari4d_bike_resmimic.sh
        object_motion_file = f"{REPO_ROOT_DIR}/assets/motions/Date03_Sub01_bike_May_31_19_34_object_upright_bikez_aligned.npz"  # overwritten by run_cari4d_bike_resmimic.sh
    class rewards(G1HOICfgDAgger.rewards):
        # 原来是 True；bike 当前物体点云对齐误差会过早触发 termination，先关闭。
        # termination_when_object_far = True
        termination_when_object_far = False
        # inherited original:
        # termination_roll = 1.0
        # termination_pitch = 1.0
        termination_roll = 4.0
        termination_pitch = 4.0
        class scales(G1HOICfgDAgger.rewards.scales):
            tracking_object_point_cloud = 2.0
