from legged_gym.envs.g1.g1_hoi_config import G1HOICfg, G1HOICfgDAgger
from legged_gym import LEGGED_GYM_ROOT_DIR


class G1HOICari4DChairCfg(G1HOICfg):
    class env(G1HOICfg.env):
        object_asset_root = f"{LEGGED_GYM_ROOT_DIR}/assets"
        object_urdf_file = "chairblack_cari4d/chairblack_cari4d.urdf"
        object_obj_file = "chairblack_cari4d/Date03_Sub03_chairblack_lift_t0001.500_k2_rgba_align.obj"
        # Position is already correct; keep yaw and tune pitch first.
        object_motion_rot_offset_deg = [0.0, 0.0, 0.0]
    class motion(G1HOICfg.motion):
        motion_file = f"{LEGGED_GYM_ROOT_DIR}/assets/motions/Date03_Sub03_chairblack_lift_it1_input_human_upright_suitcasez.pkl"
        object_motion_file = f"{LEGGED_GYM_ROOT_DIR}/assets/motions/Date03_Sub03_chairblack_lift_it1_input_object_upright_suitcasez.npz"
    class rewards(G1HOICfg.rewards):
        termination_when_object_far = False


class G1HOICari4DChairCfgDAgger(G1HOICfgDAgger):
    class env(G1HOICfgDAgger.env):
        object_asset_root = f"{LEGGED_GYM_ROOT_DIR}/assets"
        object_urdf_file = "chairblack_cari4d/chairblack_cari4d.urdf"
        object_obj_file = "chairblack_cari4d/Date03_Sub03_chairblack_lift_t0001.500_k2_rgba_align.obj"
        object_motion_rot_offset_deg = [0.0, 0.0, 0.0]
    class motion(G1HOICfgDAgger.motion):
        motion_file = f"{LEGGED_GYM_ROOT_DIR}/assets/motions/Date03_Sub03_chairblack_lift_it1_input_human_upright_suitcasez.pkl"
        object_motion_file = f"{LEGGED_GYM_ROOT_DIR}/assets/motions/Date03_Sub03_chairblack_lift_it1_input_object_upright_suitcasez.npz"
    class rewards(G1HOICfgDAgger.rewards):
        termination_when_object_far = False
