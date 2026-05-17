from legged_gym.envs.g1.g1_hoi_config import G1HOICfg, G1HOICfgDAgger
from legged_gym import LEGGED_GYM_ROOT_DIR


class G1HOIBikeCfg(G1HOICfg):
    class env(G1HOICfg.env):
        object_asset_root = f"{LEGGED_GYM_ROOT_DIR}/assets"
        object_urdf_file = "bicycle_top_tube/bike_top_tube_g1style.urdf"
        object_obj_file = "bicycle_top_tube/meshes/bike_top_tube_merged.obj"

    class motion(G1HOICfg.motion):
        motion_file = f"{LEGGED_GYM_ROOT_DIR}/assets/motions/carrying_bike_rack_g1_keybody.pkl"
        object_motion_file = f"{LEGGED_GYM_ROOT_DIR}/assets/motions/carrying_bike_rack_g1_object.npz"


class G1HOIBikeCfgDAgger(G1HOICfgDAgger):
    class env(G1HOICfgDAgger.env):
        object_asset_root = f"{LEGGED_GYM_ROOT_DIR}/assets"
        object_urdf_file = "bicycle_top_tube/bike_top_tube_g1style.urdf"
        object_obj_file = "bicycle_top_tube/meshes/bike_top_tube_merged.obj"

    class motion(G1HOICfgDAgger.motion):
        motion_file = f"{LEGGED_GYM_ROOT_DIR}/assets/motions/carrying_bike_rack_g1_keybody.pkl"
        object_motion_file = f"{LEGGED_GYM_ROOT_DIR}/assets/motions/carrying_bike_rack_g1_object.npz"
