from legged_gym.envs.g1.g1_mimic_future_config import G1MimicStuFutureCfg, G1MimicStuFutureCfgDAgger
from legged_gym import LEGGED_GYM_ROOT_DIR


class G1HOICfg(G1MimicStuFutureCfg):

    class env(G1MimicStuFutureCfg.env):
        rand_reset = False
        randomize_start_pos = False
        # Debug switch: disable all episode termination conditions to inspect trajectories.
        disable_termination_for_debug = False
        # Debug-only switch for forcing bike placement in front of robot.
        # Keep disabled in normal training/visualization to follow motion reference.
        enable_bike_test_placement = False
        object_asset_root = f'{LEGGED_GYM_ROOT_DIR}/assets'
        object_urdf_file = 'suitcase/suitcase.urdf'
        object_obj_file = 'suitcase/suitcase.obj'
        # Constant object orientation calibration in degrees [roll_x, pitch_y, yaw_z].
        # This offsets object motion quaternions while keeping positions unchanged.
        object_motion_rot_offset_deg = [0.0, 0.0, 0.0]
        # Global HOI transform applied at load-time in IsaacGym (not baked into motion files).
        # Applies to both human root and object root, preserving relative pose.
        motion_global_rot_offset_deg = [0.0, 0.0, 0.0]  # [roll_x, pitch_y, yaw_z]
        motion_global_pos_offset = [0.0, 0.0, 0.0]      # [x, y, z] meters

        num_actors = 2
        nonblind = True
        n_proprio = G1MimicStuFutureCfg.env.n_proprio
        n_obs_single = G1MimicStuFutureCfg.env.n_mimic_obs + n_proprio
        num_observations = n_obs_single * (G1MimicStuFutureCfg.env.history_len + 1) + (7 if nonblind else 0) + G1MimicStuFutureCfg.env.n_future_obs + G1MimicStuFutureCfg.env.n_masked_priv_info
        n_priv_obs_single = G1MimicStuFutureCfg.env.n_priv_mimic_obs + n_proprio + G1MimicStuFutureCfg.env.n_priv_info
        num_privileged_obs = n_priv_obs_single  

    class motion(G1MimicStuFutureCfg.motion):
        motion_file = f"{LEGGED_GYM_ROOT_DIR}/assets/motions/kneel.pkl"
        object_motion_file = f"{LEGGED_GYM_ROOT_DIR}/assets/motions/kneel.npz"
        # motion_file = f"{LEGGED_GYM_ROOT_DIR}/assets/motions/carry.pkl"
        # object_motion_file = f"{LEGGED_GYM_ROOT_DIR}/assets/motions/carry.npz"
        # motion_file = f"{LEGGED_GYM_ROOT_DIR}/assets/motions/squat.pkl"
        # object_motion_file = f"{LEGGED_GYM_ROOT_DIR}/assets/motions/squat.npz"
        # motion_file = f"{LEGGED_GYM_ROOT_DIR}/assets/motions/chair.pkl"
        # object_motion_file = f"{LEGGED_GYM_ROOT_DIR}/assets/motions/chair.npz"
        
        # Ensure motion curriculum is enabled for difficulty adaptation
        motion_curriculum = False
        motion_curriculum_gamma = 0.01
    
    class domain_rand(G1MimicStuFutureCfg.domain_rand):
        push_robots = True
        push_end_effector = True
        randomize_object = False
        randomize_object_pos = True
        randomize_object_scale = False
        # object_scale_range = [0.9, 1.0]
        randomize_object_friction = True and randomize_object
        object_friction_range = [0.1, 2.]

        randomize_object_mass = True and randomize_object
        added_object_mass_range = [-3.0, 1.0] # [-3.0, 1.0] for task 10

        randomize_object_com = True and randomize_object
        added_object_com_range = [-0.05, 0.05]

    class asset(G1MimicStuFutureCfg.asset):
        file = f'{LEGGED_GYM_ROOT_DIR}/../assets/g1/g1_custom_collision_29dof.urdf'
        object_density = 12
        object_mass = 4.0
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter

    class rewards(G1MimicStuFutureCfg.rewards):
        termination_when_object_far = True
        termination_object_far_threshold = 0.3
        class scales:
            tracking_joint_dof = 2.0
            tracking_joint_vel = 0.2
            tracking_root_translation_z = 1.0
            tracking_root_rotation = 1.0
            tracking_root_linear_vel = 1.0
            tracking_root_angular_vel = 1.0
            tracking_keybody_pos = 1.0
            tracking_keybody_pos_global = 10.0
            feet_slip = -0.1
            feet_contact_forces = -5e-4      
            feet_stumble = -1.25
            dof_pos_limits = -5.0
            dof_torque_limits = -1.0
            dof_vel = -1e-4
            dof_acc = -5e-8
            action_rate = -0.01
            feet_air_time = 5.0
            ang_vel_xy = -0.01            
            ankle_dof_acc = -5e-8 * 2
            ankle_dof_vel = -1e-4 * 2
            tracking_object_point_cloud = 2.0


class G1HOICfgDAgger(G1MimicStuFutureCfgDAgger):
    class runner(G1MimicStuFutureCfgDAgger.runner):
        max_iterations = 500_000
        save_interval = 100

    class policy(G1MimicStuFutureCfgDAgger.policy):
        # init_noise_std = 1.0
        # final_layer_gain = 0.1
        # base_policy_path = None
        init_noise_std = 0.1
        final_layer_gain = 0.01
        base_policy_path = f"{LEGGED_GYM_ROOT_DIR}/assets/checkpoints/base_policy.pt"
        nonblind = G1HOICfg.env.nonblind
        n_obs_single = G1HOICfg.env.n_obs_single
