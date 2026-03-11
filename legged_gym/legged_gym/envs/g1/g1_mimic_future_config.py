from legged_gym.envs.g1.g1_mimic_distill_config import G1MimicPrivCfg, G1MimicPrivCfgPPO, G1MimicStuRLTrackingCfgDAgger
from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


class G1MimicStuFutureCfg(G1MimicPrivCfg):
    """Student policy config with future motion support and curriculum masking.
    Extends existing G1MimicPrivCfg to add future motion capabilities."""
    
    class env(G1MimicPrivCfg.env):
        obs_type = 'student_future'
        
        # Keep original student motion steps (current frame only)
        tar_motion_steps = [1]
        
        # Future motion frames (additional input, not in history)
        tar_motion_steps_future = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        
        # Observation dimensions (keep original structure)
        n_mimic_obs_single = 6 + 29
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single  # Current frame only
        n_proprio = G1MimicPrivCfg.env.n_proprio
        
        # Masked privileged information as student input
        use_masked_priv_info = False     # Add masked motion state to student observations
        
        # Future observation dimensions (same structure as teacher priv_mimic_obs)
        # 21 (root_pos + root_pos_dist + orientation + root_vel_local + root_ang_vel_local + deltas) + 29 (dof_pos) + 27 (key_body_pos)
        n_future_obs_single = 6 + 29 + 1 # 64 dims per frame, same as teacher priv_mimic_obs
        n_future_obs = len(tar_motion_steps_future) * n_future_obs_single  # +1 for mask indicator
        
        # Masked privileged information dimensions
        n_masked_priv_info = 9 + 3 if use_masked_priv_info else 0  # motion state (9) + mask indicators (3)
        # Motion state: 3 lin_vel + 3 root_xyz + 3 root_rpy = 9 dims
        
        # Total observation size: maintain original structure + future + masked priv info (no force obs needed)
        n_obs_single = n_mimic_obs + n_proprio  # Current frame observation (for history)
        num_observations = n_obs_single * (G1MimicPrivCfg.env.history_len + 1) + n_future_obs + n_masked_priv_info
        
        # Curriculum masking parameters
        curriculum_masking = True
        masking_start_iteration = 2000  # Start applying masking after this many iterations
        masking_end_iteration = 20000   # Full masking by this iteration
        min_masking_prob = 0.5          # Initial masking probability
        max_masking_prob = 1.0          # Final masking probability
        
        # Temporal dropout parameters
        temporal_dropout = True
        temporal_dropout_start_prob = 0.1  # Start with light dropout
        temporal_dropout_end_prob = 0.5    # End with heavier dropout
        
        # Adaptive masking parameters
        adaptive_masking = True
        performance_threshold = 0.7     # Start increasing masking when performance > this
        masking_adaptation_rate = 0.001 # How fast to adapt masking
    
    class motion(G1MimicPrivCfg.motion):
        # Use a working motion file
        motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/g1_amass_omomo_1.0.yaml"
        
        # Ensure motion curriculum is enabled for difficulty adaptation
        motion_curriculum = False
        motion_curriculum_gamma = 0.01
        
        


class G1MimicStuFutureCfgDAgger(G1MimicStuFutureCfg):
    """DAgger training config for future motion student policy.
    Inherits from G1MimicStuFutureCfg and extends G1MimicStuRLTrackingCfgDAgger."""
    
    seed = 1
    
    class teachercfg(G1MimicPrivCfgPPO):
        pass
    
    class runner(G1MimicPrivCfgPPO.runner):
        policy_class_name = 'ActorCriticFuture'
        algorithm_class_name = 'DaggerPPO'
        runner_class_name = 'OnPolicyDaggerRunner'
        max_iterations = 1_000_002
        warm_iters = 100
        
        # logging
        save_interval = 500
        experiment_name = 'test'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        
        teacher_experiment_name = 'test'
        teacher_proj_name = 'g1_priv_mimic'
        teacher_checkpoint = -1
        eval_student = False
        
        # Wandb model saving option
        save_to_wandb = True  # Set to False to disable wandb model saving

    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005
        
        dagger_coef_anneal_steps = 60000  # Total steps to anneal dagger_coef to dagger_coef_min
        dagger_coef = 0.2
        dagger_coef_min = 0.1
        
        # Future motion specific parameters
        future_weight_decay = 0.95      # Decay weight for older future frames
        future_consistency_loss = 0.1   # Weight for consistency loss between future predictions

    class policy(HumanoidMimicCfgPPO.policy):
        action_std = [0.7] * 12 + [0.4] * 3 + [0.5] * 14
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
        layer_norm = True
        motion_latent_dim = 128
        
        # Future motion encoder parameters
        future_encoder_dims = [256, 256, 128]  # Separate encoder for future motion
        future_attention_heads = 4              # Multi-head attention for future frames
        future_dropout = 0.1                   # Dropout for future encoder
        temporal_embedding_dim = 64            # Temporal position embedding
        future_latent_dim = 128                # Future motion latent dimension
        num_future_steps = 10                  # Number of future steps to expect
        
        # Explicit future observation dimensions (to avoid calculation errors with masked priv info)
        num_future_observations = G1MimicStuFutureCfg.env.n_future_obs  # 360