from isaacgym.torch_utils import *
from isaacgym import gymapi, gymtorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from legged_gym.envs.g1.g1_mimic_distill import G1MimicDistill
from .g1_mimic_future_config import G1MimicStuFutureCfg
from legged_gym.gym_utils.math import *
from pose.utils import torch_utils
from legged_gym.envs.base.legged_robot import euler_from_quaternion
from legged_gym.envs.base.humanoid_char import convert_to_local_root_body_pos, convert_to_global_root_body_pos


class G1MimicFuture(G1MimicDistill):
    """Student policy environment with future motion support and curriculum masking.
    Extends G1MimicDistill to add future motion capabilities while maintaining 
    all original RL+BC functionality.
    
    Curriculum Masked Privilege Information (CMP)
    """
    
    def __init__(self, cfg: G1MimicStuFutureCfg, sim_params, physics_engine, sim_device, headless):
        # Store future motion configuration
        self.future_cfg = cfg.env
        
        # Initialize masking parameters
        self.curriculum_masking = getattr(cfg.env, 'curriculum_masking', True)
        self.masking_start_iteration = getattr(cfg.env, 'masking_start_iteration', 2000)
        self.masking_end_iteration = getattr(cfg.env, 'masking_end_iteration', 20000)
        self.min_masking_prob = getattr(cfg.env, 'min_masking_prob', 0.0)
        self.max_masking_prob = getattr(cfg.env, 'max_masking_prob', 0.8)
        
        # Temporal dropout parameters
        self.temporal_dropout = getattr(cfg.env, 'temporal_dropout', True)
        self.temporal_dropout_start_prob = getattr(cfg.env, 'temporal_dropout_start_prob', 0.1)
        self.temporal_dropout_end_prob = getattr(cfg.env, 'temporal_dropout_end_prob', 0.5)
        
        # Adaptive masking parameters
        self.adaptive_masking = getattr(cfg.env, 'adaptive_masking', True)
        self.performance_threshold = getattr(cfg.env, 'performance_threshold', 0.7)
        self.masking_adaptation_rate = getattr(cfg.env, 'masking_adaptation_rate', 0.001)
        
        # Evaluation mode parameters
        self.evaluation_mode = getattr(cfg.env, 'evaluation_mode', False)
        self.force_full_masking = getattr(cfg.env, 'force_full_masking', False)
        
        # Masked privileged information parameters
        self.use_masked_priv_info = getattr(cfg.env, 'use_masked_priv_info', False)
        
        # Initialize masking state
        self.current_masking_prob = self.min_masking_prob
        self.recent_performance = 0.0
        
        # Initialize FALCON-style curriculum force application flag BEFORE super().__init__
        # This is needed because reset_idx is called during parent initialization
        self.enable_force_curriculum = getattr(cfg.env, 'enable_force_curriculum', False)
        
        # Initialize force curriculum attributes with default values before calling super().__init__
        # This prevents AttributeError during reset_idx call in parent initialization
        if self.enable_force_curriculum:
            # Essential attributes that are needed before full initialization
            self.force_scale_curriculum = True  # Will be properly set in _init_force_curriculum_components
            # Initialize with empty values - will be properly set after super().__init__()
            self.episode_length_counter = None
            self.force_scale = None
        
        # Call parent constructor
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # Fix motion difficulty initialization - should start at 10, not 100
        num_motions = self._motion_lib.num_motions()
        self.motion_difficulty = 10 * torch.ones((num_motions), device=self.device, dtype=torch.float)
        self.mean_motion_difficulty = 10.
        
        # Only initialize future motion components if obs_type is 'student_future'
        if self.obs_type == 'student_future':
            # Initialize future motion target steps
            self._tar_motion_steps_future = torch.tensor(
                getattr(cfg.env, 'tar_motion_steps_future', [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]), 
                device=self.device, dtype=torch.long
            )
            
            # Find indices of future steps in the privileged teacher steps
            self._tar_motion_steps_future_idx = torch.searchsorted(
                self._tar_motion_steps_priv, self._tar_motion_steps_future
            )
            
            # Initialize masking buffer
            self.future_mask = torch.ones(
                (self.num_envs, len(self._tar_motion_steps_future)), 
                device=self.device, dtype=torch.bool
            )
            
            print(f"Future motion enabled with {len(self._tar_motion_steps_future)} future frames")
            print(f"Future steps: {self._tar_motion_steps_future.tolist()}")

    def _get_unified_motion_data(self):
        """Get unified motion data for both privileged and future frames in a single sampling call.
        Returns processed motion data that can be used by both _get_mimic_obs and _get_future_motion_obs.
        """
        # Determine which time steps to sample
        if self.obs_type == 'student_future' and hasattr(self, '_tar_motion_steps_future'):
            all_steps = torch.cat([self._tar_motion_steps_priv, self._tar_motion_steps_future])
            num_priv_steps = self._tar_motion_steps_priv.shape[0]
            num_future_steps = self._tar_motion_steps_future.shape[0]
        else:
            all_steps = self._tar_motion_steps_priv
            num_priv_steps = self._tar_motion_steps_priv.shape[0]
            num_future_steps = 0
        
        total_steps = all_steps.shape[0]
        assert total_steps > 0, "Invalid number of target observation steps"
        
        # Single motion sampling call for all time steps
        motion_times = self._get_motion_times().unsqueeze(-1)
        obs_motion_times = all_steps * self.dt + motion_times
        motion_ids_tiled = torch.broadcast_to(self._motion_ids.unsqueeze(-1), obs_motion_times.shape)
        motion_ids_tiled = motion_ids_tiled.flatten()
        obs_motion_times = obs_motion_times.flatten()
        
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, root_pos_delta_local, root_rot_delta_local = \
            self._motion_lib.calc_motion_frame(motion_ids_tiled, obs_motion_times)
        
        # Unified data processing for all frames
        roll, pitch, yaw = euler_from_quaternion(root_rot)
        roll = roll.reshape(self.num_envs, total_steps, 1)
        pitch = pitch.reshape(self.num_envs, total_steps, 1)
        yaw = yaw.reshape(self.num_envs, total_steps, 1)

        root_vel_local = quat_rotate_inverse(root_rot, root_vel)
        root_ang_vel_local = quat_rotate_inverse(root_rot, root_ang_vel)
        
        whole_key_body_pos = body_pos[:, self._key_body_ids_motion, :] # local body pos
        whole_key_body_pos_global = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=whole_key_body_pos)
        
        # Reshape all observations (unified)
        root_pos = root_pos.reshape(self.num_envs, total_steps, root_pos.shape[-1])
        root_vel = root_vel.reshape(self.num_envs, total_steps, root_vel.shape[-1])
        root_rot = root_rot.reshape(self.num_envs, total_steps, root_rot.shape[-1])
        root_ang_vel = root_ang_vel.reshape(self.num_envs, total_steps, root_ang_vel.shape[-1])
        dof_pos = dof_pos.reshape(self.num_envs, total_steps, dof_pos.shape[-1])
        dof_vel = dof_vel.reshape(self.num_envs, total_steps, dof_vel.shape[-1])
        root_vel_local = root_vel_local.reshape(self.num_envs, total_steps, root_vel_local.shape[-1])
        root_ang_vel_local = root_ang_vel_local.reshape(self.num_envs, total_steps, root_ang_vel_local.shape[-1])
        root_pos_delta_local = root_pos_delta_local.reshape(self.num_envs, total_steps, root_pos_delta_local.shape[-1])
        root_rot_delta_local = root_rot_delta_local.reshape(self.num_envs, total_steps, root_rot_delta_local.shape[-1])
        whole_key_body_pos = whole_key_body_pos.reshape(self.num_envs, total_steps, -1)
        whole_key_body_pos_global = whole_key_body_pos_global.reshape(self.num_envs, total_steps, -1)
        
        root_pos_distance_to_target = root_pos - self.root_states[:, 0:3].reshape(self.num_envs, 1, -1)
        
        return {
            'root_pos': root_pos,
            'root_vel': root_vel,
            'root_rot': root_rot,
            'root_ang_vel': root_ang_vel,
            'dof_pos': dof_pos,
            'dof_vel': dof_vel,
            'body_pos': body_pos,
            'root_pos_delta_local': root_pos_delta_local,
            'root_rot_delta_local': root_rot_delta_local,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'root_vel_local': root_vel_local,
            'root_ang_vel_local': root_ang_vel_local,
            'whole_key_body_pos': whole_key_body_pos,
            'whole_key_body_pos_global': whole_key_body_pos_global,
            'root_pos_distance_to_target': root_pos_distance_to_target,
            'num_priv_steps': num_priv_steps,
            'num_future_steps': num_future_steps,
            'total_steps': total_steps
        }

    def _build_future_obs_from_data(self, motion_data):
        """Build future motion observations from unified motion data."""
        if self.obs_type != 'student_future':
            return torch.zeros(self.num_envs, 0, device=self.device)
            
        if motion_data['num_future_steps'] == 0:
            return torch.zeros(self.num_envs, 0, device=self.device)
        
        # Extract future motion data (from privileged steps onwards)
        num_priv_steps = motion_data['num_priv_steps']
        
        # Get future frame data by slicing unified data
        root_pos = motion_data['root_pos'][:, num_priv_steps:]
        root_vel_local = motion_data['root_vel_local'][:, num_priv_steps:]
        root_ang_vel_local = motion_data['root_ang_vel_local'][:, num_priv_steps:]
        roll = motion_data['roll'][:, num_priv_steps:]
        pitch = motion_data['pitch'][:, num_priv_steps:]
        dof_pos = motion_data['dof_pos'][:, num_priv_steps:]
        
        # Future motion observations (same rich structure as teacher priv_mimic_obs)
        future_obs = torch.cat((
            # root position: xy velocity + z position
            root_vel_local[..., :2], # 2 dims (xy velocity instead of xy position)
            root_pos[..., 2:3], # 1 dim (z position)
            # root rotation: roll/pitch + yaw angular velocity
            roll, pitch, # 2 dims (roll/pitch orientation)
            root_ang_vel_local[..., 2:3], # 1 dim (yaw angular velocity)
            dof_pos, # num_dof dims
        ), dim=-1) # shape: (num_envs, num_future_steps, 6 + num_dof)
        
        return future_obs

    def _apply_curriculum_masking(self, future_obs):
        """Apply curriculum masking to future motion observations."""
        if self.obs_type != 'student_future':
            return future_obs
        
        # Handle case when future_obs is 2D (no future steps)
        if len(future_obs.shape) == 2:
            return future_obs
            
        num_envs, num_future_steps, obs_dim = future_obs.shape
        
        # Force full masking in evaluation mode
        if self.evaluation_mode and self.force_full_masking:
            self.current_masking_prob = 1.0
            # print(f"Evaluation mode: forcing full masking (prob=1.0)")
        # Update masking probability based on training progress
        elif self.curriculum_masking:
            # Calculate curriculum progress
            if hasattr(self, 'global_counter'):
                current_iter = self.global_counter // 24  # Convert to training iterations
            else:
                current_iter = 0
                
            progress = max(0, min(1, (current_iter - self.masking_start_iteration) / 
                                 max(1, self.masking_end_iteration - self.masking_start_iteration)))
            curriculum_prob = self.min_masking_prob + progress * (self.max_masking_prob - self.min_masking_prob)
            
            # Adaptive masking based on performance
            if self.adaptive_masking and self.recent_performance > self.performance_threshold:
                adaptation = min(0.2, (self.recent_performance - self.performance_threshold) * self.masking_adaptation_rate)
                curriculum_prob = min(1.0, curriculum_prob + adaptation)
            
            self.current_masking_prob = curriculum_prob
        
        # Apply temporal dropout
        if self.temporal_dropout:
            if hasattr(self, 'global_counter'):
                current_iter = self.global_counter // 24
            else:
                current_iter = 0
            dropout_progress = max(0, min(1, current_iter / max(1, self.masking_end_iteration)))
            dropout_prob = self.temporal_dropout_start_prob + dropout_progress * \
                          (self.temporal_dropout_end_prob - self.temporal_dropout_start_prob)
            
            # Random temporal dropout
            temporal_mask = torch.rand(num_envs, num_future_steps, device=self.device) > dropout_prob
            self.future_mask = temporal_mask
        
        # Apply progressive masking (mask later frames more aggressively)
        if (self.curriculum_masking and self.current_masking_prob > 0) or (self.evaluation_mode and self.force_full_masking):
            if self.evaluation_mode and self.force_full_masking:
                # In evaluation mode, mask all future frames
                progressive_mask = torch.zeros(num_envs, num_future_steps, device=self.device, dtype=torch.bool)
            else:
                # Create progressive mask - later frames have higher masking probability
                time_weights = torch.linspace(0, 1, num_future_steps, device=self.device)
                progressive_probs = self.current_masking_prob * (1 + time_weights)
                progressive_probs = torch.clamp(progressive_probs, 0, 1)
                
                # Apply progressive masking
                random_mask = torch.rand(num_envs, num_future_steps, device=self.device)
                progressive_mask = random_mask > progressive_probs.unsqueeze(0)
            
            # Combine with temporal dropout
            if self.temporal_dropout and not (self.evaluation_mode and self.force_full_masking):
                final_mask = self.future_mask & progressive_mask
            else:
                final_mask = progressive_mask
                
            self.future_mask = final_mask
        
        # Apply masking to observations
        masked_future_obs = future_obs.clone()
        
        # Zero out masked frames
        if self.curriculum_masking or self.temporal_dropout or (self.evaluation_mode and self.force_full_masking):
            masked_future_obs[~self.future_mask] = 0.0
            
            # Add masking indicator as additional feature
            masking_indicator = self.future_mask.float().unsqueeze(-1)  # (num_envs, num_future_steps, 1)
            masked_future_obs = torch.cat([masked_future_obs, masking_indicator], dim=-1)
        
        return masked_future_obs

    def _get_masked_motion_state_obs(self):
        """Get masked motion state as student observation input"""
        if not self.use_masked_priv_info:
            return torch.zeros(self.num_envs, 0, device=self.device)
        
        # Calculate motion state masking probability
        motion_masking_prob = self._get_motion_masking_prob()
        
        # Process linear velocity
        masked_lin_vel, lin_vel_mask_indicator = self._apply_motion_component_masking(
            self.base_lin_vel, motion_masking_prob
        )
        
        # Process root xyz
        masked_root_xyz, root_xyz_mask_indicator = self._apply_motion_component_masking(
            self.root_states[:, 0:3], motion_masking_prob
        )
        
        # Process root RPY (convert from quaternion)
        roll, pitch, yaw = euler_from_quaternion(self.root_states[:, 3:7])  # Convert quat to RPY
        root_rpy = torch.stack([roll, pitch, yaw], dim=-1)  # Stack into [N, 3] tensor
        masked_root_rpy, root_rpy_mask_indicator = self._apply_motion_component_masking(
            root_rpy, motion_masking_prob
        )
        
        # Combine motion state components and mask indicators
        masked_motion_state = torch.cat((
            masked_lin_vel,              # 3 dims (potentially masked)
            masked_root_xyz,             # 3 dims (potentially masked)  
            masked_root_rpy,             # 3 dims (potentially masked)
            lin_vel_mask_indicator,      # 1 dim (mask indicator for lin_vel)
            root_xyz_mask_indicator,     # 1 dim (mask indicator for root_xyz)
            root_rpy_mask_indicator,     # 1 dim (mask indicator for root_rpy)
        ), dim=-1)  # Total: 9 + 3 = 12 dims
        
        return masked_motion_state

    def _get_motion_masking_prob(self):
        """Calculate current motion state masking probability"""
        if hasattr(self, 'global_counter'):
            current_iter = self.global_counter // 24
        else:
            current_iter = 0
        
        progress = max(0, min(1, (current_iter - 2000) / max(1, 15000 - 2000)))
        return 0.3 + progress * (0.9 - 0.3)

    def _apply_motion_component_masking(self, component, masking_prob):
        """Apply masking to motion state component and return mask indicator"""
        # Generate mask for each environment
        mask = torch.rand(self.num_envs, device=self.device) > masking_prob
        
        # Apply masking
        masked_component = component.clone()
        masked_component[~mask] = 0.0  # Set masked values to 0
        
        # Generate mask indicator (single value per environment, 1=unmasked, 0=masked)
        mask_indicator = mask.float().unsqueeze(-1)  # (num_envs, 1)
        
        return masked_component, mask_indicator



    def _get_mimic_obs(self):
        """Override to use unified motion sampling for both privileged and future observations."""
        # Get unified motion data (SINGLE sampling call for all frames)
        motion_data = self._get_unified_motion_data()
        
        # Extract privileged motion data (first num_priv_steps)
        num_steps = motion_data['num_priv_steps']
        
        root_pos = motion_data['root_pos'][:, :num_steps]
        root_vel = motion_data['root_vel'][:, :num_steps]
        root_rot = motion_data['root_rot'][:, :num_steps]
        root_ang_vel = motion_data['root_ang_vel'][:, :num_steps]
        dof_pos = motion_data['dof_pos'][:, :num_steps]
        dof_vel = motion_data['dof_vel'][:, :num_steps]
        root_pos_delta_local = motion_data['root_pos_delta_local'][:, :num_steps]
        root_rot_delta_local = motion_data['root_rot_delta_local'][:, :num_steps]
        roll = motion_data['roll'][:, :num_steps]
        pitch = motion_data['pitch'][:, :num_steps]
        yaw = motion_data['yaw'][:, :num_steps]
        root_vel_local = motion_data['root_vel_local'][:, :num_steps]
        root_ang_vel_local = motion_data['root_ang_vel_local'][:, :num_steps]
        whole_key_body_pos = motion_data['whole_key_body_pos'][:, :num_steps]
        whole_key_body_pos_global = motion_data['whole_key_body_pos_global'][:, :num_steps]
        root_pos_distance_to_target = motion_data['root_pos_distance_to_target'][:, :num_steps]
        
        # teacher
        priv_mimic_obs_buf = torch.cat((
            root_pos, # 3 dims
            root_pos_distance_to_target, # 3 dims
            roll, pitch, yaw, # 3 dims
            root_vel_local, # 3 dims
            root_ang_vel_local, # 3 dims
            root_pos_delta_local, # 3 dims
            root_rot_delta_local, # 3 dims
            dof_pos, # num_dof dims
            whole_key_body_pos if not self.global_obs else whole_key_body_pos_global,
        ), dim=-1) # shape: (num_envs, num_steps, 21 + num_dof + num_key_bodies * 3)
        
        # student
        mimic_obs_buf = torch.cat((
            # root position: xy velocity + z position
            root_vel_local[..., :2], # 2 dims (xy velocity instead of xy position)
            root_pos[..., 2:3], # 1 dim (z position)
            # root rotation: roll/pitch + yaw angular velocity
            roll, pitch, # 2 dims (roll/pitch orientation)
            root_ang_vel_local[..., 2:3], # 1 dim (yaw angular velocity)
            dof_pos, # num_dof dims
        ), dim=-1)[:, self._tar_motion_steps_idx_in_teacher, :] # shape: (num_envs, 1, 6 + 2*num_dof)
            
        priv_mimic_obs = priv_mimic_obs_buf.reshape(self.num_envs, -1)
        mimic_obs = mimic_obs_buf.reshape(self.num_envs, -1)

        
        # Add future motion observations only if using student_future obs_type
        if self.obs_type == 'student_future':
            # Build future observations from the SAME motion_data (no additional sampling!)
            future_obs = self._build_future_obs_from_data(motion_data)
            
            # Apply curriculum masking to future observations
            masked_future_obs = self._apply_curriculum_masking(future_obs)
            
            return priv_mimic_obs, mimic_obs, masked_future_obs
        else:
            # Return original format for compatibility
            return priv_mimic_obs, mimic_obs

    def compute_observations(self):
        """Override to include future motion observations while maintaining compatibility."""
        # Get IMU observations (same as parent)
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(0*self.yaw, 0*self.yaw, self.yaw)
        
        # Get motion observations
        if self.obs_type == 'student_future':
            priv_mimic_obs, mimic_obs, future_obs = self._get_mimic_obs()
        else:
            priv_mimic_obs, mimic_obs = self._get_mimic_obs()
            future_obs = None
        
        # Proprioceptive observations (same as parent)
        proprio_obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,   # 3 dims
            imu_obs,    # 2 dims
            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
            self.reindex(self.dof_vel * self.obs_scales.dof_vel),
            self.reindex(self.action_history_buf[:, -1]),
        ), dim=-1)
        
        # Add noise if enabled (same as parent)
        if self.cfg.noise.add_noise and self.headless:
            noise_scale = min(self.total_env_steps_counter / (self.cfg.noise.noise_increasing_steps * 24), 1.)
            proprio_obs_buf += (2 * torch.rand_like(proprio_obs_buf) - 1) * self.noise_scale_vec * noise_scale
        elif self.cfg.noise.add_noise and not self.headless:
            proprio_obs_buf += (2 * torch.rand_like(proprio_obs_buf) - 1) * self.noise_scale_vec
        
        # Disable ankle dof velocity (same as parent)
        dof_vel_start_dim = 3 + 2 + self.dof_pos.shape[1]
        ankle_idx = [4, 5, 10, 11]
        proprio_obs_buf[:, [dof_vel_start_dim + i for i in ankle_idx]] = 0.
        
        # Private information for critic (same as parent)
        key_body_pos = self.rigid_body_states[:, self._key_body_ids, :3]
        key_body_pos = key_body_pos - self.root_states[:, None, :3]
        if not self.global_obs:
            key_body_pos = convert_to_local_root_body_pos(self.root_states[:, 3:7], key_body_pos)
        key_body_pos = key_body_pos.reshape(self.num_envs, -1)
        
        priv_info = torch.cat((
            self.base_lin_vel, # 3 dims
            self.root_states[:, 0:3], # 3 dims
            self.root_states[:, 3:7], # 4 dims
            key_body_pos, # num_bodies * 3 dims
            self.contact_forces[:, self.feet_indices, 2] > 5., # 2 dims
            self.mass_params_tensor,
            self.friction_coeffs_tensor,
            self.motor_strength[0] - 1, 
            self.motor_strength[1] - 1,
        ), dim=-1)
        
        # Current observation (for history) - same as parent
        obs_buf = torch.cat((
            mimic_obs,
            proprio_obs_buf,
        ), dim=-1)
        
        # Privileged observation - same as parent
        priv_obs_buf = torch.cat((
            priv_mimic_obs,
            proprio_obs_buf,
            priv_info,
        ), dim=-1)
        
        self.privileged_obs_buf = priv_obs_buf
        
        # Build final observation based on obs_type
        if self.obs_type == 'priv':
            self.obs_buf = priv_obs_buf
        elif self.obs_type == 'student_future':
            # Get masked motion state observations
            masked_motion_state_obs = self._get_masked_motion_state_obs()
            
            # Include history + future observations + masked motion state
            obs_components = [
                obs_buf, 
                self.obs_history_buf.view(self.num_envs, -1)
            ]
            
            if future_obs is not None:
                future_obs_flat = future_obs.view(self.num_envs, -1)
                obs_components.append(future_obs_flat)
            
            if self.use_masked_priv_info and masked_motion_state_obs.shape[1] > 0:
                obs_components.append(masked_motion_state_obs)
            
            self.obs_buf = torch.cat(obs_components, dim=-1)
        else:
            # Default student behavior (maintains full compatibility)
            self.obs_buf = torch.cat([obs_buf, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        
        # Update history buffers (same as parent) - using in-place operations to avoid memory leak
        if self.cfg.env.history_len > 0:
            # Find episodes that need to be reset
            reset_mask = (self.episode_length_buf <= 1)
            
            # For reset episodes, fill entire history with current observation
            if reset_mask.any():
                reset_indices = reset_mask.nonzero(as_tuple=False).squeeze(-1)
                self.privileged_obs_history_buf[reset_indices] = priv_obs_buf[reset_indices].unsqueeze(1).expand(
                    -1, self.cfg.env.history_len, -1
                )
            
            # For continuing episodes, shift history and add new observation
            continue_mask = ~reset_mask
            if continue_mask.any():
                continue_indices = continue_mask.nonzero(as_tuple=False).squeeze(-1)
                # Shift history left (remove oldest, move others)
                self.privileged_obs_history_buf[continue_indices, :-1] = self.privileged_obs_history_buf[continue_indices, 1:]
                # Add new observation at the end
                self.privileged_obs_history_buf[continue_indices, -1] = priv_obs_buf[continue_indices]
            
            if self.obs_type == 'priv':
                self.obs_history_buf[:] = self.privileged_obs_history_buf[:]
            else:
                # Use the same in-place update pattern for regular observations
                # For reset episodes, fill entire history with current observation
                if reset_mask.any():
                    reset_indices = reset_mask.nonzero(as_tuple=False).squeeze(-1)
                    self.obs_history_buf[reset_indices] = obs_buf[reset_indices].unsqueeze(1).expand(
                        -1, self.cfg.env.history_len, -1
                    )
                
                # For continuing episodes, shift history and add new observation
                if continue_mask.any():
                    continue_indices = continue_mask.nonzero(as_tuple=False).squeeze(-1)
                    # Shift history left (remove oldest, move others)
                    self.obs_history_buf[continue_indices, :-1] = self.obs_history_buf[continue_indices, 1:]
                    # Add new observation at the end
                    self.obs_history_buf[continue_indices, -1] = obs_buf[continue_indices]

    def update_performance_metrics(self, episode_info):
        """Update performance metrics for adaptive masking."""
        if self.obs_type != 'student_future':
            return
            
        if 'tracking_joint_dof' in episode_info:
            # Use joint tracking performance as the main metric
            self.recent_performance = episode_info['tracking_joint_dof'].mean().item()
        elif 'total_reward' in episode_info:
            # Fallback to total reward
            self.recent_performance = episode_info['total_reward'].mean().item()
    
    def get_masking_info(self):
        """Get current masking information for logging."""
        if self.obs_type != 'student_future':
            return {}
        
        # Calculate curriculum progress
        current_iter = getattr(self, 'global_counter', 0) // 24 if hasattr(self, 'global_counter') else 0
        curriculum_progress = max(0, min(1, (current_iter - self.masking_start_iteration) / 
                                         max(1, self.masking_end_iteration - self.masking_start_iteration)))
        
        # Calculate temporal dropout probability
        dropout_progress = max(0, min(1, current_iter / max(1, self.masking_end_iteration)))
        current_temporal_dropout_prob = self.temporal_dropout_start_prob + dropout_progress * \
                                      (self.temporal_dropout_end_prob - self.temporal_dropout_start_prob)
        
        # Calculate per-timestep masking ratios if future_mask exists
        per_timestep_ratios = []
        if hasattr(self, 'future_mask') and self.future_mask.numel() > 0:
            # Calculate masking ratio for each future timestep
            for i in range(self.future_mask.shape[1]):
                ratio = self.future_mask[:, i].float().mean().item()
                per_timestep_ratios.append(ratio)
        
        info = {
            # Core masking parameters
            'current_masking_prob': self.current_masking_prob,
            'curriculum_progress': curriculum_progress,
            'current_temporal_dropout_prob': current_temporal_dropout_prob,
            
            # Performance metrics
            'recent_performance': self.recent_performance,
            'performance_threshold': self.performance_threshold,
            
            # Masking ratios
            'future_mask_ratio': self.future_mask.float().mean().item() if hasattr(self, 'future_mask') and self.future_mask.numel() > 0 else 0.0,
            'per_timestep_mask_ratios': per_timestep_ratios,
            
            # Training progress
            'current_iteration': current_iter,
            'masking_start_iteration': self.masking_start_iteration,
            'masking_end_iteration': self.masking_end_iteration,
            'global_counter': getattr(self, 'global_counter', 0),
            
            # Configuration flags
            'curriculum_masking_enabled': self.curriculum_masking,
            'temporal_dropout_enabled': self.temporal_dropout,
            'adaptive_masking_enabled': self.adaptive_masking,
            'evaluation_mode': getattr(self, 'evaluation_mode', False),
            'force_full_masking': getattr(self, 'force_full_masking', False),
            
            # Masking schedule parameters
            'min_masking_prob': self.min_masking_prob,
            'max_masking_prob': self.max_masking_prob,
            'temporal_dropout_start_prob': self.temporal_dropout_start_prob,
            'temporal_dropout_end_prob': self.temporal_dropout_end_prob,
        }
        
        return info

    def log_masking_info(self, iteration):
        """Log masking information to wandb if available."""
        if self.obs_type != 'student_future':
            return
        
        try:
            import wandb
            masking_info = self.get_masking_info()
            
            # Create a dictionary with wandb-friendly keys
            log_dict = {}
            for key, value in masking_info.items():
                if isinstance(value, (int, float)):
                    log_dict[f"masking/{key}"] = value
                elif isinstance(value, list) and len(value) > 0:
                    # Log per-timestep ratios as a histogram or mean
                    if key == 'per_timestep_mask_ratios':
                        log_dict[f"masking/avg_per_timestep_mask_ratio"] = sum(value) / len(value)
                        # Log each timestep ratio separately
                        for i, ratio in enumerate(value):
                            log_dict[f"masking/timestep_{i}_mask_ratio"] = ratio
                elif isinstance(value, bool):
                    log_dict[f"masking/{key}"] = int(value)
            
            # Add iteration for proper x-axis alignment
            log_dict["iteration"] = iteration
            
            # Only log if wandb is initialized
            if wandb.run is not None:
                wandb.log(log_dict)
            else:
                print(f"[Masking Info] Iteration {iteration}: {log_dict}")
                
        except ImportError:
            # wandb not available, just print
            masking_info = self.get_masking_info()
            print(f"[Masking Info] Iteration {iteration}: {masking_info}")
        except Exception as e:
            print(f"Error logging masking info: {e}")