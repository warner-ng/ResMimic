from legged_gym.envs.g1.g1_mimic_future import G1MimicFuture
# from legged_gym.envs.g1.g1_mimic_full_future import G1MimicFullFuture
from .g1_hoi_config import G1HOICfg

import numpy as np
from isaacgym.torch_utils import *
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil

import trimesh
import torch
import os
from tqdm import tqdm
from legged_gym import LEGGED_GYM_ROOT_DIR
from pose.utils.motion_lib_hoi import MotionLibHOI
from pose.utils import torch_utils
from legged_gym.envs.base.humanoid_char import convert_to_global_root_body_pos, convert_to_local_root_body_pos
from legged_gym.envs.base.legged_robot import euler_from_quaternion


class G1HOI(G1MimicFuture):
    def __init__(self, cfg: G1HOICfg, sim_params, physics_engine, sim_device, headless):
        self.num_actors = cfg.env.num_actors
        self.all_actor_ids = torch.arange(self.num_actors * cfg.env.num_envs, device=sim_device, dtype=torch.int32).reshape(cfg.env.num_envs, self.num_actors)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._ensure_runtime_pair_leveling_ready()
    
    def _load_motions(self):
        self._motion_lib = MotionLibHOI(motion_file=self.cfg.motion.motion_file, 
                                     object_motion_file=self.cfg.motion.object_motion_file, device=self.device,
                                    sample_ratio=self.cfg.motion.sample_ratio,
                                    motion_decompose=self.cfg.motion.motion_decompose,
                                    motion_smooth=self.cfg.motion.motion_smooth,
                                    object_rot_offset_deg=getattr(self.cfg.env, "object_motion_rot_offset_deg", [0.0, 0.0, 0.0]))
        return
    # these are written by warner, to adjust data reconstructed from cari4d
    def _ensure_runtime_pair_leveling_ready(self):
        if not hasattr(self, "_pair_level_rot"):
            self._pair_level_rot = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float)
            self._pair_level_rot[:, 3] = 1.0
        if not hasattr(self, "_pair_level_trans"):
            self._pair_level_trans = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        if not hasattr(self, "_motion_foot_ids"):
            body_link_list = getattr(self._motion_lib, "_body_link_list", []) or []
            foot_ids = [i for i, name in enumerate(body_link_list) if any(key in name.lower() for key in ("ankle", "toe", "foot"))]
            if len(foot_ids) == 0:
                raise ValueError("No ankle/toe/foot links found in motion body link list for runtime pair leveling.")
            self._motion_foot_ids = torch.tensor(foot_ids, device=self.device, dtype=torch.long)

    def _compute_runtime_pair_level_transform(self, root_pos, root_rot, body_pos, object_root_pos, object_root_rot):
        self._ensure_runtime_pair_leveling_ready()
        num_envs = root_pos.shape[0]
        rot_quat = torch.zeros((num_envs, 4), device=self.device, dtype=root_rot.dtype)
        rot_quat[:, 3] = 1.0
        trans = torch.zeros((num_envs, 3), device=self.device, dtype=root_pos.dtype)
        if not getattr(self.cfg.env, "enable_runtime_pair_leveling", False):
            return rot_quat, trans

        foot_local = body_pos[:, self._motion_foot_ids, :]
        foot_rot = root_rot.unsqueeze(1).expand(-1, foot_local.shape[1], -1).reshape(-1, 4)
        foot_local_flat = foot_local.reshape(-1, 3)
        world_feet = quat_rotate(foot_rot, foot_local_flat).view(num_envs, foot_local.shape[1], 3) + root_pos.unsqueeze(1)
        human_support = world_feet[torch.arange(num_envs, device=self.device), torch.argmin(world_feet[:, :, 2], dim=1)]

        obj_local = self.object_points.unsqueeze(0).expand(num_envs, -1, -1)
        obj_rot_expand = object_root_rot.unsqueeze(1).expand(-1, obj_local.shape[1], -1).reshape(-1, 4)
        obj_local_flat = obj_local.reshape(-1, 3)
        world_obj = quat_rotate(obj_rot_expand, obj_local_flat).view(num_envs, obj_local.shape[1], 3) + object_root_pos.unsqueeze(1)
        object_support = world_obj[torch.arange(num_envs, device=self.device), torch.argmin(world_obj[:, :, 2], dim=1)]

        d = human_support - object_support
        h = d.clone()
        h[:, 2] = 0.0
        h_norm = torch.norm(h, dim=1)
        d_norm = torch.norm(d, dim=1)
        valid = (h_norm > 1e-6) & (d_norm > 1e-6) & (torch.abs(d[:, 2]) > 1e-6)
        if valid.any():
            axis = torch.cross(d[valid], h[valid], dim=1)
            axis_norm = torch.norm(axis, dim=1)
            valid_axis = axis_norm > 1e-6
            if valid_axis.any():
                cos = torch.sum(d[valid][valid_axis] * h[valid][valid_axis], dim=1) / (d_norm[valid][valid_axis] * h_norm[valid][valid_axis])
                cos = torch.clamp(cos, -1.0, 1.0)
                angle = torch.acos(cos)
                rot_quat_valid = torch_utils.quat_from_angle_axis(angle, axis[valid_axis] / axis_norm[valid_axis].unsqueeze(1))
                rot_quat[valid.nonzero(as_tuple=False).flatten()[valid_axis]] = rot_quat_valid

        midpoint = 0.5 * (human_support + object_support)
        midpoint_rot = quat_rotate(rot_quat, midpoint)
        trans = midpoint - midpoint_rot

        leveled_human_support = quat_rotate(rot_quat, human_support) + trans
        leveled_object_support = quat_rotate(rot_quat, object_support) + trans
        target_z = getattr(self.cfg.env, "runtime_pair_level_target_z", 0.0)
        avg_support_z = 0.5 * (leveled_human_support[:, 2] + leveled_object_support[:, 2])
        trans[:, 2] += target_z - avg_support_z
        return rot_quat, trans

    def _apply_runtime_pair_level_transform(self, env_ids, root_pos, root_rot, object_root_pos, object_root_rot):
        rot_quat = self._pair_level_rot[env_ids]
        trans = self._pair_level_trans[env_ids]
        root_pos = quat_rotate(rot_quat, root_pos) + trans
        object_root_pos = quat_rotate(rot_quat, object_root_pos) + trans
        root_rot = quat_mul(rot_quat, root_rot)
        object_root_rot = quat_mul(rot_quat, object_root_rot)
        return root_pos, root_rot, object_root_pos, object_root_rot

    def _apply_human_root_rot_offset(self, root_rot):
        offset_deg = getattr(self.cfg.env, "human_root_rot_offset_deg", [0.0, 0.0, 0.0])
        return self._apply_root_rot_offset(root_rot, offset_deg)

    def _apply_object_root_rot_offset(self, root_rot):
        offset_deg = getattr(self.cfg.env, "object_root_rot_offset_deg", [0.0, 0.0, 0.0])
        root_rot = self._apply_root_rot_offset(root_rot, offset_deg)
        local_offset_deg = getattr(self.cfg.env, "object_root_local_rot_offset_deg", [0.0, 0.0, 0.0])
        return self._apply_root_local_rot_offset(root_rot, local_offset_deg)

    def _apply_root_rot_offset(self, root_rot, offset_deg):
        if all(abs(v) < 1e-8 for v in offset_deg):
            return root_rot
        offset_quat = self._root_rot_offset_quat(root_rot, offset_deg)
        return quat_mul(offset_quat, root_rot)

    def _apply_root_local_rot_offset(self, root_rot, offset_deg):
        if all(abs(v) < 1e-8 for v in offset_deg):
            return root_rot
        offset_quat = self._root_rot_offset_quat(root_rot, offset_deg)
        return quat_mul(root_rot, offset_quat)

    def _root_rot_offset_quat(self, root_rot, offset_deg):
        offset = torch.tensor(offset_deg, device=root_rot.device, dtype=root_rot.dtype) * (torch.pi / 180.0)
        n = root_rot.shape[0]
        axes = torch.eye(3, device=root_rot.device, dtype=root_rot.dtype)
        qx = torch_utils.quat_from_angle_axis(
            torch.full((n,), offset[0], device=root_rot.device, dtype=root_rot.dtype),
            axes[0].expand(n, -1),
        )
        qy = torch_utils.quat_from_angle_axis(
            torch.full((n,), offset[1], device=root_rot.device, dtype=root_rot.dtype),
            axes[1].expand(n, -1),
        )
        qz = torch_utils.quat_from_angle_axis(
            torch.full((n,), offset[2], device=root_rot.device, dtype=root_rot.dtype),
            axes[2].expand(n, -1),
        )
        return quat_mul(qx, quat_mul(qy, qz))

    def _apply_object_root_pos_offset(self, object_root_pos, object_root_rot):
        offset = getattr(self.cfg.env, "object_root_pos_offset", [0.0, 0.0, 0.0])
        if all(abs(v) < 1e-8 for v in offset):
            return object_root_pos
        local_offset = torch.tensor(offset, device=object_root_pos.device, dtype=object_root_pos.dtype).unsqueeze(0)
        return object_root_pos + quat_rotate(object_root_rot, local_offset.expand(object_root_pos.shape[0], -1))

    def _apply_pair_root_pos_offset(self, root_pos, object_root_pos):
        offset = getattr(self.cfg.env, "motion_global_pos_offset", [0.0, 0.0, 0.0])
        if all(abs(v) < 1e-8 for v in offset):
            return root_pos, object_root_pos
        trans = torch.tensor(offset, device=root_pos.device, dtype=root_pos.dtype).unsqueeze(0)
        return root_pos + trans, object_root_pos + trans

    def _reset_ref_motion(self, env_ids, motion_ids=None):
        n = len(env_ids)
        if motion_ids is None:
            motion_ids = self._motion_lib.sample_motions(n, motion_difficulty=self.motion_difficulty)
        
        if self._rand_reset:
            motion_times = self._motion_lib.sample_time(motion_ids)
        else:
            motion_times = torch.zeros(motion_ids.shape, device=self.device, dtype=torch.float)
        
        self._motion_ids[env_ids] = motion_ids
        self._motion_time_offsets[env_ids] = motion_times
        
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, root_pos_delta_local, root_rot_delta_local, object_root_pos, object_root_rot = self._motion_lib.calc_hoi_motion_frame(motion_ids, motion_times)
        root_rot = self._apply_human_root_rot_offset(root_rot)
        object_root_rot = self._apply_object_root_rot_offset(object_root_rot)
        object_root_pos = self._apply_object_root_pos_offset(object_root_pos, object_root_rot)
        root_pos, object_root_pos = self._apply_pair_root_pos_offset(root_pos, object_root_pos)
        pair_rot, pair_trans = self._compute_runtime_pair_level_transform(root_pos, root_rot, body_pos, object_root_pos, object_root_rot)
        self._pair_level_rot[env_ids] = pair_rot
        self._pair_level_trans[env_ids] = pair_trans
        root_pos, root_rot, object_root_pos, object_root_rot = self._apply_runtime_pair_level_transform(env_ids, root_pos, root_rot, object_root_pos, object_root_rot)
        if not hasattr(self, "_printed_init_root_euler"):
            self._printed_init_root_euler = False
        if not self._printed_init_root_euler and root_rot.shape[0] > 0:
            roll, pitch, yaw = euler_from_quaternion(root_rot[:1])
            print(
                f"[G1HOI] init leveled human root euler deg: "
                f"roll={torch.rad2deg(roll[0]).item():.2f}, "
                f"pitch={torch.rad2deg(pitch[0]).item():.2f}, "
                f"yaw={torch.rad2deg(yaw[0]).item():.2f}"
            )
            self._printed_init_root_euler = True
        root_pos[:, 2] += self.cfg.motion.height_offset
        self._ref_root_pos[env_ids] = root_pos
        self._ref_root_rot[env_ids] = root_rot
        self._ref_root_vel[env_ids] = root_vel
        self._ref_root_ang_vel[env_ids] = root_ang_vel
        self._ref_dof_pos[env_ids] = dof_pos
        self._ref_dof_vel[env_ids] = dof_vel
        self._ref_root_pos_delta_local[env_ids] = root_pos_delta_local
        self._ref_root_rot_delta_local[env_ids] = root_rot_delta_local
        self._ref_body_pos[env_ids] = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=body_pos)
        
        self._ref_object_root_pos[env_ids] = object_root_pos
        self._ref_object_root_rot[env_ids] = object_root_rot
    
    
    def _update_ref_motion(self):
        motion_ids = self._motion_ids
        motion_times = self._get_motion_times()
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, root_pos_delta_local, root_rot_delta_local, object_root_pos, object_root_rot = self._motion_lib.calc_hoi_motion_frame(motion_ids, motion_times)
        root_rot = self._apply_human_root_rot_offset(root_rot)
        object_root_rot = self._apply_object_root_rot_offset(object_root_rot)
        object_root_pos = self._apply_object_root_pos_offset(object_root_pos, object_root_rot)
        root_pos, object_root_pos = self._apply_pair_root_pos_offset(root_pos, object_root_pos)
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        root_pos, root_rot, object_root_pos, object_root_rot = self._apply_runtime_pair_level_transform(env_ids, root_pos, root_rot, object_root_pos, object_root_rot)
        root_pos[:, 2] += self.cfg.motion.height_offset
        root_pos[:, :2] += self.episode_init_origin[:, :2]
        
        self._ref_root_pos[:] = root_pos
        self._ref_root_rot[:] = root_rot
        self._ref_root_vel[:] = root_vel
        self._ref_root_ang_vel[:] = root_ang_vel
        self._ref_dof_pos[:] = dof_pos
        self._ref_dof_vel[:] = dof_vel
        self._ref_root_pos_delta_local[:] = root_pos_delta_local
        self._ref_root_rot_delta_local[:] = root_rot_delta_local
        self._ref_body_pos[:] = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=body_pos)

        self._ref_object_root_pos[:] = object_root_pos
        self._ref_object_root_rot[:] = object_root_rot

    def _reset_root_states(self, env_ids, root_vel=None, root_quat=None, root_pos=None, root_ang_vel=None):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position        
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if self.cfg.env.randomize_start_pos:
                rand_pos = torch_rand_float(-0.3, 0.3, (len(env_ids), 2), device=self.device)
                self.root_states[env_ids, :2] += rand_pos # xy position within 1m of the center
                self.episode_init_origin[env_ids, :2] = self.env_origins[env_ids, :2] + rand_pos
            if self.cfg.env.randomize_start_yaw:
                rand_yaw = torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                quat = quat_from_euler_xyz(0*rand_yaw, 0*rand_yaw, rand_yaw) 
                self.root_states[env_ids, 3:7] = quat[:, :]
            
            if root_vel is not None:
                self.root_states[env_ids, 7:10] = root_vel[env_ids, :]
            if root_quat is not None:
                self.root_states[env_ids, 3:7] = root_quat[env_ids, :]
            
            if root_pos is not None:
                self.root_states[env_ids, 2] = root_pos[env_ids, 2] + self.cfg.env.human_root_z_bias
                self.root_states[env_ids, :2] += root_pos[env_ids, :2]
            if root_ang_vel is not None:
                self.root_states[env_ids, 10:13] = root_ang_vel[env_ids, :]

            self.object_root_states[env_ids, :] =  torch.zeros_like(self.object_root_states[env_ids, :])
            
            self.object_root_states[env_ids, :3] = self._ref_object_root_pos[env_ids, :3] + torch.tensor([0.0, 0.0, self.cfg.env.object_root_z_bias], device=self.device)
            self.object_root_states[env_ids, 3:7] = self._ref_object_root_rot[env_ids, :]
            self.object_root_states[env_ids, :3] += self.env_origins[env_ids]

            # Optional bike test placement override (default: disabled).
            # Keep this path for quick debugging but do not apply in normal runs.
            object_urdf_file = (self.cfg.env.object_urdf_file or "").lower()
            is_bike_task = ("bike" in object_urdf_file) or ("bicycle" in object_urdf_file)
            if is_bike_task and getattr(self.cfg.env, "enable_bike_test_placement", False):
                # Fixed placement: in front of robot (based on robot yaw), facing the robot
                forward_local = torch.tensor([1.0, 0.0, 0.0], device=self.device)
                forward_world = quat_rotate(self.root_states[env_ids, 3:7], forward_local.expand(len(env_ids), 3))
                fixed_offset = forward_world * 1.2
                fixed_offset[:, 2] = 0.5
                self.object_root_states[env_ids, :3] = self.env_origins[env_ids] + fixed_offset

                _, _, base_yaw = euler_from_quaternion(self.root_states[env_ids, 3:7])
                target_yaw = base_yaw + torch.pi
                fixed_quat = quat_from_euler_xyz(0 * target_yaw, 0 * target_yaw, target_yaw)
                self.object_root_states[env_ids, 3:7] = fixed_quat


            def quat_rotate_xyzw(q, v):
                # q: (...,4) [x,y,z,w], v: (...,3)
                qvec = q[..., :3]
                w = q[..., 3:4]
                uv  = torch.cross(qvec, v, dim=-1)
                uuv = torch.cross(qvec, uv, dim=-1)
                return v + 2.0 * (w * uv + uuv)

            if self.cfg.domain_rand.randomize_object_pos:
                rand_pos = torch_rand_float(-0.02, 0.02, (len(env_ids), 2), device=self.device)
                rand_z_angles = torch_rand_float(-5.0, 5.0, (len(env_ids), 1), device=self.device) * torch.pi / 180.0  # Convert to radians
                rand_z_quats = torch.zeros((len(env_ids), 4), device=self.device)
                rand_z_quats[:, 2] = torch.sin(rand_z_angles[:, 0] / 2)  # z component
                rand_z_quats[:, 3] = torch.cos(rand_z_angles[:, 0] / 2)  # w component

                local_offset = torch.zeros((len(env_ids), 3), device=self.device)
                local_offset[:, 0] = rand_pos[:, 0]  # x offset
                local_offset[:, 1] = rand_pos[:, 1]  # y offset
                world_offset = quat_rotate_xyzw(torch.tensor([[0.0, 0.0, 0.3671609, 0.93015744]], device=self.device), local_offset)
                self.object_root_states[env_ids, :3] += world_offset
                
                original_quat = self._ref_object_root_rot[env_ids, :4]  # [x,y,z,w]
                combined_quat = quat_mul(rand_z_quats, original_quat)
                self.object_root_states[env_ids, 3:7] = combined_quat

            if self.num_actors == 3:
                self.support_root_states[env_ids, 0:7] = torch.tensor([0.0831, 0.6450, 0.5073, 0.0, 0.0, 0.3671609, 0.93015744], device=self.device)
                self.support_root_states[env_ids, 0:3] += self.env_origins[env_ids]
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        
        env_ids_int32 = self.all_actor_ids[env_ids].flatten().to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.all_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    
    def _reset_dofs(self, env_ids, dof_pos, dof_vel):
        self.dof_pos[env_ids] = dof_pos[env_ids] * torch_rand_float(0.8, 1.2, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = dof_vel[env_ids]

        env_ids_int32 = self.all_actor_ids[env_ids, 0].to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    # places for visualising objects and keypoints in issacgym
    def _load_object_asset(self):
        object_asset_root = self.cfg.env.object_asset_root
        object_urdf_file = self.cfg.env.object_urdf_file
        object_obj_file = self.cfg.env.object_obj_file
        object_scale = float(getattr(self.cfg.env, "object_scale", 1.0))
        max_convex_hulls = 64
        density = self.cfg.asset.object_density
    
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01

        asset_options.density = density
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.max_convex_hulls = max_convex_hulls
        asset_options.vhacd_params.max_num_vertices_per_ch = 64
        asset_options.vhacd_params.resolution = 300000

        self.target_asset = self.gym.load_asset(self.sim, object_asset_root, object_urdf_file, asset_options)

        mesh_obj = trimesh.load(f"{object_asset_root}/{object_obj_file}", force='mesh')
        obj_verts = mesh_obj.vertices
        center = np.mean(obj_verts, 0)
        object_points, object_faces = trimesh.sample.sample_surface_even(mesh_obj, count=1024, seed=2024)

        object_points = to_torch((object_points - center) * object_scale)

        while object_points.shape[0] < 1024:
            object_points = torch.cat([object_points, object_points[:1024 - object_points.shape[0]]], dim=0)
        
        self.object_points = object_points

        if self.num_actors == 3:
            self.plate_size_x = 0.4   # meters
            self.plate_size_y = 0.3
            self.plate_thickness = 0.01
            asset_opts = gymapi.AssetOptions()
            asset_opts.fix_base_link = True       # <-- makes it static
            # asset_opts.disable_gravity = True
            self.support_asset = self.gym.create_box(self.sim, self.plate_size_x, self.plate_size_y, self.plate_thickness, asset_opts)


    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        self._load_object_asset()
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        self.torso_idx = self.gym.find_asset_rigid_body_index(robot_asset, self.cfg.asset.torso_name)
        self.chest_idx = self.gym.find_asset_rigid_body_index(robot_asset, self.cfg.asset.chest_name)

        for s in self.cfg.asset.feet_bodies:
            feet_idx = self.gym.find_asset_rigid_body_index(robot_asset, s)
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
            self.gym.create_asset_force_sensor(robot_asset, feet_idx, sensor_pose)
        
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        spacing = self.cfg.env.env_spacing
        if self.cfg.terrain.mesh_type == "plane":
            env_lower = gymapi.Vec3(-spacing, -spacing, 0.)
            env_upper = gymapi.Vec3(spacing, spacing, spacing)
        else:
            env_lower = gymapi.Vec3(0., 0., 0.)
            env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.object_handles = []
        self.cam_handles = []
        self.cam_tensors = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        
        print("Creating env...")
        for i in tqdm(range(self.num_envs)):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            if self.cfg.env.randomize_start_pos:
                pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            if self.cfg.env.randomize_start_yaw:
                rand_yaw_quat = gymapi.Quat.from_euler_zyx(0., 0., self.cfg.env.rand_yaw_range*np.random.uniform(-1, 1))
                start_pose.r = rand_yaw_quat
            start_pose.p = gymapi.Vec3(*(pos + self.base_init_state[:3]))

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "anymal", i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)
            
            # self.attach_camera(i, env_handle, anymal_handle)

            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device).to(torch.float)

            # create objects in the environment
            default_pose = gymapi.Transform()
            
            object_handle = self.gym.create_actor(env_handle, self.target_asset, default_pose, "object", i, 0, 0)

            props = self.gym.get_actor_rigid_shape_properties(env_handle, object_handle)
            props = self._process_object_rigid_shape_props(props, i)
            for p_idx in range(len(props)):
                props[p_idx].restitution = 0.2
                props[p_idx].rolling_friction = 0.01
                props[p_idx].torsion_friction = 0.8
                props[p_idx].contact_offset = 0.002  # a couple mm
            self.gym.set_actor_rigid_shape_properties(env_handle, object_handle, props)

            props = self.gym.get_actor_rigid_body_properties(env_handle, object_handle)
            if "box" in self.cfg.env.object_urdf_file:
                for p_idx in range(len(props)):
                    props[p_idx].mass = self.cfg.asset.object_mass
                props = self._process_object_rigid_body_props(props, i)
                assert self.gym.set_actor_rigid_body_properties(env_handle, object_handle, props)
            props = self.gym.get_actor_rigid_body_properties(env_handle, object_handle)

            self.object_handles.append(object_handle)
            object_scale = float(getattr(self.cfg.env, "object_scale", 1.0))
            if self.cfg.domain_rand.randomize_object_scale:
                rng_scale = self.cfg.domain_rand.object_scale_range
                rand_scale = np.random.uniform(rng_scale[0], rng_scale[1], size=(1, ))
                self.gym.set_actor_scale(env_handle, object_handle, rand_scale * object_scale)
            else:
                self.gym.set_actor_scale(env_handle, object_handle, object_scale)

            if self.num_actors == 3:
                plate_pose = gymapi.Transform()
                plate_handle = self.gym.create_actor(env_handle, self.support_asset, plate_pose, "support_plate", i, 0, 0)

                # increase friction so your box doesn't slide off
                shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, plate_handle)
                for s in shape_props:
                    s.restitution = 0.0
                    s.friction = 1.0
                    s.rolling_friction = 0.01
                    s.torsion_friction = 1.0
                self.gym.set_actor_rigid_shape_properties(env_handle, plate_handle, shape_props)
            
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)
        else:
            self.friction_coeffs_tensor = torch.ones(self.num_envs, 1, device=self.device, requires_grad=False)
        
        self.body_names = body_names
        self._get_body_indices()

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        
        
        waist_names = self.cfg.asset.waist_name
        self.waist_indices = torch.zeros(len(waist_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(waist_names)):
            self.waist_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], waist_names[j])
        
        hand_names = self.cfg.asset.hand_name
        self.hand_indices = torch.zeros(len(hand_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(hand_names)):
            self.hand_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], hand_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
        
        if self.cfg.env.record_video:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 720*2
            camera_props.height = 480*2
            self._rendering_camera_handles = []
            for i in range(self.num_envs):
                cam_pos = np.array([2, 0, 0.3])
                camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
                self._rendering_camera_handles.append(camera_handle)
                self.gym.set_camera_location(camera_handle, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*0*cam_pos))


    def _init_buffers(self):
        super()._init_buffers()
        self.object_root_states = self.all_root_states.view(self.num_envs, -1, 13)[:, 1, :]
        if self.num_actors == 3:
            self.support_root_states = self.all_root_states.view(self.num_envs, -1, 13)[:, 2, :]
    
        self._w_prev = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        self._update_ref_motion()

        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        
        if self.cfg.domain_rand.push_end_effector and (self.common_step_counter % self.cfg.domain_rand.push_end_effector_interval == 0):
            self._push_end_effector()
        else:
            self.forces = torch.zeros_like(self.forces)
            
        for i in range(len(self.eval_functions)):
            name = self.eval_names[i]
            error = self.eval_functions[i]()
            # running mean
            self.episode_means[name] += (-self.episode_means[name] + error) / (self.episode_length_buf + 1.0)

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        if self.common_step_counter == 0:
            self.episode_length_buf[:] = 0
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt

        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        self.episode_length[env_ids] = self.episode_length_buf[env_ids].float()
        self._w_prev[env_ids] = 0.

        self.reset_idx(env_ids)

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_root_pos[:] = self.root_states[:, 0:3]
        self.last_root_rot[:] = self.root_states[:, 3:7]

        if self.cfg.rewards.regularization_scale_curriculum:
            if torch.mean(self.episode_length.float()).item()> 420.:
                self.cfg.rewards.regularization_scale *= (1. + self.cfg.rewards.regularization_scale_gamma)
            elif torch.mean(self.episode_length.float()).item() < 50.:
                self.cfg.rewards.regularization_scale *= (1. - self.cfg.rewards.regularization_scale_gamma)
            self.cfg.rewards.regularization_scale = max(min(self.cfg.rewards.regularization_scale, self.cfg.rewards.regularization_scale_range[1]), self.cfg.rewards.regularization_scale_range[0])

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.draw_key_bodies_actual()
            self.draw_key_bodies_motion()
            self.draw_object_points_actual()
            self.draw_object_points_motion()


    def step(self, actions):
        actions = self.reindex(actions)
        actions.to(self.device)
        action_tensor = actions.clone()
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), action_tensor[:, None, :].clone()], dim=1)
        
        if self.cfg.domain_rand.action_delay:
            # Curriculum for action delay - linear increase from 0 to 0.5 probability
            start_step = 5000 * 24  # Starting step for curriculum
            target_step = 20000 * 24  # Target step where probability reaches 0.5
            
            if self.total_env_steps_counter <= start_step:
                delay_prob = 0.0
            elif self.total_env_steps_counter >= target_step:
                delay_prob = 0.5
            else:
                # Linear interpolation between start and target steps
                delay_prob = 0.5 * (self.total_env_steps_counter - start_step) / (target_step - start_step)
            
            # Directly sample delay from [0,1] with probability [1-delay_prob, delay_prob]
            if torch.rand(1, device=self.device) < delay_prob:
                self.delay = torch.tensor(1.0, device=self.device, dtype=torch.float)
            else:
                self.delay = torch.tensor(0.0, device=self.device, dtype=torch.float)
                
            indices = -self.delay - 1
            action_tensor = self.action_history_buf[:, indices.long()]

        self.global_counter += 1
        self.total_env_steps_counter += 1
        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        self.actions = torch.clip(action_tensor, -clip_actions, clip_actions).to(self.device)
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras



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

            if self.cfg.env.nonblind:
                obs_components.append(self.object_root_states[:, :7]) # TODO: add noise?
            
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


    def check_termination(self):
        if getattr(self.cfg.env, "disable_termination_for_debug", False):
            if self.cfg.rewards.termination_when_object_far:
                self._update_object_point_cloud_dist()
            self.reset_buf[:] = 0
            self.time_out_buf[:] = 0
            return

        contact_force_termination = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf = contact_force_termination.clone()
        
        # height_cutoff = self.root_states[:, 2] < self.cfg.rewards.termination_height
        root_height_diff = torch.abs(self.root_states[:, 2] - self._ref_root_pos[:, 2])
        height_cutoff = root_height_diff > self.cfg.rewards.root_height_diff_threshold

        roll_cut = torch.abs(self.roll) > self.cfg.rewards.termination_roll
        pitch_cut = torch.abs(self.pitch) > self.cfg.rewards.termination_pitch
        self.reset_buf |= roll_cut
        self.reset_buf |= pitch_cut
        motion_end = self.episode_length_buf * self.dt >= self._motion_lib.get_motion_length(self._motion_ids)
        self.reset_buf |= height_cutoff

        # check object far termination
        if self.cfg.rewards.termination_when_object_far:
            self._update_object_point_cloud_dist()
            reset_buf_object_far = self.object_point_cloud_dist > self.cfg.rewards.termination_object_far_threshold
            self.reset_buf |= reset_buf_object_far
        
        # check motion end
        if self.viewer is None:
            self.reset_buf |= motion_end
        
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        if self.viewer is None:
            self.time_out_buf |= motion_end
        
        self.reset_buf |= self.time_out_buf
        
        vel_too_large = torch.norm(self.root_states[:, 7:10], dim=-1) > 5.
        self.reset_buf |= vel_too_large
        
        if self._pose_termination:
            body_pos = self.rigid_body_states[:, self._key_body_ids, 0:3] - self.rigid_body_states[:, 0:1, 0:3]
            tar_body_pos = self._ref_body_pos[:, self._key_body_ids] - self._ref_root_pos[:, None, :] 
            
            if not self.global_obs:
                body_pos = convert_to_local_root_body_pos(self.root_states[:, 3:7], body_pos)
                tar_body_pos = convert_to_local_root_body_pos(self._ref_root_rot, tar_body_pos)
            
            body_pos_diff = tar_body_pos - body_pos # (envs, bodies, 3)
            body_pos_dist = torch.sum(body_pos_diff * body_pos_diff, dim=-1) # (envs, bodies)

            body_pos_dist = torch.max(body_pos_dist, dim=-1)[0] # (envs)
            
            # if lose tracking for 50 frames continuously, reset (corresponds to 1 second)
            # lose_tracking = body_pos_dist > self.motion_termination_dist[self._motion_ids] ** 2
            # self.deviate_tracking_frames[lose_tracking] += 1
            # self.deviate_tracking_frames[~lose_tracking] = 0
            # pose_fail = self.deviate_tracking_frames >= self.cfg.motion.reset_consec_frames # 50 frames = 1 second
            
            # use a fixed pose termination distance
            pose_fail = body_pos_dist > self._pose_termination_dist ** 2
            
            # use an adaptive pose termination distance
            # pose_fail = body_pos_dist > self.motion_termination_dist[self._motion_ids] ** 2
            
            if self._track_root:
                root_pos_diff = self._ref_root_pos[:, 0:2] - self.root_states[:, 0:2]
                root_pos_dist = torch.sum(root_pos_diff * root_pos_diff, dim=-1)
                root_pos_fail = root_pos_dist > self._root_tracking_termination_dist ** 2
                root_pos_fail = root_pos_fail.squeeze(-1)
                pose_fail |= root_pos_fail
            self.reset_buf |= pose_fail
        
        first_step = self.episode_length_buf == 0

        self.reset_buf[first_step] = 0 # Do not reset on first step
        
        # print reset reason
        if self.viewer is not None and self.reset_buf.any():
            reset_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            for id in reset_ids:
                reset_reason = ""
                if contact_force_termination[id]:
                    reset_reason = "contact force"
                elif height_cutoff[id]:
                    reset_reason = "height cutoff"
                    print("height diff: ", root_height_diff[id])
                elif roll_cut[id]:
                    reset_reason = "roll limit"
                    print("roll diff: ", self.roll[id])
                elif pitch_cut[id]:
                    reset_reason = "pitch limit"
                    print("pitch diff: ", self.pitch[id])
                elif motion_end[id]:
                    reset_reason = "motion end"
                elif vel_too_large[id]:
                    reset_reason = "velocity too large"
                elif self._pose_termination and pose_fail[id]:
                    reset_reason = "pose tracking failure"
                print(f"Env {id} reset due to: {reset_reason}")
    
    
    def _update_object_point_cloud_dist(self):
        obj_rot = self.object_root_states[:, 3:7]
        object_points = self.object_points.unsqueeze(0).repeat(self.num_envs, 1, 1)
        object_points_extend = object_points.view(-1, 3)

        obj_rot_extend = obj_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
        self.obj_points = quat_rotate(obj_rot_extend, object_points_extend).view(obj_rot.shape[0], object_points.shape[1], 3) + self.object_root_states[:, :3].unsqueeze(1)

        ref_obj_rot_extend = self._ref_object_root_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
        self.ref_obj_points = quat_rotate(ref_obj_rot_extend, object_points_extend).view(obj_rot.shape[0], object_points.shape[1], 3) + self._ref_object_root_pos.unsqueeze(1)

        self.object_point_cloud_dist = (self.obj_points - self.ref_obj_points).norm(dim=-1).mean(dim=-1)

    def _reward_tracking_object_point_cloud(self):
        object_point_cloud_scale = 10.0
        if not hasattr(self, "object_point_cloud_dist"):
            self._update_object_point_cloud_dist()
        return torch.exp(-object_point_cloud_scale * self.object_point_cloud_dist)

    def _get_object_points_world(self, root_pos, root_rot, max_points=128):
        # Downsample for visualization performance
        stride = max(1, self.object_points.shape[0] // max_points)
        obj_points = self.object_points[::stride]
        # Rotate and translate to world
        obj_rot_extend = root_rot.unsqueeze(1).repeat(1, obj_points.shape[0], 1).view(-1, 4)
        obj_points_extend = obj_points.unsqueeze(0).repeat(root_rot.shape[0], 1, 1).view(-1, 3)
        world_points = quat_rotate(obj_rot_extend, obj_points_extend).view(root_rot.shape[0], obj_points.shape[0], 3)
        world_points = world_points + root_pos.unsqueeze(1)
        return world_points

    # actual point cloud in yellow, 
    def draw_object_points_actual(self):
        if self.viewer is None:
            return
        geom = gymutil.WireframeSphereGeometry(0.02, 12, 12, None, color=(1, 1, 0))
        obj_points_world = self._get_object_points_world(self.object_root_states[:, :3], self.object_root_states[:, 3:7])
        for env_id in range(self.num_envs):
            for i in range(obj_points_world.shape[1]):
                pose = gymapi.Transform(gymapi.Vec3(obj_points_world[env_id, i, 0], obj_points_world[env_id, i, 1], obj_points_world[env_id, i, 2]), r=None)
                gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[env_id], pose)
    # reference motion point cloud in green
    def draw_object_points_motion(self):
        if self.viewer is None:
            return
        geom = gymutil.WireframeSphereGeometry(0.02, 12, 12, None, color=(0, 1, 0))
        ref_points_world = self._get_object_points_world(self._ref_object_root_pos, self._ref_object_root_rot)
        for env_id in range(self.num_envs):
            for i in range(ref_points_world.shape[1]):
                pose = gymapi.Transform(gymapi.Vec3(ref_points_world[env_id, i, 0], ref_points_world[env_id, i, 1], ref_points_world[env_id, i, 2]), r=None)
                gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[env_id], pose)
    
    def _process_object_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_object_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.object_friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.object_friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.object_friction_coeffs[env_id]
        else:
            for s in range(len(props)):
                props[s].friction = 1.0
        return props


    def _process_object_rigid_body_props(self, props, env_id):
        # No need to use tensors as only called upon env creation
        if self.cfg.domain_rand.randomize_object_mass:
            rng_mass = self.cfg.domain_rand.added_object_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
            for p_idx in range(len(props)):
                props[p_idx].mass += rand_mass

        if self.cfg.domain_rand.randomize_object_com:
            rng_com = self.cfg.domain_rand.added_object_com_range
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3, ))
            for p_idx in range(len(props)):
                props[p_idx].com += gymapi.Vec3(*rand_com)

        return props
