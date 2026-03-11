import os, pickle, yaml
import torch
from pose.utils.torch_utils import quat_diff, quat_to_exp_map, slerp, euler_from_quaternion
from tqdm import tqdm
from rich import print
from isaacgym.torch_utils import quat_rotate_inverse
import sys
from types import ModuleType
import numpy as np
from pose.utils.motion_lib_pkl import MotionLib

# Patch sys.modules to fake missing modules from numpy 2.x
class FakeModule(ModuleType):
    def __init__(self, name, real=None):
        super().__init__(name)
        if real:
            self.__dict__.update(real.__dict__)

# Patch potentially missing modules
sys.modules['numpy._core'] = FakeModule('numpy._core', np.core if hasattr(np, 'core') else np)
sys.modules['numpy._core.multiarray'] = FakeModule('numpy._core.multiarray', getattr(np.core, 'multiarray', None))


def smooth(x, box_pts, device):
    box = torch.ones(box_pts, device=device) / box_pts
    num_channels = x.shape[1]
    x_reshaped = x.T.unsqueeze(0)
    smoothed = torch.nn.functional.conv1d(
        x_reshaped,
        box.view(1, 1, -1).expand(num_channels, 1, -1),
        groups=num_channels,
        padding='same'
    )
    return smoothed.squeeze(0).T


class MotionLibHOI(MotionLib):
    def __init__(self, motion_file, object_motion_file, device, 
                 motion_decompose=False, 
                 motion_smooth=True, 
                 motion_height_adjust=False,
                 sample_ratio=1.0 # only sample a portion of the motion
                 ):
        super().__init__(motion_file, device, motion_decompose, motion_smooth, motion_height_adjust, sample_ratio)
        object_data = np.load(object_motion_file)
        self._object_root_pos = torch.tensor(object_data['trans'], dtype=torch.float32, device=device)
        self._object_root_rot = torch.tensor(object_data['rot'], dtype=torch.float32, device=device)


    def calc_hoi_motion_frame(self, motion_ids, motion_times):
        motion_loop_num = torch.floor(motion_times / self._motion_lengths[motion_ids])
        motion_times -= motion_loop_num * self._motion_lengths[motion_ids]
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_ids, motion_times)
        
        root_pos0 = self._motion_root_pos[frame_idx0]
        root_pos1 = self._motion_root_pos[frame_idx1]
        
        root_rot0 = self._motion_root_rot[frame_idx0]
        root_rot1 = self._motion_root_rot[frame_idx1]

        object_root_pos0 = self._object_root_pos[frame_idx0]
        object_root_pos1 = self._object_root_pos[frame_idx1]
        object_root_rot0 = self._object_root_rot[frame_idx0]
        object_root_rot1 = self._object_root_rot[frame_idx1]
        
        root_vel = self._motion_root_vel[frame_idx0]
        root_ang_vel = self._motion_root_ang_vel[frame_idx0]
        
        dof_pos0 = self._motion_dof_pos[frame_idx0]
        dof_pos1 = self._motion_dof_pos[frame_idx1]
        
        local_key_body_pos0 = self._motion_local_body_pos[frame_idx0]
        local_key_body_pos1 = self._motion_local_body_pos[frame_idx1]
        
        dof_vel = self._motion_dof_vel[frame_idx0]
        
        blend_unsqueeze = blend.unsqueeze(-1)
        root_pos = (1.0 - blend_unsqueeze) * root_pos0 + blend_unsqueeze * root_pos1
        root_pos += motion_loop_num.unsqueeze(-1) * self._motion_root_pos_delta[motion_ids]
        root_rot = slerp(root_rot0, root_rot1, blend)

        object_root_pos = (1.0 - blend_unsqueeze) * object_root_pos0 + blend_unsqueeze * object_root_pos1
        object_root_pos += motion_loop_num.unsqueeze(-1) * self._motion_root_pos_delta[motion_ids]
        object_root_rot = slerp(object_root_rot0, object_root_rot1, blend)

        dof_pos = (1.0 - blend_unsqueeze) * dof_pos0 + blend_unsqueeze * dof_pos1
        
        local_key_body_pos = (1.0 - blend_unsqueeze.unsqueeze(1)) * local_key_body_pos0 + blend_unsqueeze.unsqueeze(1) * local_key_body_pos1
        
        # compute the root pos delta compared to last frame
        root_pos_delta_local0 = self._motion_root_pos_delta_local[frame_idx0]
        root_pos_delta_local1 = self._motion_root_pos_delta_local[frame_idx1]
        root_pos_delta_local = (1.0 - blend_unsqueeze) * root_pos_delta_local0 + blend_unsqueeze * root_pos_delta_local1

        # compute the root rot delta compared to last frame 
        root_rot_delta_local0 = self._motion_root_rot_delta_local[frame_idx0]
        root_rot_delta_local1 = self._motion_root_rot_delta_local[frame_idx1]
        # we use linear interpolation for root rot delta, as it is euler angle
        root_rot_delta_local = (1.0 - blend_unsqueeze) * root_rot_delta_local0 + blend_unsqueeze * root_rot_delta_local1

        return root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos, root_pos_delta_local, root_rot_delta_local, object_root_pos, object_root_rot