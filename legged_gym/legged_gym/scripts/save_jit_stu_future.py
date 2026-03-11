#!/usr/bin/env python3
"""
TorchScript conversion script for g1_stu_future (student policy with future motion support)
Usage: python save_jit_stu_future.py --proj_name g1_stu_future --exptid <experiment_id> [--checkpoint <checkpoint>]
"""

import os, sys
sys.path.append("../../../rsl_rl")
import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic_future import ActorFuture, get_activation
import argparse
from termcolor import cprint

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if not os.path.isdir(root):  # use first 4 chars to match the run name
        model_name_cand = os.path.basename(root)
        model_parent = os.path.dirname(root)
        model_names = os.listdir(model_parent)
        model_names = [name for name in model_names if os.path.isdir(os.path.join(model_parent, name))]
        for name in model_names:
            if len(name) >= 6:
                if name[:6] == model_name_cand:
                    root = os.path.join(model_parent, name)
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(root, model)
    return load_path, checkpoint

class HardwareStudentFutureNN(nn.Module):
    """Hardware deployment wrapper for student policy with future motion support."""
    
    def __init__(self,  
                 num_observations,
                 num_motion_observations,
                 num_priop_observations,
                 num_motion_steps,
                 num_future_observations,
                 num_future_steps,
                 motion_latent_dim,
                 future_latent_dim,
                 num_actions,
                 actor_hidden_dims,
                 activation,
                 history_latent_dim,
                 num_history_steps,
                 layer_norm=False,
                 tanh_encoder_output=False,
                 **kwargs):
        super().__init__()

        self.num_observations = num_observations
        self.num_actions = num_actions
        self.num_motion_observations = num_motion_observations
        self.num_priop_observations = num_priop_observations
        
        activation = get_activation(activation)
        
        self.normalizer = None
        
        self.actor = ActorFuture(
            num_observations=num_observations,
            num_motion_observations=num_motion_observations,
            num_priop_observations=num_priop_observations,
            num_motion_steps=num_motion_steps,
            num_future_observations=num_future_observations,
            num_future_steps=num_future_steps,
            motion_latent_dim=motion_latent_dim,
            future_latent_dim=future_latent_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            activation=activation,
            history_latent_dim=history_latent_dim,
            num_history_steps=num_history_steps,
            layer_norm=layer_norm,
            tanh_encoder_output=tanh_encoder_output,
            **kwargs
        )

    def load_normalizer(self, normalizer):
        self.normalizer = normalizer

    def forward(self, obs):
        assert obs.shape[1] == self.num_observations, f"Expected {self.num_observations} but got {obs.shape[1]}"
        obs = self.normalizer.normalize(obs)
        return self.actor(obs)

def convert_to_jit(args):
    """Convert g1_stu_future student policy to TorchScript."""
    
    ckpt_path = args.ckpt_path
    jit_path = ckpt_path.replace(".pt", "-jit.pt")
    
    # G1 student future configuration - EXACT DIMENSIONS FROM DEBUG
    robot_name = "g1"
    num_actions = 29
    history_len = 10
    
    # Motion observations (current frame only for student)
    num_motion_steps = 1
    num_motion_observations = 35  # n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single = 1 * 35
    
    # Proprioceptive observations
    num_priop_observations = 92  # n_proprio from debug output
    
    # Future motion observations
    num_future_steps = 10  # len(tar_motion_steps_future)
    n_future_obs_single = 35 + 1  # 35 dims per frame, +1 for mask indicator
    num_future_observations = num_future_steps * n_future_obs_single  # n_future_obs = 10 * 36 = 360
    
    # Single step observation size (for history)
    n_obs_single = 127  # n_mimic_obs + n_proprio = 35 + 92 = 127
    
    # Masked privileged information dimensions (optional)
    use_masked_priv_info = args.use_masked_priv_info  # Set to False to disable and revert to original dimensions
    n_masked_priv_info = 12 if use_masked_priv_info else 0  # motion state (9) + mask indicators (3)
    # Motion state: 3 lin_vel + 3 root_xyz + 3 root_rpy = 9 dims
    
    # Total observation size
    num_observations = n_obs_single * (history_len + 1) + num_future_observations + n_masked_priv_info  # 127 * 11 + 360 + 12 = 1769
    
    # Network architecture parameters
    motion_latent_dim = 128
    future_latent_dim = 128
    history_latent_dim = 128
    actor_hidden_dims = [512, 512, 256, 128]
    activation = 'silu'
    
    print(f"G1 Student Future Policy Configuration:")
    print(f"  Robot: {robot_name}")
    print(f"  Actions: {num_actions}")
    print(f"  History length: {history_len}")
    print(f"  Motion observations: {num_motion_observations}")
    print(f"  Proprioceptive observations: {num_priop_observations}")
    print(f"  Future observations: {num_future_observations}")
    print(f"  Masked priv info: {n_masked_priv_info} (enabled: {use_masked_priv_info})")
    print(f"  Single obs size: {n_obs_single}")
    print(f"  Total observations: {num_observations}")
    print(f"  Motion latent dim: {motion_latent_dim}")
    print(f"  Future latent dim: {future_latent_dim}")
    print(f"  History latent dim: {history_latent_dim}")
    print("")
    
    device = torch.device('cpu')
    policy = HardwareStudentFutureNN(
        num_observations=num_observations,
        num_motion_observations=num_motion_observations,
        num_priop_observations=num_priop_observations,
        num_motion_steps=num_motion_steps,
        num_future_observations=num_future_observations,
        num_future_steps=num_future_steps,
        motion_latent_dim=motion_latent_dim,
        future_latent_dim=future_latent_dim,
        num_actions=num_actions,
        actor_hidden_dims=actor_hidden_dims,
        activation=activation,
        history_latent_dim=history_latent_dim,
        num_history_steps=history_len,
        layer_norm=True,
        tanh_encoder_output=False
    ).to(device)

    # Load trained model
    cprint(f"Loading model from: {ckpt_path}", "green")
    
    # Load with weights_only=False to avoid the warning for now
    ac_state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy.load_state_dict(ac_state_dict['model_state_dict'], strict=False)
    policy.load_normalizer(ac_state_dict['normalizer'])
    
    policy = policy.to(device)


    # Convert to TorchScript
    policy.eval()
    with torch.no_grad(): 
        num_envs = 2
        
        # Create dummy input with correct observation structure
        obs_input = torch.ones(num_envs, num_observations, device=device)
        cprint(f"Input observation shape: {obs_input.shape}", "cyan")
        
        # Trace the model
        traced_policy = torch.jit.trace(policy, obs_input)
        
        # Save traced model
        traced_policy.save(jit_path)
        
        cprint(f"Saved traced student future policy at: {os.path.abspath(jit_path)}", "green")
        cprint(f"Model configuration: {robot_name} student with future motion support", "green")
        
        # Verify the traced model works
        test_output = traced_policy(obs_input)
        cprint(f"Verification successful - output shape: {test_output.shape}", "green")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert g1_stu_future student policy to TorchScript')
    parser.add_argument('--use_masked_priv_info', action='store_true', help='Use masked privileged information')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Checkpoint path')
    args = parser.parse_args()
    convert_to_jit(args)