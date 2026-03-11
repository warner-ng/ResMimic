# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import code
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn.modules.activation import ReLU


def sinusoidal_positional_encoding(context_len, latent_dim):
    position = torch.arange(context_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, latent_dim, 2) * (-torch.log(torch.tensor(10000.0)) / latent_dim))
    pe = torch.zeros(context_len, latent_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# Transformer Backbone adapted from Stanford HumanPlus:
# https://github.com/MarkFzp/humanplus/blob/main/HST/rsl_rl/rsl_rl/modules/actor_critic_transformer.py

# a BERT-style transformer block
class Transformer_Block(nn.Module):
    def __init__(self, latent_dim, num_head, dropout_rate) -> None:
        super().__init__()
        self.num_head = num_head
        self.latent_dim = latent_dim
        self.ln_1 = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, num_head, dropout=dropout_rate, batch_first=True)
        self.ln_2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
            nn.Dropout(dropout_rate),
        )
    
    def forward(self, x):
        x = self.ln_1(x)
        x = x + self.attn(x, x, x, need_weights=False)[0]
        x = self.ln_2(x)
        x = x + self.mlp(x)
        
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, context_len, latent_dim=32, num_head=2, num_layer=2, dropout_rate=0.0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_len = context_len
        self.latent_dim = latent_dim
        self.num_head = num_head
        self.num_layer = num_layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Dropout(dropout_rate),
        )
        # self.weight_pos_embed = nn.Embedding(context_len, latent_dim) # learned positional encoding
        self.weight_pos_embed = sinusoidal_positional_encoding(context_len, latent_dim) # sinusoidal positional encoding
        self.attention_blocks = nn.Sequential(
            *[Transformer_Block(latent_dim, num_head, dropout_rate) for _ in range(num_layer)],
        )
        self.output_layer = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, output_dim),
        )
    
    def forward(self, x):
        x = self.input_layer(x)
        # x = x + self.weight_pos_embed(torch.arange(x.shape[1], device=x.device)) # learned positional encoding
        x = x + self.weight_pos_embed.to(x.device) # sinusoidal positional encoding
        x = self.attention_blocks(x)

        # take the last token
        x = x[:, -1, :]
        x = self.output_layer(x)

        return x


class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        # self.device = device
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10
        # last_activation = nn.ELU()

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size), self.activation_fn
                )

    def forward(self, obs):
        nd = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output

class Actor(nn.Module):
    def __init__(self, num_prop,
                 num_actions,
                 obs_context_len, 
                 priv_encoder_dims, 
                 num_priv_latent, 
                 num_hist, activation, 
                 tanh_encoder_output=False) -> None:
        super().__init__()
        # prop -> priv_explicit -> priv_latent -> hist
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        
        if len(priv_encoder_dims) > 0:
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            priv_encoder_output_dim = num_priv_latent
        
        self.actor_backbone = Transformer(input_dim=num_prop, output_dim=num_actions, context_len=obs_context_len)
        self.actor_backbone.output_layer[1].weight.data *= 0.01 # init last layer to be 100 times smaller

    def forward(self, obs_all, hist_encoding: bool, eval=False, scandots_latent=None):
        backbone_output = self.actor_backbone(obs_all)
        return backbone_output


class ActorCriticTransformer(nn.Module):
    is_recurrent = False
    def __init__(self,  num_prop,
                        num_critic_obs,
                        num_priv_latent, 
                        num_hist,
                        num_actions,
                        obs_context_len,
                        activation='elu',
                        init_noise_std=1.0,
                        fix_action_std=False,
                        action_std=None,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
            
        if fix_action_std:
            assert action_std is not None, "action_std must be provided if you need to fix action std!"
        
        super().__init__()

        self.fix_action_std = fix_action_std
        
        self.kwargs = kwargs
        priv_encoder_dims= kwargs['priv_encoder_dims']
        activation = get_activation(activation)
        
        self.actor = Actor(num_prop, num_actions, obs_context_len, priv_encoder_dims, num_priv_latent, num_hist, activation, tanh_encoder_output=kwargs['tanh_encoder_output'])
        self.critic = Transformer(input_dim=num_critic_obs, output_dim=1, context_len=obs_context_len)

        # Action noise
        if self.fix_action_std:
            self.init_action_std_tensor = torch.tensor(action_std)
            self.std = nn.Parameter(self.init_action_std_tensor, requires_grad=False)
        else:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, hist_encoding):
        mean = self.actor(observations, hist_encoding)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, hist_encoding=False, **kwargs):
        self.update_distribution(observations, hist_encoding)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, hist_encoding=False, eval=False, scandots_latent=None, **kwargs):
        if not eval:
            actions_mean = self.actor(observations, hist_encoding, eval, scandots_latent)
            return actions_mean
        else:
            actions_mean, latent_hist, latent_priv = self.actor(observations, hist_encoding, eval=True)
            return actions_mean, latent_hist, latent_priv

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    
    def reset_std(self, std, num_actions, device):
        new_std = std * torch.ones(num_actions, device=device)
        self.std.data = new_std.data
        
    def update_std(self, std_scale):
        if self.std.device != self.init_action_std_tensor.device:
            self.std.data = std_scale * self.init_action_std_tensor.to(self.std.device)
        else:
            self.std.data = std_scale * self.init_action_std_tensor
        
    def if_fix_std(self):
        return self.fix_action_std
        

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
