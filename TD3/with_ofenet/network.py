# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved.

# The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages.

# As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.

import torch as th
import torch.nn as nn
from blocks import DensenetBlock
import copy
import numpy as np
import gym
import sys


class OFENet(nn.Module):
    def __init__(self, dim_state, dim_action, dim_output, total_units, num_layers, aux_task, env_name, skip_action_branch=False):
        super().__init__()

        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        # parameters
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_output = dim_output
        self.total_units = total_units
        self.num_layers = num_layers
        self.aux_task = aux_task
        self._skip_action_branch = skip_action_branch
        state_layer_units, action_layer_units = calculate_layer_units(state_dim=dim_state,
                                                                      action_dim=dim_action,
                                                                      total_units=total_units,
                                                                      num_layers=num_layers)
        state_blocks = []
        action_blocks = []
        block_class = DensenetBlock

        for idx_layer, cur_layer_units in enumerate(state_layer_units):
            cur_block = block_class(units_per_layer=cur_layer_units)
            state_blocks.append(cur_block)

        for idx_layer, cur_layer_units in enumerate(action_layer_units):
            cur_block = block_class(units_per_layer=cur_layer_units)
            action_blocks.append(cur_block)

        # build the network
        self.state_blocks = state_blocks
        self.action_blocks = action_blocks
        self.state_model = nn.Sequential(*state_blocks)
        self.action_model = nn.Sequential(*action_blocks)
        self.out_layer = nn.LazyLinear(dim_output)
        self.state_model.to(self.device)
        self.action_model.to(self.device)
        self.out_layer.to(self.device)
        self.optimizer = th.optim.Adam(self.parameters())
        self._dim_state_features = total_units + dim_state
        self._dim_state_action_features = total_units + dim_state + dim_action + total_units

        # self.eps = 1e-10

        # some metrics of the representations
        self.mae_train = 0
        self.mae_percent_train = 0
        self.mape_train = 0
        self.mse_train = 0
        self.mse_percent_train = 0

        self.mae_test = 0
        self.mae_percent_test = 0
        self.mape_test = 0
        self.mse_test = 0
        self.mse_percent_test = 0

    @property
    def dim_state_features(self):
        return self._dim_state_features

    @property
    def dim_state_action_features(self):
        return self._dim_state_action_features

    # forward is used for training of OFENet
    def forward(self, inputs):
        [states, actions] = inputs
        features = states

        self.state_model.train(True)
        self.action_model.train(True)
        features = self.state_model(features)

        if not self._skip_action_branch:
            features = th.cat([features, actions], 1)
            features = self.action_model(features)

        values = self.out_layer(features)
        return values

    # used within TD3 -> .eval() disables batchnorm
    def features_from_states(self, states):
        features = states
        self.state_model.eval()
        features = self.state_model(features)
        return features

    def features_from_states_actions(self, states, actions):
        self.action_model.eval()
        state_features = self.features_from_states(states)
        features = th.cat([state_features, actions], 1)
        features = self.action_model(features)
        return features

    # train OFENet with the chosen auxiliary task
    def train_ofe(self, states, actions, next_states, rewards, dones):
        target_dim = self.dim_output

        if self.aux_task == "fsp":
            predicted_states = self([states, actions])
            target_states = next_states[:, :target_dim]
            feature_loss = th.mean((target_states - predicted_states) ** 2)

        elif self.aux_task == "fsdp":
            predicted_states_diff = self([states, actions])
            target_states = next_states[:, :target_dim]
            target_states_diff = target_states - states[:, :target_dim]
            feature_loss = th.mean((target_states_diff - predicted_states_diff) ** 2)

        elif self.aux_task == "rwp":
            target_rewards = rewards
            predicted_rewards = self([states, actions])
            feature_loss = th.mean((target_rewards - predicted_rewards) ** 2)

        self.mse_train = feature_loss
        self.optimizer.zero_grad(set_to_none=True)
        feature_loss.backward()
        self.optimizer.step()


def calculate_layer_units(state_dim, action_dim, total_units, num_layers):
    assert total_units % num_layers == 0
    per_unit = total_units // num_layers
    state_layer_units = [per_unit] * num_layers
    action_layer_units = [per_unit] * num_layers

    return state_layer_units, action_layer_units



