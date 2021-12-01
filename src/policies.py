# Copyright (c) 2020-2021, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of RBG2-SR a reinforcement based approach to grammar guided symbolic regression.

import math
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler

torch.autograd.set_detect_anomaly(True)

from typing import NamedTuple, Dict, Union
from gym.spaces import Box, MultiBinary

from utils.masking_categorical import CategoricalMasked

TensorDict = Dict[Union[str, int], torch.Tensor]


class CuriousDictRolloutBufferSamples(NamedTuple):
    intrinsic_rewards: torch.Tensor
    extrinsic_rewards: torch.Tensor
    log_probs: torch.Tensor
    entropies: torch.Tensor


class Policy(nn.Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_dim,
                 embedding_dim,
                 max_horizon,
                 embedding=False,
                 autoencoder=False,
                 batch_size=1,
                 reward_prediction=False,
                 use_transformer=False,
                 non_linearity=nn.LeakyReLU(),
                 **kwargs):
        super(Policy, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_horizon = max_horizon
        self.n_actions = self.action_space.shape[0]
        self.batch_size = batch_size
        self.non_linearity = non_linearity
        self.use_transformer = use_transformer
        # force embedding
        self.embedding = embedding or (sum([len(s.shape) >= 2 for s in self.observation_space.spaces.values()]) > 0)
        self.autoencoder = self.embedding & autoencoder
        self.reward_prediction = reward_prediction

        if self.embedding:
            # Define Encoder Architecture
            self.encoders = {}
            for key, space in self.observation_space.spaces.items():
                if isinstance(space, Box):
                    setattr(self, f"encoder_{key}", nn.Linear(space.shape[0], self.embedding_dim))

                elif isinstance(space, MultiBinary):
                    if len(space.shape) == 1:
                        setattr(self, f"encoder_{key}", nn.Sequential(
                            nn.Linear(space.shape[0], self.embedding_dim),
                            self.non_linearity))
                    else:
                        if self.use_transformer:
                            transformer_encoder = nn.Sequential(
                                nn.TransformerEncoderLayer(d_model=space.shape[1], nhead=1, dim_feedforward=16,
                                                           batch_first=True),
                                nn.Conv1d(space.shape[0],1, 4),
                                self.non_linearity,
                                nn.Linear(space.shape[1]-4+1, self.embedding_dim),
                                self.non_linearity
                            )

                            setattr(self, f"encoder_{key}", transformer_encoder)
                        else :
                            encoder = nn.Sequential(
                                nn.Conv1d(space.shape[0], 1, 4),
                                self.non_linearity,
                                nn.Linear(space.shape[1] - 4 + 1, self.embedding_dim),
                                self.non_linearity
                            )

                            setattr(self, f"encoder_{key}", encoder)
                self.encoders[key] = getattr(self, f"encoder_{key}")

            self.features_encoder_layer = nn.Linear(self.embedding_dim*len(self.observation_space.spaces),
                                              self.hidden_dim)
            if self.autoencoder:
                self.decoders = {}
                for key, space in self.observation_space.spaces.items():
                    if isinstance(space, Box):
                        setattr(self, f"decoder_{key}", nn.Linear(self.embedding_dim, space.shape[0]))
                    elif isinstance(space, MultiBinary):
                        if len(space.shape) == 1:
                            setattr(self, f"decoder_{key}", nn.Sequential(
                                nn.Linear(self.embedding_dim, space.shape[0]),
                                self.non_linearity))
                        else:
                            setattr(self, f"decoder_{key}", nn.Sequential(
                                nn.Linear(self.embedding_dim, space.shape[1]),
                                self.non_linearity,
                                nn.ConvTranspose1d(in_channels=self.batch_size,
                                                   out_channels=32,
                                                   kernel_size=1),
                                self.non_linearity,
                                nn.ConvTranspose1d(in_channels=32,
                                                   out_channels=space.shape[0],
                                                   kernel_size=1),
                                self.non_linearity

                            ))
                    self.decoders[key] = getattr(self, f"decoder_{key}")
                self.ae_coeff_loss = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        else:
            inputs_dim = sum([s.shape[0] for s in self.observation_space.spaces.values()])
            self.features_encoder_layer = nn.Linear(inputs_dim, self.hidden_dim)

        # Define Action predictor Architecture

        self.lstm_layer = nn.LSTM(input_size=self.hidden_dim,
                                  hidden_size=self.hidden_dim, batch_first=True)
        self.intermediate_layer = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.action_decoder_layer = nn.Linear(in_features=self.hidden_dim, out_features=self.n_actions)
        self.softmax = nn.Softmax(dim=-1)

        self.multiply = torch.multiply
        self.score_predictor = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=int(self.hidden_dim/2)),
            self.non_linearity,
            nn.Linear(in_features=int(self.hidden_dim/2), out_features=int(self.hidden_dim /4)),
            self.non_linearity,
            nn.Linear(in_features=int(self.hidden_dim / 4), out_features=self.n_actions),
        )

    def encode(self, inputs):
        if self.embedding:
            encoded_inputs = {}
            for k in self.observation_space.spaces.keys():
                if not isinstance(self.encoders[k], nn.Transformer):
                    encoded_inputs[k] = self.encoders[k](torch.Tensor(inputs[k]))
                else:
                    encoded_inputs[k] = self.encoders[k](torch.Tensor(inputs[k]), torch.Tensor(inputs[k]))
            cat_inputs = torch.cat(list(encoded_inputs.values()), -1)
            if self.autoencoder:
                decoded_inputs = {k: self.decoders[k](encoded_input) for k, encoded_input in encoded_inputs.items()}
                return self.features_encoder_layer(cat_inputs), decoded_inputs
            else:
                return self.features_encoder_layer(cat_inputs), inputs
        else:
            cat_inputs = torch.cat([torch.Tensor(inputs[k]) for k in self.observation_space.spaces.keys()], -1)
            return self.features_encoder_layer(cat_inputs), inputs

    def forward(self, inputs, h_in, c_in):

        x_inputs, inputs_hat = self.encode(inputs)
        x = self.non_linearity(x_inputs)
        x_lstm, (h_out, c_out) = self.lstm_layer(x, (h_in, c_in))
        x_lstm = nn.Tanh()(x_lstm)

        x = self.intermediate_layer(x_lstm)
        x = self.non_linearity(x)

        action_logits = self.action_decoder_layer(x)
        score_prediction = self.score_predictor(x_lstm)

        return action_logits, h_out, c_out, [inputs_hat, score_prediction]

    def select_action(self, state, h_in, c_in):
        action_logits, h_out, c_out, other_predictions = self.forward(state, h_in, c_in)

        # create a categorical distribution over the list of probabilities of actions
        m = CategoricalMasked(logits=action_logits, masks=torch.BoolTensor(state['current_mask']))

        # and sample an action using the distribution
        action = m.sample()

        # compute log_probs
        log_probs = m.log_prob(action)
        entropy = m.entropy()

        return action, log_probs, entropy, h_out, c_out, other_predictions


class ActorCriticPolicy(Policy):
    def __init__(self, **kwargs):
        super(ActorCriticPolicy, self).__init__(**kwargs)
        self.actor_intermediate_layer = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.actor_decoder_layer = nn.Linear(in_features=self.hidden_dim, out_features=self.n_actions)
        self.critic_intermediate_layer = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.critic_decoder_layer = nn.Linear(in_features=self.hidden_dim, out_features=self.n_actions)

    def forward(self, inputs, h_in, c_in):
        x_inputs, inputs_hat = self.encode(inputs)
        x, (h_out, c_out) = self.lstm_layer(x_inputs, (h_in, c_in))
        x = self.relu(x)

        # Actor
        x_actor = self.actor_intermediate_layer(x)
        x_actor = self.relu(x_actor)
        action_logits = self.actor_decoder_layer(x_actor)

        # critic
        x_critic = self.critic_intermediate_layer(x)
        x_critic = self.relu(x_critic)

        critic_logits = self.critic_decoder_layer(x_critic)

        return action_logits, h_out, c_out, inputs_hat, critic_logits

    def select_action(self, state, h_in, c_in):
        action_logits, h_out, c_out, inputs_hat, critic_logits = self.forward(state, h_in, c_in)

        # create a categorical distribution over the list of probabilities of actions
        m = CategoricalMasked(logits=action_logits, masks=torch.BoolTensor(state['current_mask']))

        # and sample an action using the distribution
        action = m.sample()

        # compute log_probs
        log_probs = m.log_prob(action)
        entropy = m.entropy()

        # get critic value
        critic_value = torch.gather(critic_logits.squeeze(1), 1, action)

        return action, log_probs, entropy, h_out, c_out, inputs_hat, critic_value


class DqnPolicy(Policy):

    def __init__(self, epsilon_greedy_start=1.0, **kwargs):
        super(DqnPolicy, self).__init__(**kwargs)
        self.epsilon_greedy = epsilon_greedy_start
        self.epsilon_greedy_start = epsilon_greedy_start
        self.epsilon_greedy_end = 0.1
        self.epsilon_greedy_decay = 200
        self.steps_done = 0

    def select_action(self, state, h_in, c_in):

        with torch.inference_mode():
            q_value_logits, h_out, c_out, inputs_hat = self.forward(state, h_in, c_in)
            q_value_logits = torch.where(torch.BoolTensor(state['current_mask']),
                                         q_value_logits,
                                         torch.ones_like(q_value_logits) * float(-1e3))
        self.epsilon_greedy = self.epsilon_greedy_end + (self.epsilon_greedy_start - self.epsilon_greedy_end) * \
                              math.exp(-1. * self.steps_done / self.epsilon_greedy_decay)
        self.steps_done += 1

        def _select_one_action(q_values, mask):
            if np.random.random(1) < self.epsilon_greedy:
                a = torch.argmax(q_values)
            else:
                a = torch.Tensor(list(WeightedRandomSampler(mask, 1)))[0, 0]
            return a
        action = torch.vstack([_select_one_action(x_i, state['current_mask'][i])
                               for i, x_i in enumerate(torch.unbind(q_value_logits, dim=0))])[:, 0]
        q_value = torch.vstack([x_i[0, action[i].detach().numpy()] for i, x_i in
                                enumerate(torch.unbind(q_value_logits, dim=0))])

        return action, q_value, h_out, c_out, inputs_hat
