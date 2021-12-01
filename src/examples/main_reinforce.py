# Copyright (c) 2020-2021, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of RBG2-SR a reinforcement based approach to grammar guided symbolic regression.

import sys
sys.path.insert(0, '..')

import torch
torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(2)

import argparse
import numpy as np

from sklearn.metrics import mean_squared_error

from envs import BatchSymbolicRegressionEnv
from algorithms import ReinforceAlgorithm
from policies import Policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Deep Symbolic Regression')
    parser.add_argument('-dataset', '--d', help='Dataset name', dest="dataset", default="nguyen3")
    parser.add_argument('-batch_size', '--b', help='Batch Size', dest="batch_size", default=1000)
    parser.add_argument('-max_horizon', '--m', help='Max Horizon', dest="max_horizon", default=50)
    parser.add_argument('-min_horizon', '--n', help='Min Horizon', dest="min_horizon", default=4)
    parser.add_argument('-hidden_dim', '--h', help='Hidden Dim', dest="hidden_dim", default=64)
    parser.add_argument('-embedding_dim', '--f', help='Embedding dim', dest="embedding_dim", default=8)
    parser.add_argument('-risk_eps', '--r', help='Risk Epsilon', dest="risk_eps", default=0.1)
    parser.add_argument('-entropy_coeff', '--e', help='Entropy Coefficient', dest="entropy_coeff", default=0.0005)
    parser.add_argument('-learning_rate', '--l', help='Learning rate', dest="learning_rate", default=0.001)
    parser.add_argument('-observe_parent', '--p', help='Observe parent (True or False)', dest="observe_parent",
                        default="True")
    parser.add_argument('-observe_siblings', '--s', help='Observe siblings (True or False)', dest="observe_siblings",
                        default="True")
    parser.add_argument('-autoencoder', '--a', help='Use autoencoder (True or False)', dest="autoencoder",
                        default="False")
    parser.add_argument('-init_type', '--i', help='Initialisation type (randint or zeros)', dest="init_type",
                        default='randint')

    args = parser.parse_args()
    dataset = args.dataset
    batch_size = int(args.batch_size)
    max_horizon = int(args.max_horizon)
    min_horizon = int(args.min_horizon)
    hidden_dim = int(args.hidden_dim)
    embedding_dim = int(args.embedding_dim)
    risk_eps = float(args.risk_eps)
    entropy_coeff = float(args.entropy_coeff)
    learning_rate = float(args.learning_rate)
    observe_parent = args.observe_parent == "True"
    observe_brotherhood = args.observe_siblings == "True"
    autoencoder = args.autoencoder == "True"
    init_type = args.init_type
    print(observe_brotherhood, observe_parent, autoencoder)

    home_path = '../..'
    env_kwargs = dict(grammar_file_path=f"{home_path}/grammars/nguyen_benchmark_v2.bnf",
                      start_symbol="<e>",
                      train_data_path=f"{home_path}/data/supervised_learning/{dataset}/train.csv",
                      test_data_path=f"{home_path}/data/supervised_learning/{dataset}/test.csv",
                      target="target",
                      eval_params={},
                      max_horizon=max_horizon,
                      min_horizon=min_horizon,
                      hidden_size=hidden_dim,
                      batch_size=batch_size,
                      normalize=False,
                      observe_parent=observe_parent,
                      observe_brotherhood=observe_brotherhood)

    policy_kwargs = dict(embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                         use_transformer=False,
                         embedding=True, autoencoder=autoencoder)

    algo_kwargs = dict(batch_size=batch_size,
                       entropy_coeff=entropy_coeff,
                       optimizer_class=torch.optim.Adam,
                       learning_rate=learning_rate,
                       init_type=init_type,
                       reward_prediction=False,
                       risk_eps=risk_eps)

    model = ReinforceAlgorithm(env_class=BatchSymbolicRegressionEnv,
                               env_kwargs=env_kwargs,
                               policy_class=Policy,
                               policy_kwargs=policy_kwargs,
                               dataset=dataset,
                               debug=1, **algo_kwargs)

    n_epochs = int(2000000/batch_size)
    model.train(n_epochs=n_epochs)

