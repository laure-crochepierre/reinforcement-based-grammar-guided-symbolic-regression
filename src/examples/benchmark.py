# Copyright (c) 2020-2021, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of RBG2-SR a reinforcement based approach to grammar guided symbolic regression.

import sys
sys.path.insert(0, '..')

import os
import time
import torch
torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(1)

import random
import argparse

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
from multiprocessing import Pool

from envs import BatchSymbolicRegressionEnv
from algorithms import ReinforceAlgorithm
from policies import Policy


def main(params): 
    dataset, i, run_name = params
    print("dataset name {} run number {}".format(dataset, i))

    # Hyperparameters

    batch_size = 1000
    max_horizon = 50
    min_horizon = 4
    hidden_dim = 64
    embedding_dim=64
    risk_eps = 0.005
    entropy_coeff = 0.0005
    learning_rate = 0.001
    observe_brotherhood = True
    observe_parent = True
    observe_previous_actions = True
    observe_hidden_state = False

    env_kwargs = dict(grammar_file_path="../../grammars/nguyen_benchmark_v2.bnf",
                      start_symbol="<e>",
                      train_data_path=f"../../data/supervised_learning/{dataset}/train.csv",
                      test_data_path=f"../../data/supervised_learning/{dataset}/test.csv",
                      target="target",
                      #metric=lambda y, yhat: np.sqrt(mean_squared_error(y, yhat)),
                      eval_params={},
                      max_horizon=max_horizon,
                      min_horizon=min_horizon,
                      hidden_size=hidden_dim,
                      batch_size=batch_size,
                      observe_brotherhood=observe_brotherhood,
                      observe_parent=observe_parent,
                      observe_previous_actions=observe_previous_actions,
                      observe_hidden_state=observe_hidden_state,
                      normalize=False)

    policy_kwargs = dict(embedding_dim=embedding_dim, hidden_dim=hidden_dim, embedding=True)

    algo_kwargs = dict(batch_size=batch_size,
                       entropy_coeff=entropy_coeff,
                       optimizer_class=torch.optim.Adam,
                       learning_rate=learning_rate,
                       init_type='randint',
                       risk_eps=risk_eps)
    try:
        model = ReinforceAlgorithm(env_class=BatchSymbolicRegressionEnv,
                               env_kwargs=env_kwargs,
                               policy_class=Policy,
                               policy_kwargs=policy_kwargs,
                               dataset=dataset,
                               debug=0, **algo_kwargs)
    except Exception as e:
        print(e, "model", dataset, i)

    n_epochs = int(2000000/batch_size)
    debut = time.time()
    try : 
        model.train(n_epochs=n_epochs)
    except Exception as e: 
        print(e, "train",  dataset, i)
    duree = time.time() - debut

    var_y = model.env.y_test.var()
    nrmse = lambda y, yhat: mean_squared_error(y, yhat) / var_y
    metrics = [mean_squared_error, mean_absolute_error, r2_score, nrmse]

    f = eval(f'lambda x : {model.logger["best_expression"]}')
    y_pred = f(model.env.X_test)
    scores = [dataset]
    for m in metrics:
        try:
            scores += [m(model.env.y_test,  y_pred)]
        except Exception as e:
            scores += [e]

    scores += [model.logger["best_expression"], duree]
    print(scores)
    pd.DataFrame(data=[scores],
                columns=['function_name', "mse", "mae", "r2", "nrmse", "result", 'time']).to_csv(f"../results/run_{run_name}/tmp/{dataset}_{i}_{time.time()}")
    return scores


if __name__ == "__main__":
    benchmarks = ['nguyen1', 'nguyen2', 'nguyen3', 'nguyen4', 'nguyen5', 'nguyen6', 'nguyen7', 'nguyen8', 'nguyen9', 'nguyen10',
                  'keijzer1', 'keijzer2', 'keijzer3', 'keijzer4', 'keijzer5', 'keijzer6', 'keijzer7','keijzer8', 'keijzer9', 'keijzer10', 'keijzer11', 'keijzer12', 'keijzer13', 'keijzer14', 'keijzer15',
                  'pagie1','vladislavleva1', 'vladislavleva2', 'vladislavleva3','vladislavleva4', 'vladislavleva5', 'vladislavleva6', 'vladislavleva7', 'vladislavleva8']
    random.shuffle(benchmarks)
    print(benchmarks)
    RUNS = 30
    
    dt = time.time()

    parser = argparse.ArgumentParser(description='Benchmark RG2SR')
    parser.add_argument('-run_name', '--n', help="Run name", dest="run_name", default=len(os.listdir("../results")))

    args = parser.parse_args()

    run_name = args.run_name
    os.makedirs('../../results', exist_ok=True)
    os.makedirs(f'../../results/run_{run_name}', exist_ok=True)
    os.makedirs(f'../../results/run_{run_name}/tmp', exist_ok=True)
    print(f"../../results/run_{run_name}/rg2sr_{dt}_process.csv")

    results = []
    
    combinaitions = []
    for i in range(RUNS):
        for b in benchmarks:
            combinaitions += [[b, i, run_name]]
    print(len(combinaitions))


    with Pool(34) as p:
        results = p.map(main, combinaitions)
        results_df = pd.DataFrame(data=results,
                columns=['function_name', "mse", "mae", "r2", "nrmse", "result", 'time'])
        results_df.to_csv(f"../../results/run_{run_name}/rg2sr_{dt}_process.csv")

