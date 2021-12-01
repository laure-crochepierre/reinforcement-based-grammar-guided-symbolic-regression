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
import json 
import time
import torch
import itertools
torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(1)

import argparse

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
from multiprocessing import Pool

from envs import BatchSymbolicRegressionEnv
from algorithms import ReinforceAlgorithm
from policies import Policy

from multiprocessing import Process


BENCHMARKS = ['nguyen1', 'nguyen2', 'nguyen3', 'nguyen4', 'nguyen5', 'nguyen6', 'nguyen7', 'nguyen8', 'nguyen9',
              'nguyen10']

baseline_time = time.time()

def main(params):
    dataset, i, ablations = params

    print(f"dataset name {dataset} run number {i} ablation {ablations}")

    np.random.seed(i)
    torch.random.manual_seed(i)

    # Hyperparameters

    batch_size = 1000
    max_horizon = 50
    min_horizon = 4
    hidden_dim = 64
    embedding_dim = 64
    learning_rate = 0.001

    # parameters to tune
    risk_eps = 0.005
    entropy_coeff = 0.0005

    observe_brotherhood = True
    observe_parent = True
    observe_previous_actions = True
    observe_hidden_state = False


    env_kwargs = dict(grammar_file_path="../../grammars/nguyen_benchmark_v2.bnf",
                      start_symbol="<e>",
                      train_data_path=f"../../data/supervised_learning/{dataset}/train.csv",
                      test_data_path=f"../../data/supervised_learning/{dataset}/test.csv",
                      target="target",
                      # metric=lambda y, yhat: np.sqrt(mean_squared_error(y, yhat)),
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

    json_params = str({'env': env_kwargs, 'policy': policy_kwargs, 'algo': algo_kwargs})
    json.dump(json_params, open(f'../../results/ablation_study_final_paper/tmp/baseline_params{baseline_time}.json', 'w'))
    for ablation_type, ablation_param_name, ablation_param_value in ablations:
        if ablation_type == 'env':
            env_kwargs[ablation_param_name]=ablation_param_value
            print(ablation_type, ablation_param_name, ablation_param_value)
        if ablation_param_name == "hidden_dim" :
            hidden_dim = ablation_param_value
            env_kwargs["hidden_size"] = ablation_param_value
            
        if ablation_type == 'policy':
            policy_kwargs[ablation_param_name]=ablation_param_value
            print(ablation_type, ablation_param_name, ablation_param_value)
        if ablation_type == 'algo':
            algo_kwargs[ablation_param_name]=ablation_param_value
            print(ablation_type, ablation_param_name, ablation_param_value)

    model = ReinforceAlgorithm(env_class=BatchSymbolicRegressionEnv,
                               env_kwargs=env_kwargs,
                               policy_class=Policy,
                               policy_kwargs=policy_kwargs,
                               dataset=dataset,
                               debug=0, **algo_kwargs)

    n_epochs = int(2000000 / batch_size)
    debut = time.time()
    try:
        model.train(n_epochs=n_epochs)
    except Exception as e:
        print(e)
    duree = time.time() - debut

    var_y = model.env.y_test.var()
    nrmse = lambda y, yhat: mean_squared_error(y, yhat) / var_y
    metrics = [mean_squared_error, mean_absolute_error, r2_score, nrmse]

    f = eval(f'lambda x : {model.logger["best_expression"]}')
    y_pred = f(model.env.X_test)
    scores = [dataset]
    for m in metrics:
        try:
            scores += [m(model.env.y_test, y_pred)]
        except Exception as e:
            scores += [e]

    scores += [model.logger["best_expression"], duree, str(ablations)]
    results_df = pd.DataFrame(data=[scores], columns=['function_name', "mse", "mae", "r2", "nmse", "result", 'time', "ablation_description"])
    
    results_df.to_csv(f"../../results/ablation_study_final_paper/tmp/rbg2_sr_results_ablation_nguyen_{time.time()}.csv")
    return scores

if __name__ == "__main__":
    result_folder = "../../results/ablation_study_final_paper/tmp/"
    list_files = os.listdir(result_folder)
    result_df = pd.concat([pd.read_csv(result_folder + f) for f in list_files if( '.csv' in f) and('nguyen' in f)]).drop(columns=['Unnamed: 0'])
    ablations = [[("algo", "risk_eps", 1.0)],
                 [("algo", "entropy_coeff", 0.0)], 
                 [("env", "observe_parent", False)],[("env", "observe_parent", True)], 
                 [('env', "observe_brotherhood", False)],
                 [('env', "observe_previous_actions", False)], 
                 [('env', "observe_hidden_state", True)], 
                 [('env', "observe_symbol", False)],
                 [('env', "observe_mask", False)], 
                 [('env', "observe_depth", False)],
                 [("env", "observe_parent", False),('env', "observe_depth", False)],
                 [("env", "observe_parent", False),('env', "observe_brotherhood", False), ('env', "observe_previous_actions", False)],
                 [('env', "observe_symbol", False),('env', "observe_mask", False),('env', "observe_depth", False)], 
                 [("env", "observe_parent", False),('env', "observe_brotherhood", False),('env', "observe_depth", False)]
                 ]
    
    RUNS = 10
    combinaitions = []
    for b in BENCHMARKS:
        already_done = result_df[result_df.function_name == b]
        print("already_done") 
        for a in ablations : 
            n_already_done = len(already_done[already_done.ablation_description == str(a)])
            print(b, a, n_already_done)
            for i in range(RUNS-n_already_done):
                combinaitions += [[b, i, a]]
    print(len(combinaitions))
    
    
    with Pool(35) as p:
        results = p.map(main, combinaitions)

        results_df = pd.DataFrame(data=results, columns=['function_name', "mse", "mae", "r2", "nmse", "result", 'time', "ablation_description"])
        results_df.to_csv(f"../../results/rbg2_sr_results_ablation_nguyen_{time.time()}.csv")
        
