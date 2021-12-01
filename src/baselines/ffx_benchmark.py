import os
import ffx

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


benchmarks = sorted([l for l in os.listdir("../../data/supervised_learning/") if (l[0] != ".") &
                     os.path.isdir("../../data/supervised_learning/" + l)])

results = []
RUNS = 1
for dataset in benchmarks:
    for i in range(RUNS):
        print("dataset {} run number {} \r".format(dataset, i))
        df_train = pd.read_csv(f"../../data/supervised_learning/{dataset}/train.csv").dropna()
        df_test = pd.read_csv(f"../../data/supervised_learning/{dataset}/test.csv").dropna()
        X_train = df_train.drop(columns=["target"]).values
        X_test = df_test.drop(columns=["target"]).values
        y_train = df_train.target.values
        y_test = df_test.target.values

        try :
            FFX = ffx.FFXRegressor()
            FFX.fit(X_train, y_train)
            y_pred = FFX.predict(X_test)
            var_y = y_test.var()

            nrmse = lambda y, y_hat: np.sqrt(np.mean((y - y_hat) ** 2 / var_y))

            metrics = [mean_squared_error, mean_absolute_error, r2_score, nrmse]
            scores = [dataset]
            for m in metrics:
                try :
                    scores += [m(y_test, y_pred)]
                except:
                    scores += [-1]
            scores += [FFX.model_]
            results.append(scores)
        except Exception as e:
            results.append([dataset, np.inf, np.inf, -1, None])


results_df = pd.DataFrame(data=results, columns=['function_name', "mse", "mae", "r2", "nrmse",  "result"])
results_df.to_csv("ffx_results_paper_results_3.csv")
