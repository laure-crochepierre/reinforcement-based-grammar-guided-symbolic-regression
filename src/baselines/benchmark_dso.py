import os
import json
import time 

from dso import DeepSymbolicRegressor
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

benchmarks = ['nguyen1', 'nguyen2', 'nguyen3', 'nguyen4', 'nguyen5', 'nguyen6', 'nguyen7', 'nguyen8', 'nguyen9', 'nguyen10', 
             'keijzer1', 'keijzer2', 'keijzer3', 'keijzer4', 'keijzer5', 'keijzer6', 'keijzer7', 'keijzer8', 'keijzer9', 'keijzer10', 
             'keijzer11', 'keijzer12', 'keijzer13','keijzer14','keijzer15', 'pagie1', 
             'vladislavleva1','vladislavleva2', 'vladislavleva3','vladislavleva4', 'vladislavleva5', 'vladislavleva6', 'vladislavleva7', 'vladislavleva8']
benchmarks=['keijzer1', 'keijzer2', 'keijzer3', 'keijzer7', 'keijzer8', 'keijzer14', 'keijzer15', 'vladislavleva1',
            'vladislavleva4', 'vladislavleva5', 'vladislavleva7', 'vladislavleva8']

#benchmarks = benchmarks[:2]
results = []
RUNS = 30

date = time.time()
for i in range(RUNS):
    for dataset in benchmarks:
        home_folder = "../../../benchmark_sr"
        print("dataset {} run number {} \r".format(dataset, i), flush=True)
        df_train = pd.read_csv(f"{home_folder}/data/supervised_learning/{dataset}/train.csv").dropna()
        df_test = pd.read_csv(f"{home_folder}/data/supervised_learning/{dataset}/test.csv").dropna()
        X_train = df_train.drop(columns=["target"]).values
        X_test = df_test.drop(columns=["target"]).values
        y_train = df_train.target.values
        y_test = df_test.target.values

        try :
            config = "config/config_regression_no_comments.json"
            # Create the model
            with open(config, encoding='utf-8') as f:
                json_config_dict = json.load(f)
                json_config_dict['task']['metric'] = "inv_mse"
                json_config_dict["experiment"] = {"seed": i}
                print(json_config_dict, flush=True)
            model = DeepSymbolicRegressor(json_config_dict)

            # Fit the model
            debut = time.time()
            model.fit(X_train, y_train)
            duree = time.time() - debut 

            # View the best expression
            print(model.program_.pretty(), flush=True)

            # Make predictions
            y_pred = model.predict(X_test)

            nrmse = lambda y, y_hat: np.sqrt(np.mean((y - y_hat) ** 2 / var_y))

            metrics = [mean_squared_error, mean_absolute_error, r2_score, nrmse]
            scores = [dataset]
            for m in metrics:
                try :
                    scores += [m(y_test, y_pred)]
                except:
                    scores += [-1]
            scores += [model.program_.pretty(), duree]
            results.append(scores)

        except Exception as e :
            print(e)
            results.append([dataset, 0, 0, 0, 0, None, 0])

        results_df = pd.DataFrame(data=results, columns=['function_name', "mse", "mae", "r2", 'nrmse', "result", 'time'])
        results_df.to_csv(f"dsr_results_{date}.csv")


