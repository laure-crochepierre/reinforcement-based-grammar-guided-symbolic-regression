# Inspired by https://nathankjer.com/evolutionary-algorithms-python/
import os
import sys
sys.path.append(f"./../utils")

import warnings
warnings.simplefilter("ignore")

import time
import pickle
import random
import pandas as pd
import numpy as np
random.seed()
from multiprocessing import Pool

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

from grammar_parser import ProbabilisticGrammar
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


nmse = lambda y, y_hat : np.sqrt(np.mean((y - y_hat)**2 / np.var(y))) 
metrics = [mean_squared_error, mean_absolute_error, r2_score, nmse]


class PMB_GP:
    '''

    Probabilistic Model Building with Genetic Programming
    '''

    def __init__(self, symbols, grammar, probs, start_symbol, evaluation_metric, X_train, y_train,
                 population_size=20, num_attributes=30, n_evals_tot_max=2000000, alpha=0.1, N=10):

        self.symbols = symbols
        self.grammar = grammar
        
        self.probability_distribution = probs
        self.start_symbol = start_symbol
        self.max_nb_rules = max([len(x) for x in grammar.values()])
        self.evaluation_metric = evaluation_metric
        self.X_train = X_train
        self.y_train = y_train

        self.n_evals_tot = 0
        self.n_evals_tot_max = n_evals_tot_max

        self.alpha = alpha
        self.N = N

        class MyContainer(object):
            # This class does not require the fitness attribute
            # it will be  added later by the creator
            def __init__(self, code, dict_rules_usage):
                # Some initialisation with received values
                self.code = code
                self.dict_rules_usage = dict_rules_usage

            def __len__(self):
                return len(self.code)

        def initIndividual(ind_class, size):
            # ind_class will receive a class inheriting from MyContainer
            grammar_based_structure = []
            symbol = self.start_symbol
            queue = []
            dict_rules_usage = {key: [0 for _ in range(len(value))] for key, value in self.grammar.items()}
            for _ in range(size):

                if symbol is None:
                    grammar_based_structure.append(np.random.randint(0, high=self.max_nb_rules))
                else:
                    nb_symbols = len(self.grammar[symbol])
                    try:
                        i_selected_rule = random.choices(range(nb_symbols), k=1,
                                                         weights=self.probability_distribution[symbol])[0]
                    except Exception as e :
                        print(symbol, e)
                        print(self.grammar[symbol])
                        print(self.probability_distribution[symbol])
                    selected_rule = self.grammar[symbol][i_selected_rule]
                    queue += selected_rule['descendant_symbols']
                    grammar_based_structure.append(i_selected_rule)
                    dict_rules_usage[symbol][i_selected_rule] += 1
                if queue:
                    symbol = queue.pop(0)
                else:
                    symbol = None
            ind = ind_class(grammar_based_structure, dict_rules_usage)
            return ind

        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', MyContainer, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register('individual', initIndividual, creator.Individual, num_attributes)
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register('evaluate', self.evaluate)
        self.algorithm = algorithms.eaSimple

        self.population_size = population_size
        self.population = self.toolbox.population(n=population_size)
        self.elite_size = 1

        self.halloffame = tools.HallOfFame(1)

        # Prevents re-evaluation of the same individuals (assumes determinism)
        self.score_cache = {}

    def express(self, individual):
        expression = [self.start_symbol]
        for branch in individual.code:
            expression_old = expression
            for symbol in self.symbols:
                if symbol in expression:

                    i = expression.index(symbol)
                    rules = self.grammar[symbol]
                    replacement = rules[branch % len(rules)]['value']
                    expression = expression[:i] + replacement + expression[i + 1:]
                    break
            if expression == expression_old:
                break
        return expression

    def evaluate(self, individual):
        expression = "".join(self.express(individual))
        individual.expression = expression
        if expression not in self.score_cache.keys():
            if any([symbol in expression for symbol in self.symbols]):
                score = 0 #float('-inf')
            else:
                try:
                    func = eval("lambda x:" + expression)
                    y_pred = func(self.X_train)
                    score = self.evaluation_metric(y_pred, self.y_train)
                    print(expression, score)
                except Exception as e:
                    score = -1 #float('-inf')
            self.score_cache[expression] = score
        else:
            score = 0
        return score,

    def train(self, generations):
        stats_fit = tools.Statistics(lambda p: p.fitness.values[0])
        stats_fit.register("avg", np.mean)
        stats_fit.register("min", np.min)
        stats_fit.register("max", np.max)
        stats_fit.register("med", np.median)

        #stats_size = tools.Statistics(len)
        #stats_size.register("avg", np.mean)
        mstats = tools.MultiStatistics(fitness=stats_fit)
        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'nevals'] + (mstats.fields if mstats else [])

        print('Start of evolution')

        # Evaluate the entire population
        invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        record = mstats.compile(self.population) if mstats else {}
        self.logbook.record(gen=0, nevals=len(invalid_ind), **record)
        print(self.logbook.stream)

        for g in range(1, generations+1):
            # Update grammar weights
            rules_usage = {key: np.zeros(len(value)) for key, value in self.grammar.items()}

            # # Select N top individuals
            top_N = tools.selBest(self.population, self.elite_size)

            # # Get rules usage
            for ind in top_N:
                for key, value in ind.dict_rules_usage.items():
                    rules_usage[key] += value

            # # Update grammar probs
            for sy, probs in self.probability_distribution.items():

                new_probs = []

                for i, prob_ij in enumerate(probs):
                    new_prob_ij = prob_ij
                    if sum(rules_usage[sy])> 0: 
                        proportion_ij = rules_usage[sy][i] / sum(rules_usage[sy])
                        new_prob_ij = (1 - self.alpha) * prob_ij + self.alpha * proportion_ij
                    new_probs.append(new_prob_ij)

                self.probability_distribution[sy] = np.array(new_probs)

            # Resample a new population
            elite = tools.selBest(self.population, self.elite_size)

            n_evals = self.population_size - self.elite_size
            self.population = self.toolbox.population(n=self.population_size - self.elite_size) + elite
            # Evaluate the entire population
            invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            self.halloffame.update(self.population)

            try:
                record = mstats.compile(self.population) if mstats else {}
                self.logbook.record(gen=g, nevals=n_evals, **record)
            except Exception as e:
                print(e)
                record = {}
                self.logbook.record(gen=g, nevals=n_evals, **record)
            print(self.logbook.stream)

            self.n_evals_tot += n_evals
            if self.n_evals_tot >= self.n_evals_tot_max:
                break

            if 1 - self.halloffame[0].fitness.values[0] < 10**(-7):
                break


def main(arguments):
    dataset, i = arguments
    print("dataset name {} run number {}".format(dataset, i))
    df_train = pd.read_csv(f"../../data/supervised_learning/{dataset}/train.csv").dropna()
    df_test = pd.read_csv(f"../../data/supervised_learning/{dataset}/test.csv").dropna()
    X_train = df_train.drop(columns=["target"])
    X_test = df_test.drop(columns=["target"])
    y_train = df_train.target.values
    y_test = df_test.target.values

    start_symbol = '<e>'
    grammar_file_path = "../../grammars/nguyen_benchmark_v2.bnf"

    probabilistic_grammar = ProbabilisticGrammar(grammar_file_path, start_symbol=start_symbol,
                                                 dataset_n_vars=len(df_train.columns))
    symbols = probabilistic_grammar.symbols
    grammar = probabilistic_grammar.productions_dict
    probs = probabilistic_grammar.probabilities_dict

    optimizer = PMB_GP(symbols, grammar, probs, '<e>',
                       X_train=X_train,
                       y_train=y_train,
                       evaluation_metric=lambda y, yhat: 1 / (1 + mean_squared_error(y, yhat)),
                       population_size=1000,
                       num_attributes=30)
    debut = time.time()
    optimizer.train(generations=2000)
    duree = debut - time.time()

    best_ind = optimizer.halloffame[0]
    expression = "".join(optimizer.express(best_ind))
    best_func = eval("lambda x:" + expression)
    y_pred = best_func(X_test)
     
    scores = [dataset] 
    for m in metrics :
        try : 
            scores += [m(y_test, y_pred)]
        except Exception as e : 
            scores += [e]
    scores += [expression, duree]

    lb_file = f"../results/log_book_pmb_gp_{dataset}_run_{i}_{time.time()}"
    pickle.dump(optimizer.logbook, open(lb_file, 'wb'))
    return scores


if __name__ == '__main__':
    benchmarks = ['nguyen1', 'nguyen2', 'nguyen3', 'nguyen4', 'nguyen5', 'nguyen6', 'nguyen7', 'nguyen8', 'nguyen9','nguyen10', 
                  'keijzer1', 'keijzer2', 'keijzer3','keijzer4', 'keijzer5', 'keijzer6', 'keijzer7', 'keijzer8', 'keijzer9', 'keijzer10', 
                  'keijzer11', 'keijzer12', 'keijzer13', 'keijzer14', 'keijzer15', 'pagie1', 
                  'vladislavleva1', 'vladislavleva2', 'vladislavleva3', 'vladislavleva4', 'vladislavleva5', 'vladislavleva6', 'vladislavleva7', 'vladislavleva8']
    RUNS = 2
    combinaitions = []
    for i in range(RUNS):
        for b in benchmarks:
            combinaitions += [[b, i]]

    with Pool(2) as p:
        results = p.map(main, combinaitions)

        results_df = pd.DataFrame(data=results, columns=['function_name', "mse", "mae", "r2", "nmse", "result", 'time'])
        results_df.to_csv(f"../results/pmb_gp_results_{time.time()}.csv")


