# Inspired by https://nathankjer.com/evolutionary-algorithms-python/

import sys
sys.path.append(f"./..")
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
from deap import gp
from deap import algorithms
from itertools import product

from grammar_parser import ProbabilisticGrammar
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class EvolutionaryCFG:
    '''
    Evolutionary Context-Free Grammar
    '''

    def __init__(self, symbols, grammar, start_symbol, evaluation_metric, X_train, y_train,
                 population_size=20, num_attributes=30, n_evals_tot_max=2000000):

        self.symbols = symbols
        self.grammar = grammar
        self.start_symbol = start_symbol
        attribute_size = max([len(x) for x in grammar.values()])
        self.evaluation_metric = evaluation_metric
        self.X_train = X_train
        self.y_train = y_train

        self.n_evals_tot = 0
        self.n_evals_tot_max = n_evals_tot_max

        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register('attr_int', random.randint, 0, attribute_size - 1)
        self.toolbox.register('individual', tools.initRepeat, creator.Individual, self.toolbox.attr_int, num_attributes)
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register('evaluate', self.evaluate)
        self.toolbox.register('mate', tools.cxTwoPoint)
        self.toolbox.register('mutate',
                              tools.mutUniformInt,
                              low=0,
                              up=attribute_size-1,
                              indpb=0.1)
        self.toolbox.register('select', tools.selTournament, tournsize=3)
        self.algorithm = algorithms.eaSimple

        self.population = self.toolbox.population(n=population_size)
        self.mate_prob = 0.95
        self.mutant_prob = 0.05
        self.elite_size = 10

        self.halloffame = tools.HallOfFame(1)

        # Prevents re-evaluation of the same individuals (assumes determinism)
        self.score_cache = {}

    def express(self, individual):
        complexity = 0
        expression = [self.start_symbol]
        for branch in individual:
            expression_old = expression
            for symbol in self.symbols:
                if symbol in expression:

                    i = expression.index(symbol)
                    rules = self.grammar[symbol]
                    replacement = rules[branch % len(rules)]['value']
                    expression = expression[:i] + replacement + expression[i + 1:]
                    complexity+=1
                    break
            if expression == expression_old:
                break
        return expression, complexity

    def evaluate(self, individual):
        expression, _ = self.express(individual)
        expression = "".join(expression)
        individual.expression = expression
        if expression not in self.score_cache.keys():
            if any([symbol in expression for symbol in self.symbols]):
                score = 0 #float('-inf')
            else:
                try:
                    func = eval("lambda x:" + expression)
                    y_pred = func(self.X_train)
                    score = self.evaluation_metric(self.y_train, y_pred)
                    if score ==1 : 
                        print(self.dataset, expression, self.y_train, y_pred)
                except Exception as e:
                    #print(e)
                    score = -1 #float('-inf')
            self.score_cache[expression] = score
        else:
            score = 0
        if np.isnan(score): 
            score = 0
        return score,

    def train(self, generations):

        stats_fit = tools.Statistics(lambda p: p.fitness.values[0])
        stats_fit.register("avg", np.mean)
        stats_fit.register("min", np.min)
        stats_fit.register("max", np.max)
        stats_fit.register("med", np.median)

        stats_size = tools.Statistics(len)
        stats_size.register("avg", np.mean)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
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
            elite = tools.selBest(self.population, self.elite_size)
            parents = self.toolbox.select(self.population, len(self.population))

            offspring = []
            n_evals = 0
            while len(offspring) < len(self.population) - self.elite_size:

                # Randomly choose two parents from the parent population.
                child1, child2 = list(map(self.toolbox.clone, random.sample(parents, 2)))

                if random.random() < self.mate_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

                for child in [child1, child2]:
                    if random.random() < self.mutant_prob:
                        self.toolbox.mutate(child)

                    fitness = self.toolbox.evaluate(child)
                    
                    n_evals += 1

                    if fitness[0] <= 0:
                        continue
                    else:
                        child.fitness.values = fitness
                        offspring.append(child)

                if n_evals >= self.n_evals_tot_max:
                    break

            self.population.sort(key=lambda x: x.fitness, reverse=True)

            offspring = offspring[: len(self.population) - self.elite_size]
            self.population[-len(offspring):] = offspring
            self.halloffame.update(self.population)

            try:
                record = mstats.compile(self.population) if mstats else {}
            except:
                record = {}
            self.logbook.record(gen=g, nevals=n_evals, **record)
            print(self.logbook.stream)

            self.n_evals_tot += n_evals
            if self.n_evals_tot >= self.n_evals_tot_max:
                break

            if 1 - self.halloffame[0].fitness.values[0] < 10**(-7):
                break


def main(inputs):
    dataset, i = inputs
    print("dataset name {} run number {}".format(dataset, i))
    df_train = pd.read_csv(f"../../data/supervised_learning/{dataset}/train.csv").dropna()
    df_test = pd.read_csv(f"../../data/supervised_learning/{dataset}/test.csv").dropna()
    X_train = df_train.drop(columns=["target"])
    X_test = df_test.drop(columns=["target"])
    y_train = df_train.target.values
    y_test = df_test.target.values
    var_y = y_train.var()

    start_symbol = '<e>'
    grammar_file_path = "../../grammars/nguyen_benchmark_v2.bnf"

    probabilistic_grammar = ProbabilisticGrammar(grammar_file_path, start_symbol=start_symbol,
                                                 dataset_n_vars=len(X_train.columns))
    symbols = probabilistic_grammar.symbols
    grammar = probabilistic_grammar.productions_dict
    
    nrmse = lambda y, yhat: 1 / (1 + mean_squared_error(y, yhat)/var_y)
    optimizer = EvolutionaryCFG(symbols, grammar, '<e>',
                                X_train=X_train,
                                y_train=y_train,
                                evaluation_metric=nrmse,
                                population_size=1000,
                                num_attributes=50)
    debut = time.time()
    optimizer.train(generations=2000)
    duree = debut - time.time()

    best_ind = optimizer.halloffame[0]
    expression, complexity = "".join(optimizer.express(best_ind))
    print("dataset name {} run number {} best expression {}".format(dataset, i, expression))
    best_func = eval("lambda x:" + expression)
    y_pred = best_func(X_test)
    
    var_y = y_test.var()
    nrmse = lambda y, yhat: mean_squared_error(y, yhat)/var_y
    metrics = [mean_squared_error, mean_absolute_error, r2_score, nrmse]

    scores = [dataset] 
    for m in metrics: 
        try : 
            scores += [m(y_test, y_pred)]
        except Exception as e: 
            scores += [e]
    
    scores += [expression, duree, complexity]

    lb_file = f"../results/log_book_{dataset}_run_{i}_{time.time()}"
    pickle.dump(optimizer.logbook, open(lb_file, 'wb'))

    return scores


if __name__ == '__main__':
    benchmarks = ['nguyen1', 'nguyen2', 'nguyen3', 'nguyen4', 'nguyen5', 'nguyen6', 'nguyen7', 'nguyen8', 'nguyen9','nguyen10', 
                  'keijzer1', 'keijzer2', 'keijzer3','keijzer4', 'keijzer5', 'keijzer6', 'keijzer7', 'keijzer8', 'keijzer9', 'keijzer10', 
                  'keijzer11', 'keijzer12', 'keijzer13', 'keijzer14', 'keijzer15', 'pagie1', 
                  'vladislavleva1', 'vladislavleva2', 'vladislavleva3', 'vladislavleva4', 'vladislavleva5', 'vladislavleva6', 'vladislavleva7', 'vladislavleva8']
    benchmarks= ['nguyen1']
    RUNS = 1

    combinaitions = []
    for i in range(RUNS): 
        for b in benchmarks: 
            combinaitions +=[[b, i]]
    
    with Pool(1) as p:
        print('Starting computation')
        results = p.map(main, combinaitions)
        print('Gathering results')
        results_df = pd.DataFrame(data=results, columns=['function_name', "mse", "mae", "r2", "nrmse", "result", 'time', 'complexity'])
        results_df.to_csv("../results/g3p_results_missing_functions.csv")
