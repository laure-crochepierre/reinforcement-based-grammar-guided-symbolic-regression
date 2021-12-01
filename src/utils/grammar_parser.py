# Copyright (c) 2020-2021, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of RBG2-SR a reinforcement based approach to grammar guided symbolic regression.

import re
import os
import numpy as np

# for grammars
SYMBOL_SEP = '::='
RULE_SEP = '|'
DELIMITER = '\n'

from sklearn.preprocessing import OneHotEncoder


class ProbabilisticGrammar:
    def __init__(self, grammar_file_path, start_symbol=None, dataset_n_vars=None):
        self.start_symbol = start_symbol
        self.dataset_n_vars = dataset_n_vars
        self.terminals = []
        self.n_discrete_actions = 0
        self.n_symbols = 0
        self.max_brother_symbols = 0
        self.productions_list = []
        self.symbols_to_mask = {}

        file_format = grammar_file_path.split(".")[-1]
        if file_format == "bnf":
            with open(grammar_file_path, 'r') as f:
                self.symbols,  self.productions_dict, self.probabilities_dict = self.parse(f)
            for parent_symbol, rules in self.productions_dict.items():
                self.n_discrete_actions += len(rules)
                self.n_symbols += 1
                for rule in rules:
                    rule['parent_symbol'] = parent_symbol

                    if len(rule['descendant_symbols']) > self.max_brother_symbols+1:
                        self.max_brother_symbols = len(rule['descendant_symbols'])

            def compute_distance_to_terminal(symbol=self.start_symbol, recursive=False, counter=10):
                distances_to_this_symbol = []
                # Compute distance to terminal
                for r in self.productions_dict[symbol]:
                    if r['distance_to_terminal'] != np.inf:
                        distances_to_this_symbol.append(r['distance_to_terminal'])
                    else:
                        distances = []
                        for d_s in r['descendant_symbols']:
                            if d_s == symbol:
                                if recursive:
                                    distance = 0
                                    if counter > 0:
                                        distance = compute_distance_to_terminal(d_s, counter=counter-1)
                                    distances.append(distance)
                                else :
                                    distances.append(np.inf)
                            elif r['type'] == 'NT':
                                distance = 0
                                if counter >0:
                                    distance = compute_distance_to_terminal(d_s, recursive=True, counter=counter-1)

                                distances.append(distance)
                            else:
                                raise TypeError('Unknown grammatical rule type')

                        distance_to_terminal = min(r['distance_to_terminal'], sum(distances))
                        r['distance_to_terminal'] = distance_to_terminal
                        distances_to_this_symbol.append(distance_to_terminal)

                if not recursive:
                    distances_to_this_symbol = [d for d in distances_to_this_symbol if d != np.inf]

                res = min(distances_to_this_symbol) + 1
                return res

            final_depth = compute_distance_to_terminal(recursive=True)

            i_from = 0
            i_to = 0
            for parent_symbol, rules in self.productions_dict.items():
                for rule in rules:
                    self.productions_list.append(rule)

                i_to += len(rules)
                self.symbols_to_mask[parent_symbol] = np.zeros((self.n_discrete_actions, ))
                self.symbols_to_mask[parent_symbol][i_from:i_to] = 1

                i_from = i_to

        else:
            raise ValueError('Invalid file format')

        encoders = {"symbols": OneHotEncoder(sparse=False, handle_unknown="ignore").fit([[s] for s in self.symbols + ['#']]),
                    "actions": OneHotEncoder(sparse=False, handle_unknown="ignore").fit([[p['raw']] for p in
                                                                                         self.productions_list] + [["#"]])
                    }
        self.all_symbols = self.symbols + ["#"]
        self.all_actions = [p['raw'] for p in self.productions_list] + ['#']
        self.symbol_encoding = {s : encoders["symbols"].transform([[s]]) for s in self.all_symbols}
        self.action_encoding = {p : encoders["actions"].transform([[p]]) for p in self.all_actions}

    def parse(self, f, symbol_sep=SYMBOL_SEP, rule_sep=RULE_SEP):
        symbols = []
        productions_dict = {}
        probabilities_dict = {}
        for i, line in enumerate(f.readlines()):

            if '::=' in line:
                symbol, rules_str = line.split(symbol_sep)
                symbol = symbol.strip()
                if (i == 0) & (self.start_symbol is None):
                    self.start_symbol = symbol

                rules_split = rules_str.split(rule_sep)

                # generate initial probabilities
                probabilities = [1 for _ in range(len(rules_split))]
                if "probs" in rules_split[-1]:
                    probs = rules_split.pop(-1)
                    probs = eval(probs.replace('probs', '').strip())

                    if len(probs) == len(rules_split):
                        probabilities = probs

                probabilities = [p/sum(probabilities) for p in probabilities]

                # generate rules
                rules = []
                for r in rules_split:
                    rules_dicts, found_terminals = self.prettify_rule(r, symbol)
                    rules += rules_dicts
                    self.terminals += found_terminals

                    # generate initial probabilities
                    probabilities = [1 for _ in range(len(rules))]
                    if "probs" in rules_split[-1]:
                        probs = rules_split.pop(-1)
                        probs = eval(probs.replace('probs', '').strip())

                        if len(probs) == len(rules):
                            probabilities = probs

                    probabilities = [p / sum(probabilities) for p in probabilities]

                symbols.append(symbol)
                productions_dict[symbol] = rules
                probabilities_dict[symbol] = probabilities

        return symbols, productions_dict, probabilities_dict

    def prettify_rule(self, r, symbol):
        #r = r.strip()
        rule_dict = {"raw": str(r), "type": "NT", "value": [], 'n_descendants': 0, 'descendant_symbols': [],
                     'recursive': False, "distance_to_terminal": np.inf}
        rules_dicts = []
        terminals = []
        if ('<' in r) & ('>' in r):
            rule_dict['n_descendants'] = r.count("<")

            descendant_symbols = re.findall('<.*?>', r)
            rule_dict["descendant_symbols"] = descendant_symbols
            if symbol in rule_dict['descendant_symbols']:
                rule_dict['recursive'] = True

            rule_dict['value'] = [el for el in re.split(r'(<.+?>)', r.replace(' ', ''), flags=re.A) if el != '']

            rules_dicts.append(rule_dict)
        else:
            rule_dict["type"] = "T"
            rule_dict['distance_to_terminal'] = 1
            if "GE_RANGE" in r:
                dataset_n_vars = self.dataset_n_vars
                stop = eval(r.split(":")[-1])
                for i in np.arange(start=0, stop=stop, step=1):
                    current_rule_dict = rule_dict.copy()
                    current_rule_dict["value"] = [str(i)]
                    current_rule_dict['raw'] = str(i)
                    terminals.append(str(i))
                    rules_dicts.append(current_rule_dict)
            else:
                rule_dict['value'] = [r]
                terminals.append(r)
                rules_dicts.append(rule_dict)

        return rules_dicts, terminals


if __name__ == "__main__":
    #g = ProbabilisticGrammar(grammar_file_path="../grammars/power_system_exemple.bnf")
    g = ProbabilisticGrammar(grammar_file_path="../grammars/nguyen_benchmark_v2.bnf",
                             start_symbol="<e>",
                             dataset_n_vars=1)
    print(g.productions_dict['<e>'])

    print()
    print('Terminals')
    print(g.terminals)


