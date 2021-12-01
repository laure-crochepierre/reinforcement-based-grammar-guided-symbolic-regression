# Copyright (c) 2020-2021, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of RBG2-SR a reinforcement based approach to grammar guided symbolic regression.
from copy import deepcopy
import numpy as np


class Constraints(object):
    def __init__(self, grammar, max_horizon, min_horizon):
        self.grammar = grammar
        self.start_symbol = self.grammar.start_symbol
        self.nb_actions = grammar.n_discrete_actions
        self.max_horizon = max_horizon
        self.min_horizon = min_horizon

        self.constraints_dict = {"min": {},
                                 "max": {},
                                 "trig": {}}

    def init_constraint(self):
        mask = np.ones((self.nb_actions,), dtype=np.float32)
        return mask

    def make_min_constraint(self, c, symbol, size):
        if (symbol, size) in self.constraints_dict['min'].keys():
            return np.multiply(c, self.constraints_dict['min'][(symbol, size)])

        else:
            for i, rule in enumerate(self.grammar.productions_list):
                if (c[i] == 0) or (symbol != rule['parent_symbol']):
                    c[i] = 0
                    continue

                distance_to_terminal = rule['distance_to_terminal']
                if (size + distance_to_terminal < self.min_horizon) and (not rule['recursive']):
                    c[i] = 0
            self.constraints_dict['min'][(symbol, size)] = deepcopy(c)
        return c

    def make_max_constraint(self, c, symbol, size):
        if (symbol, size) in self.constraints_dict['max'].keys():
            return np.multiply(c, self.constraints_dict['max'][(symbol, size)])

        else:

            for i, rule in enumerate(self.grammar.productions_list):
                if (c[i] == 0) or (symbol != rule['parent_symbol']):
                    continue

                distance_to_terminal = rule['distance_to_terminal']
                if size + distance_to_terminal > self.max_horizon:
                    c[i] = 0
            self.constraints_dict['max'][(symbol, size)] = deepcopy(c)
        return c

    def make_min_max_constraint(self, c, symbol, size):
        c = self.make_min_constraint(c, symbol, size)
        c = self.make_max_constraint(c, symbol, size)
        return c

    def make_trig_constraint(self, m, symbol, translation):
        if symbol in self.constraints_dict['trig'].keys():
            return np.multiply(m, self.constraints_dict['trig'][symbol])

        else:
            for i, rule in enumerate(self.grammar.productions_list):
                if (m[i] == 0) or (symbol != rule['parent_symbol']):
                    continue

                previous_rule = [r for r in previous_rules if not '<' in r]
                if len(previous_rule) == 0:
                    continue
                if (previous_rule[-1] in ['np.sin', 'np.cos']) & (rule['raw'] in ['np.sin', 'np.cos']):
                    m[i] = 0
            self.constraints_dict['trig'][symbol] = deepcopy(m)
        return m