# Copyright (c) 2020-2021, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of RBG2-SR a reinforcement based approach to grammar guided symbolic regression.

import os
import time
import subprocess
import argparse 

parser = argparse.ArgumentParser(description='Parallel Benchmark RG2SR')
parser.add_argument('-run_name', '--n', help="Run name", dest="run_name")
args = parser.parse_args()


command = "python benchmark.py -process_id {} -run_name " + args.run_name
nb_runs = 5
# Run commands in parallel
processes = []

for n in range(nb_runs):
    process = subprocess.Popen(command.format(n), shell=True, env=os.environ)
    time.sleep(10)
    processes.append(process)

# Collect statuses
output = [p.wait() for p in processes]
print(output)
