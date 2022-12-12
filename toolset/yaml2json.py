#!/usr/bin/env python3
# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.


import sys
import os
import json
import yaml

try:
    infile = sys.argv[1]
except IndexError:
    infile = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))), 'doc', 'schema', 'solvers.yaml')

try:
    outfile = sys.argv[2]
except IndexError:
    outfile = os.path.splitext(infile)[0] + '.json'


data = yaml.safe_load(open(infile))

json.dump(data, open(outfile, 'w'), indent=2)
