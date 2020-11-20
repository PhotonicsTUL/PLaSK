#!/usr/bin/env python

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