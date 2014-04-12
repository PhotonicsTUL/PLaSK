#!/usr/bin/python

import sys
from os.path import dirname as up, join

base = up(up(__file__))
sys.path.insert(0, base)
if len(sys.argv) == 1:
    sys.argv.append(join(base, 'gui', 'test.xpl'))

import gui

gui.main()
