#!/usr/bin/python

import sys
from os.path import dirname as up

sys.path.insert(0, up(up(__file__)))

import gui

gui.main()
