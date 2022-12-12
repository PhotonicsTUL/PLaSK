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
from os.path import dirname as up, join

base = up(up(__file__))
sys.path.insert(0, base)
if len(sys.argv) == 1:
    sys.argv.append(join(base, 'tests', 'plaskgui', 'test.xpl'))

os.environ['PLASKGUI_DEBUG'] = 'true'

try:
    import gui
    gui.main()
except SystemExit as e:
    sys.exit(e.code)
except:
    import traceback as tb
    tb.print_exc()
    sys.stderr.flush()
    sys.exit(1)
