#!/usr/bin/python

import sys
import os
from os.path import dirname as up, join

base = up(up(__file__))
sys.path.insert(0, base)
if len(sys.argv) == 1:
    sys.argv.append(join(base, 'tests', 'gui', 'test.xpl'))

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
