from __future__ import division

import plask
from plask import *

plask.config.log.output = "stdout"
plask.config.log.coloring = "ansi"
plask.__globals = globals()

#switch_backend("module://IPython.zmq.pylab.backend_inline")
rc.savefig.dpi = 96
