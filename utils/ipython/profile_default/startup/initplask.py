from __future__ import division

import plask
from plask import *

olask.config.log.use_python()
plask.config.log.output = "stdout"
plask.config.log.colors = "ansi"
plask.__globals = globals()

#switch_backend("module://IPython.zmq.pylab.backend_inline")
rc.savefig.dpi = 96
