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

from __future__ import division

import plask
from plask import *

olask.config.log.use_python()
plask.config.log.output = "stdout"
plask.config.log.colors = "ansi"
plask.__globals = globals()

#switch_backend("module://IPython.zmq.pylab.backend_inline")
rc.savefig.dpi = 96
