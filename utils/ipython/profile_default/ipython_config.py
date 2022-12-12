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

import plask

c = get_config()
c.TerminalInteractiveShell.confirm_exit = False
c.IPKernelApp.pylab = 'inline'
c.IPythonWidget.gui_completion = 'droplist'
c.IPythonWidget.banner = """\
PLaSK {} --- Photonic Laser Simulation Kit
(c) 2014 Lodz University of Technology, Photonics Group

You are entering interactive mode of PLaSK.
Package 'plask' is already imported into global namespace.\
""".format(plask.version)
c.InteractiveShellApp.exec_lines = ['from __future__ import division']

