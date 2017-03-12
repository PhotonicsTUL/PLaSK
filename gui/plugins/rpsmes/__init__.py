# Copyright (C) 2014 Photonics Group, Lodz University of Technology
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of GNU General Public License as published by the
# Free Software Foundation; either version 2 of the license, or (at your
# opinion) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

from importlib import import_module

import gui

import_dan_operation = import_module('.import', package=__name__).import_dan_operation  # import is not good module name
from .export import export_dan_operation
from .efm import import_efm_operation

gui.ACTIONS.append(import_efm_operation)
gui.ACTIONS.append(import_dan_operation)

if export_dan_operation is not None:
    gui.ACTIONS.append(export_dan_operation)
