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


# plugin: RPSMES Import/Export
# description: Add an import/export of RPSMES *.dan and *.efm files.

from importlib import import_module

import gui

import_dan_operation = import_module('.import', package=__name__).import_dan_operation  # import is not good module name
from .export import export_dan_operation
from .efm import import_efm_operation

if gui.ACTIONS:
    gui.ACTIONS.append(None)
gui.ACTIONS.append(import_efm_operation)
gui.ACTIONS.append(import_dan_operation)

if export_dan_operation is not None:
    gui.ACTIONS.append(export_dan_operation)
