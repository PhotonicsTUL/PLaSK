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

from .model import solvers as model
from .controller import solvers as controller


def register_config(category, **kwargs):
    for solver,config in kwargs.items():
        model.SOLVERS[category,solver] = model.ConfSolverFactory(category, solver, config)
