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

from gui import solvers

Static = {
    'desc': 'Static thermal solver based on finite-element method.',
    'mesh': 'Rectangular2D',
    'conf': (
        ('loop', 'Self-consistent loop', (
            ('inittemp', 'Initial temperature [K]'),
            ('maxerr', 'Maximum allowed error [K]'))),
        ('matrix', 'Matrix solver', (
            ('algorithm', 'Solution algorithm', ('cholesky', 'gauss', 'iterative')),
            ('itererr', 'Max. iterative residual error'),
            ('iterlim', 'Max. number of iterations'),
            ('logfreq', 'Progress logging frequency'))),
        ('temperature', 'Thermal boundary conditions', None),  # TODO
        ('heatflux', 'Heat flux boundary conditions', None),  # TODO
        ('convection', 'Convective boundary conditions', None),  # TODO
        ('radiation', 'Radiative boundary conditions', None),  # TODO
    ),
    'flow': ('inHeat', 'outTemperature'),
}

solvers.register_config('thermal', Static2D=Static, StaticCyl=Static, Static3D=Static)