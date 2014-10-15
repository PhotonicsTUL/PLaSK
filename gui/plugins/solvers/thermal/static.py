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
    u'mesh': 'Rectangular2D',
    u'conf': (
        ('loop', 'Self-consistent loop', (
            ('inittemp', 'Initial temperature [K]',
             u'Initial temperature used for the first computation. (float&nbsp;[K])'),
            ('maxerr', 'Maximum allowed error [K]',
             u'Maximum allowed error in a self-consistent loop. (float&nbsp;[K])'))),
        ('matrix', 'Matrix solver', (
            ('algorithm', 'Solution algorithm',
             u'Algorithm used for solving set of linear positive-definite equations.', ('cholesky', 'gauss', 'iterative')),
            ('itererr', 'Max. residual error',
             u'Maximum allowed residual error for the iterative algorithm. (float&nbsp;[-])'),
            ('iterlim', 'Max. number of iterations',
             u'Maximum number of iterations for the iterative algorithm. (int)'),
            ('logfreq', 'Progress logging frequency',
             u'Number of iterations after which the progress is logged. (int)'))),
        ('temperature', 'Thermal boundary conditions', None),  # TODO
        ('heatflux', 'Heat flux boundary conditions', None),  # TODO
        ('convection', 'Convective boundary conditions', None),  # TODO
        ('radiation', 'Radiative boundary conditions', None),  # TODO
    ),
    u'flow': ('inHeat', 'outTemperature'),
}

solvers.register_config('thermal', Static2D=Static, StaticCyl=Static, Static3D=Static)