# coding: utf8

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

Shockley = {
    'desc': 'Phenomenological electrical solver based Shockley equation and using finite-element method.',
    'mesh': 'Rectangular2D',
    'conf': (
        ('loop', 'Configuration of the self-consistent loop', (
            ('maxerr', 'Maximum current density error [%]',
                u'Maximum allowed current density error. (float [%])'),)),
        ('matrix', 'Matrix solver', (
            ('algorithm', 'Solution algorithm',
                u'Algorithm used for solving set of linear positive-definite equations.',
                ('cholesky', 'gauss', 'iterative')),
            ('itererr', 'Max. residual error',
                u'Maximum allowed residual error for the iterative algorithm. (float&nbsp;[-])'),
            ('iterlim', 'Max. number of iterations',
                u'Maximum number of iterations for the iterative algorithm. (int)'),
            ('logfreq', 'Progress logging frequency',
                u'Number of iterations after which the progress is logged. (int)'))),
        ('junction', 'Configuration of the effective model of p-n junction.', (
            ('beta#', 'Junction coefficients [1/V]\n(one per line)',
                u'Junction coefficient. This is an inverse of the junction thermal voltage. (float [1/V])'),
            ('js#', 'Reverse bias current densities [A/m<sup>2</sup>]<br/>(one per line)',
                u'Reverse bias current density. (float [A/m<sup>2</sup>])'),
            ('pnjcond', 'Initial junctions conductivity [S/m]',
                u'Initial vertical conductivity of the junctions. (float [S/m])'),
            ('heat', 'Heat generation method',
                u'Method of determination of the heat generated in the junction.', ('joules', 'wavelength')),
            ('wavelength', 'Emitted wavelength',
                u'Emitted wavelength if heat generation method is set to ‘wavelength’.'))),
        ('contacts', 'Properties of the contacts.', (
            ('pcond', 'p-contact conductivity [S/m]',
                u'p-contact conductivity. (float [S/m])'),
            ('ncond', 'n-contact conductivity [S/m]',
                u'n-contact conductivity. (float [S/m])'))),
        ('voltage', 'Electric potential boundary conditions.', None)  # TODO
    ),
    'flow': ('inTemperature', 'outCurrentDensity'),
}

Shockley3D = {
    'desc': 'Phenomenological electrical solver based Shockley equation and using finite-element method.',
    'mesh': 'Rectangular2D',
    'conf': (
        ('loop', 'Configuration of the self-consistent loop', (
            ('maxerr', 'Maximum current density error [%]',
                u'Maximum allowed current density error. (float [%])'),)),
        ('matrix', 'Matrix solver', (
            ('algorithm', 'Solution algorithm',
                u'Algorithm used for solving set of linear positive-definite equations.',
                ('cholesky', 'gauss', 'iterative')),
            ('itererr', 'Max. residual error',
                u'Maximum allowed residual error for the iterative algorithm. (float&nbsp;[-])'),
            ('iterlim', 'Max. number of iterations',
                u'Maximum number of iterations for the iterative algorithm. (int)'),
            ('logfreq', 'Progress logging frequency',
                u'Number of iterations after which the progress is logged. (int)'))),
        ('junction', 'Configuration of the effective model of p-n junction.', (
            ('beta', 'Junction coefficients [1/V]',
                u'Junction coefficient. This is an inverse of the junction thermal voltage. (float [1/V])'),
            ('js', 'Reverse bias current densities [A/m<sup>2</sup>]',
                u'Reverse bias current density. (float [A/m<sup>2</sup>])'),
            ('pnjcond', 'Initial junctions conductivity [S/m]',
                u'Initial vertical conductivity of the junctions. (float [S/m])'),
            ('heat', 'Heat generation method',
                u'Method of determination of the heat generated in the junction.', ('joules', 'wavelength')),
            ('wavelength', 'Emitted wavelength',
                u'Emitted wavelength if heat generation method is set to ‘wavelength’.'))),
        ('contacts', 'Properties of the contacts.', (
            ('pcond', 'p-contact conductivity [S/m]',
                u'p-contact conductivity. (float [S/m])'),
            ('ncond', 'n-contact conductivity [S/m]',
                u'n-contact conductivity. (float [S/m])'))),
        ('voltage', 'Electric potential boundary conditions.', None)  # TODO
    ),
    'flow': ('inTemperature', 'outCurrentDensity'),
}

solvers.register_config('electrical', 'fem', Shockley2D=Shockley, ShockleyCyl=Shockley)
solvers.register_config('electrical', 'fem3d', Shockley3D=Shockley3D)
