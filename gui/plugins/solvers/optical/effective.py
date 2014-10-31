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

EffectiveIndex2D = {
    'desc': 'Scalar optical solver based on the effective index method.',
    'mesh': 'Rectangular2D',
    'conf': (
        ('mode', 'Mode properties', (
            ('polarization', 'Polarization',
             u'Longitudnally-propagating mode polarization (TE or TM).', ('TE', 'TM')),
            ('wavelength', 'Wavelength',
             u'Mode wavelength [nm].'),
            ('vat', u'Vertical solution coord [µm]',
             u'Horizontal position of at which the vertical part of the field is calculated. (float&nbsp;[µm])'))),
        ('root', 'Parameters of the global root-finding algorithm', (
            ('method', 'Root finding method',
             u'Root finding algorithm (Muller\'s method or Broyden\'s method.', ('muller', 'broyden')),
            ('tolx', 'Tolerance on effective index',
             u'Maximum change of the effective index which is allowed for onvergent solution. (float)'),
            ('tolf-min', 'Sufficient determinant value',
             u'Minimum value of the determinant sufficient to assume convergence. (float&nbsp;[a.u.])'),
            ('tolf-max', 'Rquired determinant value',
             u'Maximum value of the determinant required to assume convergence. (float&nbsp;[a.u.])'),
            ('maxstep', 'Maximum Broyden iteration step',
             u'Maximum step in one iteration of root finding. Significant for the Broyden\'s method only. '
             '(float&nbsp;[a.u.])'),
            ('maxiter', 'Maximum number of iterations', 'Maximum number of root finding iterations. (int)'),
            ('alpha', 'Broyden decrease factor',
             u'Parameter ensuring sufficient decrease of determinant in each step (Broyden method only). '
             '(float&nbsp;[a.u.])'),
            ('lambda', 'Minimum Broyden step decrease ratio',
             u'Minimum decrease ratio of one step (Broyden method only). (float&nbsp;[a.u.])'),
            ('initial-range', 'Initial Muller range size',
             u'Initial range size (Muller method only).  (complex [a.u.])'))),
        ('stripe-root', 'Parameters of the vertical root-finding algorithm', (
            ('method', 'Root finding method',
             u'Root finding algorithm (Muller\'s method or Broyden\'s method.', ('muller', 'broyden')),
            ('tolx', 'Tolerance on effective index',
             u'Maximum change of the effective index which is allowed for onvergent solution. (float)'),
            ('tolf-min', 'Sufficient determinant value',
             u'Minimum value of the determinant sufficient to assume convergence. (float&nbsp;[a.u.])'),
            ('tolf-max', 'Rquired determinant value',
             u'Maximum value of the determinant required to assume convergence. (float&nbsp;[a.u.])'),
            ('maxstep', 'Maximum Broyden iteration step',
             u'Maximum step in one iteration of root finding. Significant for the Broyden\'s method only. '
             '(float&nbsp;[a.u.])'),
            ('maxiter', 'Maximum number of iterations', 'Maximum number of root finding iterations. (int)'),
            ('alpha', 'Broyden decrease factor',
             u'Parameter ensuring sufficient decrease of determinant in each step (Broyden method only). '
             '(float&nbsp;[a.u.])'),
            ('lambda', 'Minimum Broyden step decrease ratio',
             u'Minimum decrease ratio of one step (Broyden method only). (float&nbsp;[a.u.])'),
            ('initial-range', 'Initial Muller range size',
             u'Initial range size (Muller method only).  (complex [a.u.])'))),
        ('mirrors', 'Mirror losses', (
            ('R1', 'Front reflectivity', 'Reflectivity of the front mirror.'),
            ('R2', 'Back reflectivity', 'Reflectivity of the back mirror.')))
    ),
    u'flow': ('inTemperature', 'inGain',
             u'outLightMagnitude', 'outRefractiveIndex', 'outHeat', 'outLoss', 'outWavelenght'),
}

EffectiveFrequencyCyl = {
    'desc': 'Scalar optical solver based on the frequency index method.',
    'mesh': 'Rectangular2D',
    'conf': (
        ('mode', 'Mode properties', (
            ('lam0', 'Approximate wavelength [nm]',
             u'Approximate wavelength, around which the solution is searched. '
             'The refractive and group idexes are computed for this wavelength. (float&nbsp;[nm])'),
            ('emission', 'Direction of emission',
             u'Direction of the useful light emission. Necessary for the over-threshold model to correctly compute'
             'the output power.', ('top', 'bottom')),
            ('vlam', 'Vertical wavelength [nm]',
             u'"Vertical wavelength" i.e. the wavelength what would be in the absence of lateral confinement; '
             'setting this value helps to find models in very long resonators (float [nm])'),
            ('vat', u'Vertical solution coord [µm]',
             u'Horizontal position of at which the vertical part of the field is calculated. (float&nbsp;[µm])'))),
        ('root', 'Parameters of the global root-finding algorithm', (
            ('method', 'Root finding method',
             u'Root finding algorithm (Muller\'s method or Broyden\'s method.', ('muller', 'broyden')),
            ('tolx', 'Tolerance on effective index',
             u'Maximum change of the effective frequency parameter which is allowed for convergent solution. (float)'),
            ('tolf-min', 'Sufficient determinantfloat [ value',
             u'Minimum value of the determinant sufficient to assume convergence. (float&nbsp;[a.u.])'),
            ('tolf-max', 'Rquired determinant value',
             u'Maximum value of the determinant required to assume convergence. (float&nbsp;[a.u.])'),
            ('maxstep', 'Maximum Broyden iteration step',
             u'Maximum step in one iteration of root finding. Significant for the Broyden\'s method only. '
             '(float&nbsp;[a.u.])'),
            ('maxiter', 'Maximum number of iterations',
             u'Maximum number of root finding iterations. (int)'),
            ('alpha', 'Broyden decrease factor',
             u'Parameter ensuring sufficient decrease of determinant in each step (Broyden method only). '
             '(float&nbsp;[a.u.])'),
            ('lambda', 'Minimum Broyden step decrease ratio',
             u'Minimum decrease ratio of one step (Broyden method only). (float&nbsp;[a.u.])'),
            ('initial-range', 'Initial Muller range size',
             u'Initial range size (Muller method only).  (complex [a.u.])'))),
        ('stripe-root', 'Parameters of the vertical root-finding algorithm', (
            ('method', 'Root finding method',
             u'Root finding algorithm (Muller\'s method or Broyden\'s method.', ('muller', 'broyden')),
            ('tolx', 'Tolerance on effective index',
             u'Maximum change of the effective index which is allowed for convergent solution. (float)'),
            ('tolf-min', 'Sufficient determinant value',
             u'Minimum value of the determinant sufficient to assume convergence. (float&nbsp;[a.u.])'),
            ('tolf-max', 'Rquired determinant value',
             u'Maximum value of the determinant required to assume convergence. (float&nbsp;[a.u.])'),
            ('maxstep', 'Maximum Broyden iteration step',
             u'Maximum step in one iteration of root finding. Significant for the Broyden\'s method only. '
             '(float&nbsp;[a.u.])'),
            ('maxiter', 'Maximum number of iterations', 'Maximum number of root finding iterations. (int)'),
            ('alpha', 'Broyden decrease factor',
             u'Parameter ensuring sufficient decrease of determinant in each step (Broyden method only). '
             '(float&nbsp;[a.u.])'),
            ('lambda', 'Minimum Broyden step decrease ratio',
             u'Minimum decrease ratio of one step (Broyden method only). (float&nbsp;[a.u.])'),
            ('initial-range', 'Initial Muller range size',
             u'Initial range size (Muller method only).  (complex [a.u.])'))),
    ),
    'flow': ('inTemperature', 'inGain',
              'outLightMagnitude', 'outRefractiveIndex', 'outHeat', 'outLoss', 'outWavelenght'),
}

solvers.register_config('optical', 'effective',
                        EffectiveIndex2D=EffectiveIndex2D,
                        EffectiveFrequencyCyl=EffectiveFrequencyCyl)
