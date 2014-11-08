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

Fourier2D = {
    'desc': 'Vectorial optical solver based on the plane-wave reflection transfer method.',
    'mesh': None,
    'conf': (
        ('expansion', 'Details on Fourier expansion', (
            ('size', 'Expansion size',
                u'Expansion size. (integer)'),
            ('refine', 'Averaging points',
                u'Number of refinement points for refractive index averaging. (integer)'),
            ('smooth', 'Smoothing parameter',
                u'Smoothing parameter for material boundaries (increases convergence). (float)'),
            ('group-layers', 'Layers grouping',
                u'Should similar layers be grouped for better performance. (bool)', ('yes', 'no')))),
        ('interface', 'Matching interface position in the stack '
                      '<span style="font-weight: normal;">(set only one)</span>', (
            ('index', 'Layer index',
                u'Index of the layer, below which the interface is located. (integer)'),
            ('position', 'Position',
                u'Interface will be located as close as possible to the vertical coordinate specified '
                u'in this attribute. (float)'),
            ('object', 'Object name',
                u'Name of the geometry object below which the interface is located.'),
            ('path', 'Object path',
                u'Optional path name, specifying particular instance of the object given in the object attribute.'))),
        ('vpml', 'Vertical PMLs', (
            ('factor', 'Scaling factor',
                u'PML scaling factor. (complex)'),
            ('shift', u'Distance [µm]',
                u'PML shift from the structure. (float [µm])'),
            ('size', u'Size [µm]',
                u'PML size. (float [µm])'))),
        ('pml', 'Horizontal PMLs', (
            ('factor', 'Scaling factor',
                u'PML scaling factor. (complex)'),
            ('order', 'Shape order',
                u'PML shape order (0 → flat, 1 → linearly increasing, 2 → quadratic, etc.). (float)'),
            ('shift', u'Distance [µm]',
                u'PML shift from the structure. (float [µm])'),
            ('size', u'Size [µm]',
                u'PML size. (float [µm])'))),
        ('mode', 'Mode properties', (
            ('wavelength', 'Wavelength [nm]',
                u'Light wavelength. (float [nm])'),
            ('k-tran', u'Transverse wave-vector [1/µm]',
                u'Transverse wave-vector component. (float [1/µm])'),
            ('k-long', u'Longitudinal wave-vector [1/µm]',
                u'Longitudinal wave-vector component. (float [1/µm])'),
            ('symmetry', 'Mode symmetry',
                u'Mode symmetry. Specify a symmetric field component here (e.g. ``Etran``, ``Hx``).',
                ('Etran', 'Elong')),
            ('polarization', 'Mode polarization',
                u'Mode polarization. Give an existing field component here (e.g. ``Etran``, ``Hx``).',
                ('Etran', 'Elong')))),
        ('root', 'Parameters of the root-finding algorithm', (
            ('method', 'Root finding method',
             u'Root finding algorithm (Muller\'s method or Broyden\'s method.', ('muller', 'broyden', 'brent')),
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
            ('initial-range', 'Initial range size',
             u'Initial range size (Muller and Brent methods only).  (complex [a.u.])'))),
        ('mirrors', 'Mirror losses', (
            ('R1', 'Front reflectivity', 'Reflectivity of the front mirror.'),
            ('R2', 'Back reflectivity', 'Reflectivity of the back mirror.')))
    ),
    u'flow': ('inTemperature', 'inGain',
             u'outLightMagnitude', 'outRefractiveIndex', 'outHeat', 'outLoss', 'outWavelenght'),
}

Fourier3D = {
    'desc': 'Vectorial optical solver based on the plane-wave reflection transfer method.',
    'mesh': None,
    'conf': (
        ('expansion', 'Details on Fourier expansion', (
            ('size', 'Expansion sizes',
                u'Expansion sizes along longitudinal and transverse directions.\n\n'
                u'You cannot set this attribute together with ‘Longitudinal expansion size’ and'
                u'‘Transverse expansion size’. (integer)'),
            ('size-long', 'Longitudinal expansion size',
                u'Expansion size along longitudinal axis.\n\n'
                u'You cannot set this attribute together with ‘Expansion sizes’. (integer)'),
            ('size-tran', 'Transverse expansion size',
                u'Expansion size along transverse axis.\n\n'
                u'You cannot set this attribute together with ‘Expansion sizes’. (integer)'),
            ('refine', 'Averaging points',
                u'Number of refinement points for refractive index averaging '
                u'along longitudinal and transverse directions.\n\n'
                u'You cannot set this attribute together with ‘Longitudinal averaging points’ and'
                u'‘Transverse averaging points’. (integer)'),
            ('refine-long', 'Longitudinal averaging points',
                u'Number of refinement points for refractive index averaging along longitudinal axis.\n\n'
                u'You cannot set this attribute together with ‘Averaging points’. (integer)'),
            ('refine-tran', 'Transverse averaging points',
                u'Number of refinement points for refractive index averaging along transverse axis.\n\n'
                u'You cannot set this attribute together with ‘Averaging points’. (integer)'),
            ('smooth', 'Smoothing parameter',
                u'Smoothing parameter for material boundaries (increases convergence). (float)'),
            ('group-layers', 'Layers grouping',
                u'Should similar layers be grouped for better performance. (bool)', ('yes', 'no')))),
        ('interface', 'Matching interface position in the stack '
                      '<span style="font-weight: normal;">(set only <i>Layer index</i>, <i>Position</i> '
                      'or <i>Object</i>)</span>', (
            ('index', 'Layer index',
                u'Index of the layer, below which the interface is located. (integer)'),
            ('position', 'Position',
                u'Interface will be located as close as possible to the vertical coordinate specified '
                u'in this attribute. (float)'),
            ('object', 'Object name',
                u'Name of the geometry object below which the interface is located.'),
            ('path', 'Object path',
                u'Optional path name, specifying particular instance of the object given in the object attribute.'))),
        ('vpml', 'Vertical PMLs', (
            ('factor', 'Scaling factor',
                u'PML scaling factor. (complex)'),
            ('shift', u'Distance [µm]',
                u'PML shift from the structure. (float [µm])'),
            ('size', u'Size [µm]',
                u'PML size. (float [µm])'))),
        ('pmls', 'Longitudinal and transverse PMLs', (
            ('factor', 'Scaling factor',
                u'PML scaling factor. (complex)'),
            ('order', 'Shape order',
                u'PML shape order (0 → flat, 1 → linearly increasing, 2 → quadratic, etc.). (float)'),
            ('shift', u'Distance [µm]',
                u'PML shift from the structure. (float [µm])'),
            ('size', u'Size [µm]',
                u'PML size. (float [µm])'))),
        ('pmls/long', u'Longitudinal PML '
                      u'<span style="font-weight: normal;">(overrides ‘Longitudinal and transverse PMLs’)</span>', (
            ('factor', 'Scaling factor',
                u'PML scaling factor. (complex)'),
            ('order', 'Shape order',
                u'PML shape order (0 → flat, 1 → linearly increasing, 2 → quadratic, etc.). (float)'),
            ('shift', u'Distance [µm]',
                u'PML shift from the structure. (float [µm])'),
            ('size', u'Size [µm]',
                u'PML size. (float [µm])'))),
        ('pmls/tran', u'Transverse PML '
                      u'<span style="font-weight: normal;">(overrides ‘Longitudinal and transverse PMLs’)</span>', (
            ('factor', 'Scaling factor',
                u'PML scaling factor. (complex)'),
            ('order', 'Shape order',
                u'PML shape order (0 → flat, 1 → linearly increasing, 2 → quadratic, etc.). (float)'),
            ('shift', u'Distance [µm]',
                u'PML shift from the structure. (float [µm])'),
            ('size', u'Size [µm]',
                u'PML size. (float [µm])'))),
        ('mode', 'Mode properties', (
            ('wavelength', 'Wavelength [nm]',
                u'Light wavelength. (float [nm])'),
            ('k-tran', u'Transverse wave-vector [1/µm]',
                u'Transverse wave-vector component. (float [1/µm])'),
            ('k-long', u'Longitudinal wave-vector [1/µm]',
                u'Longitudinal wave-vector component. (float [1/µm])'),
            ('symmetry', 'Mode symmetries',
                u'Mode symmetries along longitudinal and transverse directions. '
                u'Specify symmetric field components here (e.g. ``Etran``, ``Hx``).\n\n'
                u'You cannot set this attribute together with ‘Longitudinal mode symmetry’ '
                u'and Transverse mode symmetry’.', ('Etran', 'Elong')),
            ('symmetry-long', 'Longitudinal mode symmetry',
                u'Mode symmetry along longitudinal axis. Specify a symmetric field component here '
                u'(e.g. ``Etran``, ``Hx``).\n\nYou cannot set this attribute together with ‘Mode symmetries’.',
                ('Etran', 'Elong')),
            ('symmetry-tran', 'Transverse mode symmetry',
                u'Mode symmetry along transverse axis. Specify a symmetric field component here '
                u'(e.g. ``Etran``, ``Hx``).\n\nYou cannot set this attribute together with ‘Mode symmetries’.',
                ('Etran', 'Elong')))),
        ('root', 'Parameters of the root-finding algorithm', (
            ('method', 'Root finding method',
             u'Root finding algorithm (Muller\'s method or Broyden\'s method.', ('muller', 'broyden', 'brent')),
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
            ('initial-range', 'Initial range size',
             u'Initial range size (Muller and Brent methods only).  (complex [a.u.])'))),
    ),
    u'flow': ('inTemperature', 'inGain',
             u'outLightMagnitude', 'outRefractiveIndex', 'outHeat', 'outLoss', 'outWavelenght'),
}

solvers.register_config('optical', 'slab', Fourier2D=Fourier2D, Fourier3D=Fourier3D)
