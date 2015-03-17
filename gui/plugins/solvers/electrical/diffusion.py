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

Diffusion = {
    'desc': 'Lateral carriers diffusion solver.',
    'mesh': 'Regular',
    'conf': (
        ('config', 'Solver configuration', (
            ('fem-method', 'FEM order',
                u'Order of the finite-element method.', ('linear', 'parabolic')),
            ('accuracy', 'Relative accuracy [%]',
                u'Required relative accuracy. (float&nbsp;[%])'),
            ('abs-accuracy', 'Absolute accuracy [cm<sup>-3</sup>])',
                u'Required absolute minimal concentration accuracy. (float&nbsp;[cm<sup>-3</sup>])'),
            ('maxiters', 'Iterations to refine',
                u'Maximum number of allowed iterations before attempting to refine mesh. (integer)'),
            ('maxrefines', 'Allowed refinements',
                u'Maximum number of allowed mesh refinements. (integer)'),
            ('interpolation', 'Interpolation method',
                u'Current density interpolation method name.', ('linear', 'spline')))),
    ),
    'flow': ('inTemperature', 'inCurrentDensity', 'inGain', 'inGainOverCarriersConcentration', 'inLightMagnitude',
             'outCarriersConcentration'),
}

solvers.register_config('electrical', 'diffusion', Diffusion2D=Diffusion, DiffusionCyl=Diffusion)
solvers.save()
