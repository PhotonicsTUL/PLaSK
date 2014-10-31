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

Fermi = {
    'desc': 'Simple gain solver based on Fermi Golden Rule.',
    'mesh': 'Ordered',
    'conf': (
        ('config', 'Gain parameters', (
            ('lifetime', 'Carriers lifetime',
                u'Carriers lifetime. (float)'),
            ('matrix-elem', 'Matrix element',
                u'Value of the matrix element in gain computations '
                u'(if not set it is estimated automatically). (float)'),
            ('strained', 'Strained',
                u'Boolean attribute indicated if the solver should consider strain in the active region. '
                u'If set to *yes* than there must a layer with the role "\ *substrate*\ " in the geometry. '
                u'The strain is computed by comparing the atomic lattice constants of the substrate and '
                u'the quantum wells.', ('yes', 'no')))),
        ('levels', 'Custom energy levels '
                   '<span style="font-weight: normal;">(either none or all attributes must be given)</span>', (
            ('el', 'Electron levels [eV]',
                u'Comma-separated list of electron levels. (list of floats&nbsp;[eV])'),
            ('hh', 'Heavy hole levels [eV]',
                u'Comma-separated list of heavy hole levels. (list of floats&nbsp;[eV])'),
            ('lh', 'Light hole levels [eV]',
                u'Comma-separated list of light hole levels. (list of floats&nbsp;[eV])'))),
    ),
    'flow': ('inTemperature', 'inCarriersConcentration', 'outGain', 'outGainOverCarriersConcentration'),
}

solvers.register_config('gain', 'simple', Fermi2D=Fermi, FermiCyl=Fermi)
