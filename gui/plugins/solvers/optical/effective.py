# coding: utf8

from gui import solvers

EffectiveFrequencyCyl = {
    'class': 'EffectiveFrequencyCyl',
    'desc': 'Scalar optical solver based on the effective index method',
    'mesh': True,
    'conf': (
        ('mode', 'Mode properties', (
            ('lam0', 'Approximate wavelength [nm]'),
            ('emission', 'Direction of emission', ('top', 'bottom')),
            ('vlam', 'Vertical wavelength [nm]'),
            ('vat', u'Vertical solution coord [µm]'))),
        ('root', 'Parameters of the global root-finding algorithm', (
            ('method', 'Root finding method', ('muller', 'broyden')),
            ('tolx', 'Tolerance on effective frequency'),
            ('tolf-min', 'Sufficient determinant value'),
            ('tolf-max', 'Rquired determinant value'),
            ('maxstep', 'Maximum Broyden iteration step'),
            ('maxiter', 'Maximum number of iterations'),
            ('alpha', 'Broyden decrease factor'),
            ('lambda', 'Minimum Broyden step decrease ratio'),
            ('initial-range', 'Initial Muller range size'))),
        ('stripe-root', 'Parameters of the vertical root-finding algorithm', (
            ('method', 'Root finding method', ('muller', 'broyden')),
            ('tolx', 'Tolerance on effective frequency'),
            ('tolf-min', 'Sufficient determinant value'),
            ('tolf-max', 'Rquired determinant value'),
            ('maxstep', 'Maximum Broyden iteration step'),
            ('maxiter', 'Maximum number of iterations'),
            ('alpha', 'Broyden decrease factor'),
            ('lambda', 'Minimum Broyden step decrease ratio'),
            ('initial-range', 'Initial Muller range size'))),
    ),
    'flow': ('inTemperature', 'inGain',
             'outLightMagnitude', 'outRefractiveIndex', 'outHeat', 'outLoss', 'outWavelenght'),
}

solvers.register_config('optical', EffectiveFrequencyCyl=EffectiveFrequencyCyl)