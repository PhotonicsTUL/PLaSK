from __future__ import absolute_import

syntax = {
    'formats': {
        'member': '{syntax_member}',
        'plask': '{syntax_plask}',
        'provider': '{syntax_provider}',
        'receiver': '{syntax_receiver}',
        'log': '{syntax_log}',
        'solver': '{syntax_solver}',
        'define': '{syntax_define}',
        'loaded': '{syntax_loaded}',
        'pylab': '{syntax_pylab}',
    },

    'tokens': [
        ('provider', 'out[A-Z]\w*', '\\.'),
        ('receiver', 'in[A-Z]\w*', '\\.'),
        ('member', '[A-Za-z_]\w*', '\\.'),
        ('plask', [
            # 'plask',
            'geometry',
            'mesh',
            'material',
            'flow',
            'phys',
            'algorithm',
            'config',
            'vec',
            'Data',
            'Manager',
            'StepProfile',
            'ComputationError',
            'loadxpl',
            'runxpl',
            'print_log',
            'print_exc',
            'save_field',
            'load_field',
            'plot_field',
            'plot_vectors',
            'plot_stream',
            'plot_geometry',
            'plot_mesh',
            'plot_boundary',
            'plot_profile',
            'wl',
        ], '(^|[^\\.\\w]|\\bplask\\.)', '(?:[\x08\\W]|$)'),
        ('log', 'LOG_[A-Z_]+(?!\w)'),
        ('loaded', [
            'DEF',
            'GEO',
            'PTH',
            'MSH',
        ], '(^|[^\\.\\w])', '(?:[\x08\\W]|$)'),
    ]
}

# try:
#     import plask
# except ImportError:
#     pass
# else:
#     syntax['token_scanner'][0][1] = [k for k in plask.__dict__ if not k.startswith('_')]

try:
    import pylab
except ImportError:
    pass
else:
    _pylab = [k for k in pylab.__dict__ if not k.startswith('_') and
        not k in (
            'sys',
        )
    ]
    _pylab.extend([
        'aspect', 'window_title'
    ])
    syntax['tokens'].append(
        ('pylab', _pylab,
         '(^|[^\\.\\w]|\\bplask\\.)', '(?:[\x08\\W]|$)')
    )
