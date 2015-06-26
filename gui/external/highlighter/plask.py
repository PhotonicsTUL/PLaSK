from __future__ import absolute_import

syntax = {
    'formats': {
        'member': '%(syntax_member)s',
        'plask': '%(syntax_plask)s',
        'provider': '%(syntax_provider)s',
        'receiver': '%(syntax_receiver)s',
        'log': '%(syntax_log)s',
        'solver': '%(syntax_solver)s',
        'define': '%(syntax_define)s',
        'loaded': '%(syntax_loaded)s',
        'pylab': '%(syntax_pylab)s',
    },

    'scanner': [
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
        ], '(^|[^\\.\\w]|\\bplask\\.)', '[\x08\\W]'),
        ('log', 'LOG_[A-Z]+(?!\w)'),
        ('loaded', [
            'DEF',
            'GEO',
            'PTH',
            'MSH',
            'MSG',
        ], '(^|[^\\.\\w])', '[\x08\\W]'),
    ]
}

# try:
#     import plask
# except ImportError:
#     pass
# else:
#     syntax['scanner'][0][1] = [k for k in plask.__dict__ if not k.startswith('_')]

try:
    import pylab
except ImportError:
    pass
else:
    syntax['scanner'].append(
        ('pylab', [k for k in pylab.__dict__ if not k.startswith('_')],
         '(^|[^\\.\\w]|\\bplask\\.)', '[\x08\\W]')
    )