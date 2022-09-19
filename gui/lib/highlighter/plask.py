from re import escape
from copy import copy

from .python import SYNTAX, PYTHON_DEFAULT_TOKENS

plask_syntax = {
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
    plask_syntax['tokens'].append(
        ('pylab', _pylab,
         '(^|[^\\.\\w]|\\bplask\\.)', '(?:[\x08\\W]|$)')
    )


SYNTAX['formats'].update(plask_syntax['formats'])
SYNTAX['tokens'][PYTHON_DEFAULT_TOKENS][-1:-1] = plask_syntax['tokens']


def get_syntax(defines=None, solvers=None, **kwargs):
    syntax = {'formats': SYNTAX['formats'],
              'contexts': SYNTAX['contexts'],
              'tokens': copy(SYNTAX['tokens'])}
    syntax['tokens'][PYTHON_DEFAULT_TOKENS] = copy(SYNTAX['tokens'][PYTHON_DEFAULT_TOKENS])
    defs = ['ARRAYID', 'PROCID', 'JOBID']
    if defines is not None:
        defs += [escape(e.name) for e in defines.model.entries]
        syntax['tokens'][PYTHON_DEFAULT_TOKENS].insert(0, ('define', defs, '(^|[^\\.\\w])', '(?:[\x08\\W]|$)'))
    # current_syntax['tokens'][PYTHON_DEFAULT_TOKENS].insert(0, ('special', '__value__', '(^|[^\\.\\w])', '(?:[\x08\\W]|$)'))
    if solvers is not None:
        solvs = [escape(e.name) for e in solvers.model.entries if e.name]
        if solvs:
            syntax['tokens'][PYTHON_DEFAULT_TOKENS].insert(0, ('solver', solvs, '(^|[^\\.\\w])', '(?:[\x08\\W]|$)'))
    for key, val in kwargs.items():
        if isinstance(val, tuple):
            item = (key,) + val
        else:
            item = (key, val, '(^|[^\\.\\w])', '(?:[\x08\\W]|$)')
        syntax['tokens'][PYTHON_DEFAULT_TOKENS].insert(0, item)
    return syntax
