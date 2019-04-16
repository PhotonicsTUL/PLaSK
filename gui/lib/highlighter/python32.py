default_context = [
    ('#', 'comment'),
    ('{', 'dict'),
    ("[buBU]?[rR]?'''", 'string multi single'),
    ("[buBU]?[rR]?'", 'string single'),
    ('[buBU]?[rR]?"""', 'string multi double'),
    ('[buBU]?[rR]?"', 'string double'),
    ("[rR]?[fF][rR]?'''", 'string format multi single'),
    ("[rR]?[fF][rR]?'", 'string format single'),
    ('[rR]?[fF][rR]?"""', 'string format multi double'),
    ('[rR]?[fF][rR]?"', 'string format double'),
]

default_key = ('default', 'dict', 'format')

syntax = {
    'formats': {
        'builtin': '{syntax_builtin}',
        'comment': '{syntax_comment}',
        'keyword': '{syntax_keyword}',
        'number': '{syntax_number}',
        'decorator': '{syntax_decorator}',
        'string single': '{syntax_string}',
        'string multi single': '{syntax_string}',
        'string double': '{syntax_string}',
        'string multi double': '{syntax_string}',
        'string format single': '{syntax_string}',
        'string format multi single': '{syntax_string}',
        'string format double': '{syntax_string}',
        'string format multi double': '{syntax_string}',
        'fspec': '{syntax_string}',
    },

    'contexts': [
        ('default', default_context, True),
        ('dict', [('}', None)] + default_context, True),
        ('format', default_context + [(r'\}', None)], True),
        ('comment', [('\n', None)]),
        ('string multi single', [("'''", None)], True),
        ('string single', [("'", None)]),
        ('string multi double', [('"""', None)], True),
        ('string double', [('"', None)]),
        ('string format multi single', [("'''", None), (r"\{\{", '#'), (r"\{", 'format')], True),
        ('string format single', [("'", None), (r"\{\{", '#'), (r"\{", 'format')]),
        ('string format multi double', [('"""', None), (r"\{\{", '#'), (r"\{", 'format')], True),
        ('string format double', [('"', None), (r"\{\{", '#'), (r"\{", 'format')]),
    ],

    'tokens': {
        default_key: [
            ('hexnumber', '(0x)([0-9a-fA-F])+?'),
            ('number', '[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?[jJ]?|0x[0-9a-f]+'),
            ('keyword', [
                'and',
                'as',
                'assert',
                'break',
                'class',
                'continue',
                'def',
                'del',
                'elif',
                'else',
                'except',
                'finally',
                'for',
                'from',
                'global',
                'if',
                'import',
                'in',
                'is',
                'lambda',
                'nonlocal',
                'not',
                'or',
                'pass',
                'raise',
                'return',
                'try',
                'while',
                'with',
                'yield'
            ], '(^|[\x08\\W])', '[\x08\\W]'),
            ('builtin', [
                'ArithmeticError',
                'AssertionError',
                'AttributeError',
                'BaseException',
                'BufferError',
                'BytesWarning',
                'DeprecationWarning',
                'EOFError',
                'Ellipsis',
                'EnvironmentError',
                'Exception',
                'False',
                'FloatingPointError',
                'FutureWarning',
                'GeneratorExit',
                'IOError',
                'ImportError',
                'ImportWarning',
                'IndentationError',
                'IndexError',
                'KeyError',
                'KeyboardInterrupt',
                'LookupError',
                'MemoryError',
                'NameError',
                'None',
                'NotImplemented',
                'NotImplementedError',
                'OSError',
                'OverflowError',
                'PendingDeprecationWarning',
                'ReferenceError',
                'ResourceWarning',
                'RuntimeError',
                'RuntimeWarning',
                'StopIteration',
                'SyntaxError',
                'SyntaxWarning',
                'SystemError',
                'SystemExit',
                'TabError',
                'True',
                'TypeError',
                'UnboundLocalError',
                'UnicodeDecodeError',
                'UnicodeEncodeError',
                'UnicodeError',
                'UnicodeTranslateError',
                'UnicodeWarning',
                'UserWarning',
                'ValueError',
                'Warning',
                'ZeroDivisionError',
                '__import__',
                'abs',
                'all',
                'any',
                'ascii',
                'bin',
                'bool',
                'bytearray',
                'bytes',
                'callable',
                'chr',
                'classmethod',
                'compile',
                'complex',
                'copyright',
                'credits',
                'delattr',
                'dict',
                'dir',
                'divmod',
                'enumerate',
                'eval',
                'exec',
                'exit',
                'filter',
                'float',
                'format',
                'frozenset',
                'getattr',
                'globals',
                'hasattr',
                'hash',
                'help',
                'hex',
                'id',
                'input',
                'int',
                'isinstance',
                'issubclass',
                'iter',
                'len',
                'license',
                'list',
                'locals',
                'map',
                'max',
                'memoryview',
                'min',
                'next',
                'object',
                'oct',
                'open',
                'ord',
                'pow',
                'print',
                'property',
                'quit',
                'range',
                'repr',
                'reversed',
                'round',
                'self',
                'set',
                'setattr',
                'slice',
                'sorted',
                'staticmethod',
                'str',
                'sum',
                'super',
                'tuple',
                'type',
                'vars',
                'zip'
            ], '([^\\.\\w]|^)', '[\x08\\W]'),
            ('decorator', '^\s*@[A-Za-z_][A-Za-z_0-9]*'),
            ('ident', '[A-Za-z_][A-Za-z_0-9]*')
        ],
        'format': [
            ('fspec', r'(![rs])?(:([^}]?[&lt;&gt;=^])?[ +-]?#?0?[0-9]*(\.[0-9]+)?[bcdeEfFgGnosxX%]?)\s*$')
        ]
    }
}
