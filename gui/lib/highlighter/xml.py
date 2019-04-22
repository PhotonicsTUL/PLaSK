syntax = {
    'formats': {
        'comment': '{syntax_comment}',
        'tag': '{syntax_tag}',
        'attr': '{syntax_attr}',
        'equals': '{syntax_attr}',
        'value': '{syntax_value}',
        'text': '{syntax_text}',
        'define attr': '{syntax_define}',
        'define text': '{syntax_define}',
        'dict': '{syntax_define}',
    },

    'contexts': [
        ('default', [
            ('comment', '<!--'),
            ('tag', '<'),
            ('value', '"'),
            ('#', '{{'),
            ('define text', '{', 'text')
        ], True),
        ('comment', [(None, '-->')], True),
        ('tag', [(None, '>'), ('value', '"')], True),
        ('value', [(None, '"'), ('#', '{{'), ('define attr', '{', 'value')]),
        ('define attr', [(None, '}', 'value'), ('dict', '{')]),
        ('define text', [(None, '}', 'text'), ('dict', '{')]),
        ('dict', [(None, '}')]),
    ],

    'tokens': {
        'tag': [
            ('tag', r'^[\s]+'),
            ('attr', '[A-Za-z_][A-Za-z_0-9-:]*(?==)'),
            ('equals', '='),
        ]
    }
}
