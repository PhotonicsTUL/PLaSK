# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

SYNTAX = {
    'formats': {
        'comment': '{syntax_comment}',
        'tag': '{syntax_tag}',
        'attr': '{syntax_attr}',
        'equals': '{syntax_attr}',
        'value': '{syntax_value}',
        'text': '{syntax_text}',
        'cdata': '{syntax_text}',
        'cdata tag': '{syntax_attr}',
        'define attr': '{syntax_define}',
        'define text': '{syntax_define}',
        'dict': '{syntax_define}',
    },

    'contexts': [
        ('default', [
            ('cdata tag', r'<!\[CDATA\['),
            ('comment', '<!--'),
            ('tag', '<'),
            ('value', '"'),
            (0, '{{'),
            ('define text', '{', 'text')
        ], True),
        ('cdata tag', [('cdata', '')]),
        ('cdata', [(-2, r'\]\]>', 'cdata tag')], True),
        ('comment', [(None, '-->')], True),
        ('tag', [(None, '>'), ('value', '"')], True),
        ('value', [(None, '"'), (0, '{{'), ('define attr', '{', 'value')]),
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
