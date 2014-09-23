syntax = {
    'formats': {
        'comment': '%(syntax_comment)s',
        'tag': '%(syntax_tag)s',
        'attr': '%(syntax_attr)s',
        'value': '%(syntax_value)s',
        'text': '%(syntax_text)s',
    },

    'partitions': [
        ('comment', '<!--', '-->', True),
        ('tag', '<', '>', True),
    ],

    'scanner': {
        'tag': [
            ('tag', '^[\s]+'),
            ('attr', '[A-Za-z_-]\w*(?==)'),
            ('equals', '='),
            ('value', '"[^"]*"'),
        ]
    }
}
