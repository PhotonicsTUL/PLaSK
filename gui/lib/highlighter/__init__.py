# coding: utf-8
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

# This code is a heavily based on: https://bitbucket.org/henning/syntaxhighlighter/

import re
from ...qt.QtGui import QSyntaxHighlighter, QColor, QTextCharFormat, QFont, QBrush, QTextFormat

try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    unicode = str
    basestring = (str, bytes)
else:
    # 'unicode' exists, must be Python 2
    bytes = str


class Format(object):

    __slots__ = ("name", "tcf")

    NAME = QTextFormat.UserProperty + 1

    def __init__(self, name, color=None, bold=None, italic=None, underline=None, base_format=None):
        self.name = name
        tcf = QTextCharFormat()
        if base_format is not None:
            if isinstance(base_format, Format):
                base_format = base_format.tcf
            tcf.merge(base_format)
            tcf.setFont(base_format.font())
        if color is not None:
            if not isinstance(color, QColor):
                color = QColor(color)
            tcf.setForeground(QBrush(color))
        if bold is not None:
            if bold:
                tcf.setFontWeight(QFont.Bold)
            else:
                tcf.setFontWeight(QFont.Normal)
        if italic is not None:
            tcf.setFontItalic(italic)
        if underline is not None:
            tcf.setFontUnderline(underline)
        tcf.setProperty(Format.NAME, name)
        self.tcf = tcf


class Context(object):
    # Every context maps to a specific state in QSyntaxHighlighter

    __slots__ = 'name', 'groups', 'next', 'is_multiline', 'search_next'

    def __init__(self, name, next, is_multiline=False):
        self.name = name
        self.groups = []
        next_groups = []
        for p, n in next:
            next_groups.append(p)
            self.groups.append(n)
        self.is_multiline = is_multiline
        next_pattern = '(' + ')|('.join(next_groups) + ')'
        self.search_next = re.compile(next_pattern, re.M|re.S).search


class ContextScanner(object):

    def __init__(self, contexts):
        self.contexts = []
        cidx = {}
        for i, c in enumerate(contexts):
            if isinstance(c, (tuple,list)): c = Context(*c)
            self.contexts.append(c)
            cidx[c.name] = i
        for c in self.contexts:
            c.groups = [-1 if g is None else -2 if g == '#' else cidx[g] for g in c.groups]
        self.modulo = len(self.contexts)

    def scan(self, current_state, text):
        last_pos = 0
        length = len(text)
        contexts = self.contexts
        modulo = self.modulo
        current_context = contexts[current_state % modulo]
        # loop yields (start, end, context, new_state, is_inside)
        while last_pos < length:
            found = current_context.search_next(text, last_pos)
            if found:
                start, end = found.span()
                yield last_pos, start, current_context.name, current_state, True
                next_state = current_context.groups[found.lastindex-1]
                if next_state == -1:
                    yield start, end, current_context.name, current_state, False
                    current_state //= modulo
                    current_context = contexts[current_state % modulo]
                elif next_state == -2:
                    yield start, end, current_context.name, current_state, False
                else:
                    current_state = current_state * modulo + next_state
                    current_context = contexts[next_state]
                    yield start, end, current_context.name, current_state, False
                last_pos = end
            else:
                yield last_pos, length, current_context.name, current_state, True
                break
        if current_state != 0:
            c = contexts[current_state % modulo]
            if not c.is_multiline:
                current_state //= modulo
        yield length, length, None, current_state, False


class Token(object):
    __slots__ = 'name', 'pattern', 'prefix', 'suffix'

    def __init__(self, name, pattern, prefix="", suffix=""):
        self.name = name
        if isinstance(pattern, list):
            pattern = "|".join(pattern)
        self.pattern = pattern
        self.prefix = prefix
        self.suffix = suffix


class TokenScanner(object):
    __slots__ = 'tokens', 'search'

    def __init__(self, tokens):
        self.tokens = []
        groups = []
        for t in tokens:
            if isinstance(t, (list,tuple)):
                t = Token(*t)
            elif isinstance(t, dict):
                t = Token(**t)
            else:
                assert isinstance(t, Token), "Token expected, got {!r}".format(t)
            gdef = "?P<{}>".format(t.name)
            if gdef in t.pattern:
                p = t.pattern
            else:
                p = "({}{})".format(gdef, t.pattern)
            p = t.prefix + p + t.suffix
            groups.append(p)
            self.tokens.append(t)
        pat = "|".join(groups)
        self.search = re.compile(pat).search

    def scan(self, s):
        search = self.search
        #length = len(s)
        last_pos = 0
        # loop yields (token, start_pos, end_pos)
        while 1:
            found = search(s, last_pos)
            if found:
                lg = found.lastgroup
                start, last_pos = found.span(lg)
                yield lg, start, last_pos
            else:
                break


class SyntaxHighlighter(QSyntaxHighlighter):

    def __init__(self, parent, context_scanner, token_scanner, formats, default_font=None):
        """
        :param parent: QDocument or QTextEdit/QPlainTextEdit instance
        'context_scanner:
            ContextScanner instance
        :param token_scanner:
            dictionary of token scanners for each context
            The key is the name of the context, the value is a TokenScanner instance
            The default scanner has the key None
        :formats:
            list of tuples consisting of a name and a format definition
            The name is the name of a context or token

        """
        QSyntaxHighlighter.__init__(self, parent)
        parent.setDefaultFont(default_font)
        self.context_scanner = context_scanner
        self.token_scanner = token_scanner

        self.formats = {}
        for f in formats:
            if isinstance(f, tuple):
                fname, f = f
            else:
                assert isinstance(f, Format)
            if isinstance(f, basestring):
                f = (f,)  # only color specified
            if isinstance(f, (tuple,list)):
                f = Format(*((fname,) + f))
            elif isinstance(f, dict):
                f = Format(**dict(name=fname, **f))
            else:
                assert isinstance(f, Format), "Format expected, {!r} found".format(f)
            f.tcf.setFontFamily(parent.defaultFont().family())
            self.formats[f.name] = f.tcf

        scan_inside = {}
        for inside_part, inside_scanner in self.token_scanner.items():
            scan_inside[inside_part] = inside_scanner.scan
        # reduce name look-ups for better speed
        self.get_tokens = scan_inside.get
        self.scan_contexts = context_scanner.scan
        self.get_format = self.formats.get

    def highlightBlock(self, text):
        """Automatically called by Qt"""

        text = unicode(text) + "\n"
        previous_state = self.previousBlockState() + 1
        new_state = previous_state
        # speed-up name-lookups
        get_format = self.get_format
        set_format = self.setFormat
        get_tokens = self.get_tokens

        for start, end, context, new_state, is_inside in self.scan_contexts(previous_state, text):
            f = get_format(context, None)
            if f: set_format(start, end-start, f)
            if is_inside:
                tokens = get_tokens(context)
                if tokens:
                    for token, token_pos, token_end in tokens(text[start:end]):
                        f = get_format(token)
                        if f: set_format(start+token_pos, token_end-token_pos, f)

        self.setCurrentBlockState(new_state - 1)


def load_syntax(syntax, context=None):
    context = context or {}

    context_scanner = ContextScanner(syntax.get("contexts", []))

    tokens = {}
    for names, items in syntax.get("tokens", {}).items():
        scanner = TokenScanner(items)
        if not isinstance(names, tuple):
            names = names,
        for name in names:
            tokens[name] = scanner

    formats = []
    for fname, fstyle in syntax.get("formats", {}).items():
        if isinstance(fstyle, basestring):
            if fstyle.startswith("{") and fstyle.endswith("}"):
                key = fstyle[1:-1]
                fstyle = context[key]
            else:
                fstyle = fstyle.format(context)
        formats.append((fname,fstyle))

    return context_scanner, tokens, formats


__all__ = "Format", "Context", "ContextScanner", "Token", "TokenScanner", "SyntaxHighlighter", "load_syntax"