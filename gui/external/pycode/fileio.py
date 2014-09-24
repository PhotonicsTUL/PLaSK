# -*- coding: utf-8 -*-
import re


ENCODING = "utf-8"
STRIP_BLANK_LINES = True
UNTABIFY = True
TAB_WIDTH = 4


def guess_encoding(source, peek_lines=100, encoding=ENCODING):
    """
    :param source: source code as byte string
    :param peek_lines: number of lines to check for encoding hint
    :param encoding: default encoding ist utf-8
    :return: name of encoding
    """
    lines = source.splitlines()
    enc_match = re.compile("#\s*-\*-\s*coding:\s*(.*?)\s*-\*-").match
    for i, l in enumerate(lines):
        if i > peek_lines:
            break
        if not l.startswith("#"):
            break
        found = enc_match(l)
        if found:
            encoding = found.group(1)
            print "found encoding", repr(encoding), "for", filename
            break
    return encoding


def load(filename):
    """
    :param filename: existing filenname
    :return: source code as unicode string
    """
    source = open(filename).read()
    encoding = guess_encoding(source)
    return source.decode(encoding)


def save(filename, source, tab_width=TAB_WIDTH, strip_blank_lines=STRIP_BLANK_LINES, untabify=UNTABIFY):
    """
    :param filename:
    :param source:
    :param tab_width:
    :param strip_blank_lines:
    :param untabify:
    :return:
    """
    assert isinstance(
        source, unicode), "expected source code as unicode string"
    encoding = guess_encoding(source)
    tab = u" " * tab_width
    lines = source.splitlines()
    out = []
    for l in lines:
        if strip_blank_lines and len(l.strip()) == 0:
            l = u""
        elif untabify:
            l = l.replace(u"\t", tab)
        out.append(out)
    source = u"\n".join(out).encode(encoding)

    f = open(filename, "w")
    f.write(source)
    f.close()
