# -*- coding: utf-8 -*-
import sys
import re
from keyword import kwlist as python_keywords

# TODO: add scraper which removes sidebar and navigation footer/header


class ContextWrapper(object):

    def __init__(self, d):
        self.d = d

    def lower(self, s):
        return s.lower()

    def __getitem__(self, key):
        if "|" in key:
            key, _sep, value_filter_name = key.partition("|")
            value_filter = getattr(self, value_filter_name)
        else:
            value_filter = lambda x: x
        return value_filter(self.d[key])


class CodeHelp(object):

    pattern = "(?P<module>\w+)\.(?P<function>\w+)"
    url = "http://docs.python.org/release/%(pyversion)s/library/%(module)s.html#%(module)s.%(function)s"
    context = {
        "pyversion": ".".join(map(str, sys.version_info[:3]))
    }
    context_wrapper = ContextWrapper

    # TODO: web-scraping content with jquery-like api

    def match(self, fully_qualified_name):
        found = re.match(self.pattern, fully_qualified_name)
        if found:
            ctx = dict(self.context)
            ctx.update(found.groupdict())
            return self.url % self.context_wrapper(ctx)


class PyQtHelp(CodeHelp):

    pattern = "PyQt4\.(QtCore|QtGui|QtNetwork)\.(?P<class>\w+)\.(?P<method>\w+)"
    url = "http://www.riverbankcomputing.co.uk/static/Docs/PyQt4/html/%(class|lower)s.html#%(method)s"


class WxClassHelp(CodeHelp):
    # api documentation used epydoc
    pattern = "wx\.(?P<class>\w+)\.(?P<method>\w+)"
    url = "http://wxpython.org/docs/api/wx.%(class)s-class.html#%(method)s"


class WxModuleHelp(CodeHelp):
    pattern = "wx\.(?P<module>\w+)\.(?P<class>\w+)\.(?P<method>\w+)"
    url = "http://wxpython.org/docs/api/wx.%(module)s.%(class)s-class.html#%(method)s"


class NumpyHelp(CodeHelp):

    pattern = "numpy\.(?P<method>\w+)"
    url = "http://docs.scipy.org/doc/numpy/reference/generated/numpy.%(method)s.html#numpy.%(method)s"


help_providers = [
    PyQtHelp(),
    WxModuleHelp(),
    WxClassHelp(),
    NumpyHelp(),
    CodeHelp(),
]


def help_url(name):
    """
    Context sensitive help for a function or class/method.
    Support for PyQt, wxWidgets, Numpy and Stdlib is available right now.

    :param name: fully qualified dotted name (e.g. re.match)
    :return: url string or None if no help page was found
    """
    for h in help_providers:
        url = h.match(name)
        if url:
            return url


if __name__ == "__main__":
    print help_url("re.sub")
    print help_url("PyQt4.QtGui.QPushButton.setText")
    print help_url("wx.Window.Create")
    print help_url("wx.richtext.RichTextObject.__init__")
