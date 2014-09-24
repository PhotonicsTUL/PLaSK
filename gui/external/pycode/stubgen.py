#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect
import os
import re
import imp
import textwrap
from types import (
    ModuleType, ClassType, ObjectType, MethodType, FunctionType, UnboundMethodType,
    BuiltinFunctionType, BuiltinMethodType)
from pprint import pformat


class SignatureParser(object):

    examples = """
    recv(buffersize[, flags]) -> data
    recv_into(buffer, [nbytes[, flags]]) -> nbytes_read
    sendto(data[, flags], address) -> count
    trunc(x:Real) -> Integral
    sinh(x)
    log(x[, base])
    hypot(x, y)
    factorial(x) -> Integral
    foo(arg=None) -> str
    QX11Info.isCompositingManagerRunning() -> bool
    qDrawWinPanel(QPainter, int, int, int, int, QPalette, bool sunken=False, QBrush fill=None)
    qDrawWinPanel(QPainter, QRect, QPalette, bool sunken=False, QBrush fill=None)
    qDrawShadeRect(QPainter, int, int, int, int, QPalette, bool sunken=False, int lineWidth=1, int midLineWidth=0, QBrush fill=None)
    qDrawShadeRect(QPainter, QRect, QPalette, bool sunken=False, int lineWidth=1, int midLineWidth=0, QBrush fill=None)
    func1(*args)
    func2(**kwargs)
    func3(*args, **kwargs)
    func4(x, *args, **kwargs)
    func5(x, y=123, **kwargs)
    """

    example2 = """
    :param foo: int argument
    :param bar:
    :param baz
    """

    def tokenize(self, s):
        s = s.strip()
        pos = 0
        buf = ""
        state = 0
        optional = 0
        ST_NORMAL = 0
        ST_STRING = 1
        string_start = -1
        while pos < len(s):
            char = s[pos]
            if state == ST_STRING:
                yield "STRING", s[string_start:pos]
                state = ST_NORMAL
                pos += 1
            elif state == ST_NORMAL:
                if char.isdigit():
                    buf = char
                    while pos < len(s) and char.isdigit():
                        pos += 1
                        char = s[pos]
                    yield "NUMBER", int(buf)
                elif char == ",":
                    yield "COMMA", ","
                    pos += 1
                elif char in ('"', "'"):
                    state = ST_STRING
                    string_start = pos
                    pos += 1
                elif s.startswith("->", pos):
                    yield "RETURNS", "->"
                    pos += 2
                elif char == "[":
                    optional += 1
                    pos += 1
                elif char == "]":
                    optional -= 1
                    pos += 1
                elif char == "(":
                    pos += 1
                    yield "LPAREN", "("
                elif char == ")":
                    pos += 1
                    yield "RPAREN", ")"
                elif char == ":":
                    pos += 1
                    yield "COLON", ":"
                elif char == "=":
                    pos += 1
                    yield "EQUALS", "="
                elif char == "*":
                    pos += 1
                    yield "STAR", "*"
                elif char == ".":
                    pos += 1
                    yield "DOT", "."
                elif char == " ":
                    pos += 1  # skip whitespace
                #  number = re.match("^\d+(\.\d*)?", s[pos:])
                else:
                    ident = re.match("^([A-Za-z_][A-Za-z_0-9]*)", s[pos:])
                    if ident:
                        length = ident.end()
                        yield "IDENT", s[pos:pos + length]
                        pos += length
                    else:
                        yield "UNKNOWN", char
                        pos += 1

    def parse(self, s):
        for line in s.splitlines():
            result = self.parse_line(line)
            if result:
                yield result

    def parse_line(self, s):
        next_token = self.tokenize(s).next
        args = []
        defaults = {}
        types = {}
        returns = None
        name = None
        token = next_token()
        if token[0] != "IDENT":
            return
        name = token[1]
        token = next_token()
        while token[0] == "DOT":
            token = next_token()
            if token[0] == "IDENT":
                name = token[1]
                token = next_token()
            else:
                assert token[0] == "LPAREN"
        if token[0] != "LPAREN":
            return
        token = next_token()
        while 1:
            if token[0] == "RPAREN":
                break
            if token[0] == "IDENT":
                args.append(token[1])
            token = next_token()
            if token[0] == "IDENT":
                # c-like declaration of type
                types[token[1]] = args[-1]
                args[-1] = token[1]
                token = next_token()
            if token[0] == "EQUALS":
                token = next_token()
                assert token[0] in (
                    "IDENT", "STRING", "NUMBER"), "default value is %s" % repr(token)
                defaults[args[-1]] = token[1]
                token = next_token()

            if token[0] == "COMMA":
                token = next_token()
            elif token[0] == "COLON":
                token = next_token()
                assert token[0] == "IDENT"
                types[args[-1]] = token[1]
                token = next_token()
            else:
                assert token[0] == "RPAREN", ") expected, %s found" % repr(
                    token)

        try:
            token = next_token()
        except StopIteration:
            token = (None, None)
        if token[0] == "RETURNS":
            try:
                token = next_token()
                if token[0] == "IDENT":
                    returns = token[1]
            except StopIteration:
                pass
        return name, args, returns, defaults, types


class StubCreator(object):

    def __init__(self, module_name):
        if "/" in module_name:
            name_ext = os.path.basename(module_name)
            name, suffix = os.path.splitext(name_ext)
            mod = imp.load_module(
                name, open(module_name), module_name, (suffix, "rb", 3))
            mod.__file__ = module_name
            module_name = name
        else:
            # fh, filename, (suffix, mode, type_ = imp.find_module(mdule_name)
            #mod = imp.load_module(module_name, fh, module_name, (suffix, mode, type_))
            mod = __import__(module_name, {}, {}, [])
        for name in module_name.split(".")[1:]:
            mod = getattr(mod, name)
        try:
            filename = self.mod.__file__.rstrip("co")
        except:
            filename = "<%s>" % module_name
        self.mod = mod
        self.indention_level = 0
        self.indention = " " * 4
        self.code = [
            "# -*- coding: utf-8 -*-",
            "# generated stub module from %r" % filename
        ]
        doc = self.mod.__doc__
        if doc:
            self.emit('"""\n%s\n"""' % doc.rstrip("\n"))
        self.emit()
        self.generate(self.mod)

    def emit(self, line=""):
        indent = self.indention * self.indention_level
        if line:
            line = indent + line
        self.code.append(line)

    def indent(self, count=1):
        self.indention_level += count

    def dedent(self, count=1):
        self.indention_level = max(0, self.indention_level - count)

    def is_direct_instance(self, obj, *types):
        for t in types:
            if isinstance(obj, t) and obj.__class__ is t:
                return True
        return False

    def is_save(self, root):
        if self.is_direct_instance(root, basestring, str, unicode, int, float, bool):
            return True
        elif self.is_direct_instance(root, tuple, list):
            for item in root:
                if not self.is_save(item):
                    return False
            return True
        elif self.is_direct_instance(root, dict):
            for key, value in root.items():
                if not self.is_save(key):
                    return False
                if not self.is_save(value):
                    return False
            return True
        else:
            return False

    def generate(self, root):
        for name in dir(root):
            if name == "__all__":  # and root is self.mod:
                # XXX: not working correctly (see email module)
                pass
            elif name.startswith("_"):
                continue

            try:
                value = getattr(root, name)
            except AttributeError, e:
                print "!", e
                continue

            if not callable(value) and self.is_save(value):
                # constant
                self.emit("%s = %s" % (name, pformat(value)))

            elif inspect.ismethoddescriptor(value) or isinstance(value, (FunctionType, MethodType, UnboundMethodType, BuiltinFunctionType, BuiltinMethodType)):
                # function
                try:
                    val_name = value.__name__
                except AttributeError:
                    val_name = None
                if val_name != name:
                    try:
                        real = value.__name__
                    except AttributeError:
                        real = None
                    if real:
                        self.emit("%s = %s" % (name, real))
                        self.emit()
                else:
                    self.generate_function(value)

            elif isinstance(value, ModuleType):
                # module
                if name != value.__name__:
                    self.emit("import %s as %s" % (value.__name__, name))
                else:
                    self.emit("import %s" % name)

            elif isinstance(value, (ClassType, ObjectType)):
                # class
                if hasattr(value, "__name__"):
                    if value.__name__ != name:
                        self.emit("%s = %s" % (name, value.__name__))
                        self.emit()
                    else:
                        self.generate_class(value)

    def generate_class(self, cls):
        bases = (b.__name__ for b in getattr(cls, "__bases__", []))
        self.emit("class %s(%s):" % (cls.__name__, ", ".join(bases)))
        self.indent()
        doc = cls.__doc__
        if not doc:
            doc = ""
        self.emit('"""%s"""' % doc)
        self.generate(cls)
        self.dedent()
        self.emit()

    def extract_signature_from_doc(self, doc):
        try:
            sig = SignatureParser().parse(doc).next()
            name, args, results = sig
        except:
            return "()"
        self.emit(
            "# real signature not available, using information from __doc__")
        if self.indention_level > 0:
            args.insert(0, "self")
        return "(%s)" % ", ".join(args)

    def old_extract_signature_from_doc(self, doc):

        lines = [l.strip() for l in (doc or "").splitlines() if l.strip()]
        args = []
        if self.indention_level > 0:
            args.append("self")
        for l in lines:
            found = re.match(".*\((?P<args>.*)\)($|[ ]*->)", l)
            if found:
                self.emit(
                    "# real signature not available, using information from __doc__")
                for a in found.group("args").split(","):
                    a = a.strip()
                    if "=" in a:
                        a, _sep, v = a.partition("=")
                        a = a.strip()
                    else:
                        v = None
                    a = a.replace(" ", "_")
                    if v and a:
                        a = "%s=%s" % (a, v)
                    if a == "...":  # Found in PyQt
                        a = "*args"
                    if a:
                        args.append(a)
                break
        return "(%s)" % ", ".join(args)

    def generate_function(self, func):
        doc = func.__doc__
        try:
            (args, varargs, varkw, defaults) = inspect.getargspec(func)
            signature = inspect.formatargspec(args, varargs, varkw, defaults)
        except TypeError, e:
            signature = self.extract_signature_from_doc(doc)
        if not doc:
            doc = "%s%s" % (func.__name__, signature)
        self.emit("def %s%s:" % (func.__name__, signature))
        self.indent()
        indent = self.indention * self.indention_level
        self.emit('"""\n%s%s\n%s"""' % (indent, doc.rstrip("\n"), indent))
        self.dedent()
        self.emit()

    def __str__(self):
        return "\n".join(self.code)


def test_sp():
    sp = SignatureParser()
    for line in sp.examples.splitlines():
        print line.strip()
        for sig in sp.parse(line):
            print "\t", sig

    # print sp.parse_line("sin(x)")


def test_sc():
    #s = StubCreator("math")
    #s = StubCreator("PyQt4.QtGui")
    #s = StubCreator("_socket")
    s = StubCreator("/usr/lib64/python2.7/site-packages/sip.so")
    print s


if __name__ == "__main__":
    # test_sp()
    test_sc()
