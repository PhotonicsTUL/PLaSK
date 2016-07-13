"""
Generate PLaSK stubs
"""

from __future__ import print_function

from inspect import getargspec, isclass
import sys
import os
import errno
import re
import collections


INDENT = " " * 4


class StubCreator(object):

    def __init__(self, module):
        self._module = module
        self._out = []
        self._doc = ""
        self._imports = set()

    def emit(self, code, depth):
        self._out.append("%s%s" % (INDENT * depth, code))

    def emit_doc(self, doc, depth):
        if depth == 0:
            self._doc = '"""\n' + doc + '"""\n\n'
        elif doc:
            lines = doc.splitlines()
            if not lines[0].strip(): del lines[0]
            if not lines[0].strip(): del lines[0]
            if not lines[-1].strip(): del lines[-1]
            if not lines[-1].strip(): del lines[-1]
            doc = "\n".join(depth * INDENT + l for l in lines)
            self.emit('"""\n{}\n{}"""'.format(doc, depth * INDENT), depth)

    def __str__(self):
        imports = ["import {}".format(mod) for mod in self._imports]
        if imports: imports.append("")
        return "#coding: utf8\n" + self._doc + "\n".join(imports + self._out)

    def create_stub(self, root, depth=0):
        emitted = False
        doc = getattr(root, "__doc__")
        if doc:
            self.emit_doc(doc, depth)
            emitted = True
        items1 = [(n, getattr(root, n)) for n in dir(root)
                  if ((n[:2] != '__' or n[-2:] != '__') or n == '__call__')
                  and hasattr(root, n)]
        items = []
        for name, value in items1:
            i = 0
            for _, other in items:
                if isclass(value) and isclass(other) and issubclass(other, value) and value.__name__ == name:
                    break
                i += 1
            items.insert(i, (name, value))
        # print(items)
        for name, value in items:
            if isinstance(value, int) or isinstance(value, float) or \
               isinstance(value, str) or value is None:
                self.emit("%s = %r" % (name, value), depth)
                emitted = True
            elif isinstance(value, property):
                self.create_property_stub(name, value, depth)
                emitted = True
            elif isinstance(value, collections.Callable) and hasattr(value, "__name__"):
                if isclass(value):
                    if name != "__class__":
                        self.create_class_stub(name, value, depth)
                        emitted = True
                else:
                    self.create_function_stub(name, value, depth)
                    emitted = True
            elif hasattr(value, "__class__") and isclass(value.__class__):
                cls = value.__class__
                if cls.__module__ == self._module:
                    self.emit("{} = {}()".format(name, cls.__name__), depth)
                else:
                    self.emit("{} = {}.{}()".format(name, cls.__module__, cls.__name__), depth)
                    self._imports.add(cls.__module__)
                self.emit("", depth)
                emitted = True
                #if isfunction(value) or isbuiltin(value):
                #    self.create_function_stub(name, value, depth)
                #elif isclass(value):
                #    self.create_class_stub(name, value, depth)
                #else:
                #    print "*", name, type(value), callable(value)
            else:
                if name not in ("__dict__", "__weakref__"):
                    print("*", name, type(value), isinstance(value, collections.Callable))
        return emitted

    def create_class_stub(self, name, cls, depth):
        if depth == 0:
            self.emit("", 0)
        #print name
        if cls.__name__ != name:
            self.emit("%s = %s # class alias" % (name, cls.__name__), depth)
        else:
            bases = list(cls.__bases__)
            for i,b in enumerate(bases):
                if b.__name__ == 'instance' and b.__module__ == 'Boost.Python':
                    bases[i] = 'object'
                elif b.__module__ != self._module:
                    bases[i] = b.__module__ + '.' + b.__name__
                    self._imports.add(b.__module__)
                else:
                    bases[i] = b.__name__
            bases = ", ".join(b for b in bases if b != name)
            if not bases: bases = "object"
            self.emit("class {}({}):".format(name, bases), depth)
            if not self.create_stub(cls, depth + 1):
                self.emit("pass", depth + 1)
        self.emit("", 0)

    rtype_re = re.compile(r"\s*:rtype:\s+([\w_](?:.?[\w\d_]+)*)")

    def create_property_stub(self, name, prop, depth):
        doc = getattr(prop, "__doc__", "")
        try:
            rtype = self.rtype_re.findall(doc)
        except TypeError:
            rtype = None
        if rtype:
            self.emit("@property", depth)
            self.emit("def {}():".format(name), depth)
            depth += 1
            # self.emit("{} = {}()".format(name, rtype[0]), depth)
        else:
            self.emit("{} = None".format(name), depth)
        if doc: self.emit_doc(doc, depth)
        self.emit("", 0)

    #e.g. BottomOf( (GeometryObject)object [, (PathHints)path=None]) -> Boundary :
    # search for: (type)name[=default_value]
    funcarg_re = re.compile("\\(([a-zA-Z0-9_]+)\\)"
                            "([a-zA-Z0-9_]+)"
                            "(?:=([a-zA-Z0-9_+-]+|'[^']*'|\"[^\"]*\"|\\([0-9eEj+-]+\\)))?")

    def func_args(self, func):
        try:
            args, varargs, keywords, defaults = getargspec(func)
        except TypeError as e:
            doc = getattr(func, "__doc__", "").splitlines()
            arglist = []
            aftsig = False
            if doc:
                for i,l in enumerate(doc):
                    if aftsig and not l.strip():
                        aftsig = False
                        doc[i] = None
                        continue
                    aftsig = False
                    args = []
                    for m in self.funcarg_re.finditer(l):
                        arg_s = m.group(2)
                        v = m.group(3)
                        if v is not None: arg_s += '=' + v;
                        args.append(arg_s)
                    if args and args[0] == 'arg1':
                        args[0] = "self"
                    if args:
                        arglist.append(args)
                        aftsig = True
                        doc[i] = None
                # print("{}: {}".format(func.__name__, ", ".join(str(a) for a in arglist)))
                sigs = []
                for args in arglist:
                    sigs.append("{}({})".format( func.__name__,  ', '.join(args)))
                doc = "\n".join(sigs) + "\n" + "\n".join(d for d in doc if d is not None) + "\n"
                if len(arglist) == 1:
                    return ', '.join(arglist[0]), doc
            return "*args, **kwargs", doc
        else:
            if defaults:
                defaults = dict(list(zip(args[-len(defaults):], defaults)))
            else:
                defaults = {}
            if varargs:
                args.append("*" + varargs)
            if keywords:
                args.append("**" + keywords)
            for i, a in enumerate(args):
                try:
                    d = defaults[a]
                except KeyError as e:
                    pass
                else:
                    args[i] = "%s=%r" % (a, d)
            return ", ".join(args), getattr(func, "__doc__", "")

    def create_function_stub(self, name, func, depth):
        # self.emit("", 0)
        if func.__name__ != name and func.__name__ != '<lambda>':
            self.emit("%s = %s # function alias" % (name, func.__name__), depth)
        else:
            args, doc = self.func_args(func)
            self.emit("def %s(%s):" % (name, args), depth)
            if doc:
                self.emit_doc(doc, depth + 1)
            else:
                self.emit("pass", depth + 1)
                self.emit("", 0)
        self.emit("", 0)

    def create_stub_from_module(self, module_name):
        mod = __import__(module_name, {}, {}, [])
        for part in module_name.split(".")[1:]:
            mod = getattr(mod, part)
        doc = getattr(mod, "__doc__")
        #if doc:
        #    self.emit_doc(doc)
        self.create_stub(mod)


if __name__ == "__main__":
    e = 0
    for arg in sys.argv[1:]:
        try:
            c = StubCreator(arg)
            c.create_stub_from_module(arg)
            path_comp = arg.split('.')
            path = os.path.join(*path_comp[:-1])
            try:
                os.makedirs(path)
            except OSError as exc: # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(path):
                    pass
                else: raise
            open(os.path.join(path, "__init__.py"), 'a+')
            file = open(os.path.join(path, path_comp[-1] + '.py'), 'w+', encoding='utf-8')
            print(c, file=file)
        except:
            print("Error while generating stubs for module:", arg, file=sys.stderr)
            import traceback
            traceback.print_exc()
            e += 1
    if e: print("", file=sys.stderr)
    sys.stderr.flush()
    #sys.exit(e)
