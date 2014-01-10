# -*- coding: utf-8 -*-
"""
    Autodoc extension for C++ classes
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This extension adds autodoc support for functions, methods amd classes
    with multiple signatures as provided by Boost Python.

    :seealso: http://sphinx-doc.org/ext/autodoc.html

    :copyright: Copyright 2014 Maciej Beling.
    :license: BSD, see LICENSE for details.
"""

import re

from sphinx.ext import autodoc
from sphinx.util.docstrings import prepare_docstring

# boost_signature_re = re.compile(r"^\s*(?:[\w.]+::)?(\w*)\((.*)\) -> (\w*) :$")
boost_signature_re = re.compile(r"^\s*(\w+ )?(?:[\w.]+::)?(\w*)\(tuple args, dict kwds\) :$")

class MultiSigMixin(object):
    '''Mixin that can properly output multiple signatures'''

    def _find_signatures(self, sig, encoding=None):
        docstrings = super(MultiSigMixin, self).get_doc(encoding, 0)
        if len(docstrings) != 1: return
        doclines = docstrings[0]
        setattr(self, '__new_doclines', doclines)
        if not doclines: return

        if sig: sigs = [sig]
        else: sigs = []
        generic_sig = False
        todel = []
        for lineno, docline in enumerate(doclines):
            match = autodoc.py_ext_sig_re.match(docline)
            if match:
                exmod, path, base, args, retan = match.groups()
                # the base name must match ours
                if self.objpath and base == self.objpath[-1]:
                    todel.append(lineno)
                    sg = "(%s)" % (args)
                    sigs.append(fix_signature(self.objtype, sg))
            else:
                match = boost_signature_re.match(docline)
                if match:
                    retan, base = match.groups()
                    if self.objpath and base == self.objpath[-1]:
                        todel.append(lineno)
                        generic_sig = True

        # delete doclines with signatures
        todel.reverse()
        for i in todel:
            del doclines[i]
            while i < len(doclines) and not doclines[i].strip():
                del doclines[i]

        if not sigs and generic_sig:
            sigs = ['(*args, **kwargs)']

        # unindent docstring
        doclines = prepare_docstring("\n".join(doclines), 0)

        setattr(self, '__new_doclines', doclines)

        return sigs

    def add_directive_header(self, sig):
        domain = getattr(self, 'domain', 'py')
        directive = getattr(self, 'directivetype', self.objtype)
        name = self.format_name()
        if self.sigs:
            self.add_line(u'.. %s:%s:: %s%s' %
                         (domain, directive, name, self.sigs[0]), '<autodoc>')
            for s in self.sigs[1:]:
                self.add_line(u'   %s%s' % (name, s),
                              '<autodoc>')
        else:
            self.add_line(u'.. %s:%s:: %s%s' % (domain, directive, name, sig),
                          '<autodoc>')
        if self.options.noindex:
            self.add_line(u'   :noindex:', '<autodoc>')
        if self.objpath:
            # Be explicit about the module, this is necessary since .. class::
            # etc. don't support a prepended module name
            self.add_line(u'   :module: %s' % self.modname, '<autodoc>')

        if isinstance(self, autodoc.ClassDocumenter):
            # add inheritance info, if wanted
            if not self.doc_as_attr and self.options.show_inheritance:
                self.add_line(u'', '<autodoc>')
                if len(self.object.__bases__):
                    bases = [b.__module__ == '__builtin__' and
                            u':class:`%s`' % b.__name__ or
                            u':class:`%s.%s`' % (b.__module__, b.__name__)
                            for b in self.object.__bases__]
                    self.add_line(autodoc._(u'   Bases: %s') % ', '.join(bases),
                                '<autodoc>')

        elif isinstance(self, autodoc.AttributeDocumenter):
            if not self._datadescriptor:
                try:
                    objrepr = autodoc.safe_repr(self.object)
                except ValueError:
                    pass
                else:
                    self.add_line(u'   :annotation: = ' + objrepr, '<autodoc>')


    def format_signature(self):
        sig = super(MultiSigMixin, self).format_signature()
        self.sigs = self._find_signatures(sig)
        if self.sigs:
            if len(self.sigs) > 1: sig = '(...)'
            elif not sig: sig = self.sigs[0]
        return sig

    def get_doc(self, encoding=None, ignore=1):
        lines = getattr(self, '__new_doclines', None)
        if lines is not None:
            return [lines]
        return autodoc.Documenter.get_doc(self, encoding, ignore)


class CppMethodDocumenter(MultiSigMixin, autodoc.MethodDocumenter):
    objtype = "method"


class CppFunctionDocumenter(MultiSigMixin, autodoc.FunctionDocumenter):
    objtype = "function"


class CppClassDocumenter(MultiSigMixin, autodoc.ClassDocumenter):
    objtype = "class"


class CppAttributeDocumenter(MultiSigMixin, autodoc.AttributeDocumenter):
    objtype = "attribute"
    option_spec = {'noindex': autodoc.bool_option, 'show-signature': autodoc.bool_option}

    def format_signature(self):
        if 'show-signature' in self.options:
            return MultiSigMixin.format_signature(self)
        else:
            self.sigs = []
            return autodoc.AttributeDocumenter.format_signature(self)

    def get_doc(self, encoding=None, ignore=1):
        if 'show-signature' in self.options:
            return MultiSigMixin.get_doc(self, encoding, ignore)
        else:
            return autodoc.AttributeDocumenter.get_doc(self, encoding, ignore)


# Hooks modifying default output for autosummary

arg1_in_signature_re = re.compile(r'\( \(\w*?\)(arg1|self)(, )?')
var_type_pattre = re.compile(r'\(\w*?\)(\w+)')


def fix_plask_namespace(signature):
   '''Remove all "_plask." prefixes.
      _plask module is loaded by default in plask.
   '''
   return signature.replace('_plask', '')


def fix_signature(what, signature):

    # remove optional arguments marks (Sphinx will add them later)
    signature = signature.replace(' [,', ',').replace('[', '').replace(']', '')

    # remove first argument, which is self named as arg1:
    if (what == 'method' or what == 'class'):
        # remove "(sth)arg1" and "(sth)arg1, "
        signature = re.sub(arg1_in_signature_re, r'(', signature)

    # change: (type)var -> var:
    signature = re.sub(var_type_pattre, r'\1', signature)

    return signature


def process_signature(app, what, name, obj, options, signature, retann):
    if not signature: return signature, None
    signature = fix_plask_namespace(signature)
    if (what == 'class' or what == 'method' or what == 'function'):
        return fix_signature(what, signature), None
    else:
        return signature, None


def process_docstr(app, what, name, obj, options, lines):
    if not lines: return
    for index, l in enumerate(lines):
        l = fix_plask_namespace(l)


# Register everything

def setup(app):
    app.add_autodocumenter(CppMethodDocumenter)
    app.add_autodocumenter(CppFunctionDocumenter)
    app.add_autodocumenter(CppClassDocumenter)
    app.add_autodocumenter(CppAttributeDocumenter)

    app.connect('autodoc-process-docstring', process_docstr)
    app.connect('autodoc-process-signature', process_signature)
