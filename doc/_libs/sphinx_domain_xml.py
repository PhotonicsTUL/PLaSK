# -*- coding: utf-8 -*-
"""
    sphinx.domains.rst
    ~~~~~~~~~~~~~~~~~~

    The reStructuredText domain.

    :copyright: Copyright 2007-2011 by the Sphinx team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import re

from sphinx import addnodes
from sphinx.domains import Domain, ObjType
from sphinx.locale import l_, _
from sphinx.directives import ObjectDescription
from sphinx.roles import XRefRole
from sphinx.util.nodes import make_refnode
from sphinx.util.docfields import TypedField, Field

class ParsedName:
   def __init__(self, name, displ_name, desc, context):
      self.name = name              # raw tag name
      self.displ_name = displ_name  # display name, in format given by user
      self.desc = desc              # extra information
      self.context = context        # context in []

   def displ_name_with_desc(self):
      if len(self.desc) > 0: return "%s %s" % (self.displ_name, self.desc)
      return self.displ_name

   def ref_target(self):
        if not self.has_ref():
            return None
        if len(self.context) > 0:
            return '%s %s' % (self.name, self.context)
        return self.name

   def has_ref(self):
      return self.context != '[]'

def parse_tag(tagstr):
    """Parse a tag signature.

    Returns:
      name, display name, extra info, context (in [])
    """
    s = tagstr.strip()
    extra = ''
    if s.startswith('<'):
      displ_name, sep, extra = s.partition('>')
      if len(sep) == 0:
        self.state_machine.reporter.warning('bad format for XML tag, missing \'>\' in: %s' % tagstr)
        name = s
      else:
        name = displ_name.strip('<>/').split(None, 1)[0]
        displ_name += sep
    else:
      l = s.split(None, 1)
      name = displ_name = l[0]
      if len(l) > 1: extra = l[1]
    extra = extra.strip()
    context = ''
    if extra.endswith(']'):
      before, sep, c = extra.rpartition('[')
      if len(sep) > 0:
        extra = before.strip()
        context = sep + c
    return ParsedName(name, displ_name, extra, context)

class XMLTag(ObjectDescription):
    """
    Description of XML tag.
    """

    doc_field_types = [
        TypedField('attributes', label=l_('Attributes'),
                   names=('attribute', 'attr', 'parameter', 'param'),
                   typerolename='tag', typenames=('attrtype', 'paramtype', 'type')),
        #GroupedField('attributes', label=l_('Attributes'), rolename='tag',
        #             names=('attribute', 'attr', 'parameter', 'param'),
        #             can_collapse=True),
        Field('contents', label=l_('Contents'), has_arg=False,
              names=('contents', 'content', 'Contents', 'Content'))
    ]

    def add_target_and_index(self, name, sig, signode):
        # name is returned by handle_signature
        if name == None: return
        targetname = self.objtype + '-' + name
        if targetname not in self.state.document.ids:
            signode['names'].append(targetname)
            signode['ids'].append(targetname)
            signode['first'] = (not self.names)
            self.state.document.note_explicit_target(signode)

            objects = self.env.domaindata['xml']['objects']
            key = (self.objtype, name)
            if key in objects:
                self.state_machine.reporter.warning(
                    'duplicate description of %s %s, ' % (self.objtype, name) +
                    'other instance in ' + self.env.doc2path(objects[key]),
                    line=self.lineno)
            objects[key] = self.env.docname
        indextext = self.get_index_text(self.objtype, name)
        if indextext:
            self.indexnode['entries'].append(('single', indextext, targetname, ''))

    def get_index_text(self, objectname, name):
        if self.objtype == 'tag' and name:
            return _('%s (XML tag)') % name
        return ''

    def handle_signature(self, sig, signode):
        p = parse_tag(sig)
        signode += addnodes.desc_name(p.displ_name, p.displ_name)
        if len(p.desc) > 0:
            p.desc = ' ' + p.desc
            signode += addnodes.desc_addname(p.desc, p.desc)
        return p.ref_target()

class XMLRefRole(XRefRole):
   def process_link(self, env, refnode, has_explicit_title, title, target):
      p = parse_tag(target)
      target = p.ref_target()
      if has_explicit_title: return title, target
      return p.displ_name_with_desc(), target

#class ReSTRole(ReSTMarkup):
#    def handle_signature(self, sig, signode):
#        signode += addnodes.desc_name(':%s:' % sig, ':%s:' % sig)
#        return sig


class XMLDomain(Domain):
    """XML domain."""
    name = 'xml'
    label = 'XML'

    object_types = {
        'tag': ObjType(l_('tag'), 'tag'),
    }
    directives = {
        'tag': XMLTag,
    }
    roles = {
        'tag':  XMLRefRole(),
    }
    initial_data = {
        'objects': {},  # fullname -> docname, objtype
    }

    def clear_doc(self, docname):
        for (typ, name), doc in list(self.data['objects'].items()):
            if doc == docname:
                del self.data['objects'][typ, name]

    def resolve_xref(self, env, fromdocname, builder, typ, target, node, contnode):
        objects = self.data['objects']
        objtypes = self.objtypes_for_role(typ)
        for objtype in objtypes:
            if (objtype, target) in objects:
                return make_refnode(builder, fromdocname,
                                    objects[objtype, target],
                                    objtype + '-' + target,
                                    contnode, target + ' ' + objtype)

    def get_objects(self):
        for (typ, name), docname in self.data['objects'].items():
            yield name, name, typ, docname, typ + '-' + name, 1

def setup(app):
    app.add_domain(XMLDomain)

