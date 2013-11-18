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

# name description [context]
# name is required and can be in brackets <>
# both description and [context] are optional
# TODO support for context and <>
tag_sig_re = re.compile(r'(\S*)\s*(.*)$')	#(?:\s+\[(.*?)\])

def parse_tag(tagstr):
    """Parse a tag signature.

    Returns (tag name, description, context) string tuple.
    Empty string as description or context if are not available.
    """
    s = tagstr.strip()
    m = tag_sig_re.match(s)
    if not m:
        self.state_machine.reporter.warning('bad format for XML tag: %s' % tagstr)
        return (s, '', '')
    name, desc = m.groups()
    return (name, desc, '')

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
            self.indexnode['entries'].append(('single', indextext,
                                              targetname, ''))

    def get_index_text(self, objectname, name):
        if self.objtype == 'tag':
            return name
            #return _('%s (XML tag)') % name
        return ''

    def handle_signature(self, sig, signode):
        name, desc, context = parse_tag(sig)
        name_in_tag = '<%s>' % name
        signode += addnodes.desc_name(name_in_tag, name_in_tag)
        if len(desc) > 0:
            desc = ' ' + desc
            signode += addnodes.desc_addname(desc, desc)
        if len(context) > 0:
            return '%s [%s]' % name_in_tag, context
        return name_in_tag





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
        'tag':  XRefRole(),
    }
    initial_data = {
        'objects': {},  # fullname -> docname, objtype
    }

    def clear_doc(self, docname):
        for (typ, name), doc in list(self.data['objects'].items()):
            if doc == docname:
                del self.data['objects'][typ, name]

    def resolve_xref(self, env, fromdocname, builder, typ, target, node,
                     contnode):
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

