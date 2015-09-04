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


class Attr(object):
    def __init__(self, name, label, help):
        self.name = name
        self.label = label
        self.help = help


class AttrMulti(Attr):
    pass


class AttrChoice(Attr):
    def __init__(self, name, label, help, choices):
        super(AttrChoice, self).__init__(name, label, help)
        self.choices = choices


class AttrBool(AttrChoice):
    def __init__(self, name, label, help):
        super(AttrBool, self).__init__(name, label, help, ('yes', 'no'))


class AttrGeometryObject(Attr):
    pass


class AttrGeometryPath(Attr):
    pass


def read_attr(attr, xns):
        an = attr.attrib['name']
        al = attr.attrib['label']
        ah = attr.text
        at = attr.attrib.get('type', '')
        au = attr.attrib.get('unit', None)
        if au is not None:
            al += u' [{}]'.format(au)
            at += u' [{}]'.format(au)
        if at == u'choice':
            ac = tuple(ch.text.strip() for ch in attr.findall(xns+'choice'))
            return AttrChoice(an, al, ah, ac)
        elif at == u'bool':
            return AttrBool(an, al, ah)
        elif at == u'geometry object':
            return AttrGeometryObject(an, al, ah)
        elif at == u'geometry path':
            return AttrGeometryPath(an, al, ah)
        else:
            if at:
                ah += u' ({})'.format(at)
            if an.endswith('#'):
                return AttrMulti(an, al, ah)
            else:
                return Attr(an, al, ah)
