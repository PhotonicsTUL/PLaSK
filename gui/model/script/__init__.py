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

from lxml import etree
from ...utils.xml import require_no_children, require_no_attributes

from .. import SectionModel
from .completer import prepare_completions


class ScriptModel(SectionModel):

    def __init__(self, info_cb=None):
        SectionModel.__init__(self, 'script', info_cb)
        self._code = ''
        prepare_completions()

    def set_xml_element(self, element):
        require_no_children(element)
        require_no_attributes(element)
        if element is not None:
            self.set_text(element.text[1:] if element.text[0] == '\n' else element.text)
        else:
            self.set_text('')

    # XML element that represents whole section
    def get_xml_element(self):
        res = etree.Element(self.name)
        res.text = etree.CDATA('\n' + self._code)
        return res

    def get_text(self):
        return self._code

    def set_text(self, text):
        self._code = text.expandtabs()
        self.fire_changed()

