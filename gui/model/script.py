from lxml import etree as ElementTree
from ..utils.xml import require_no_children, require_no_attributes

from .base import SectionModel

class ScriptModel(SectionModel):

    def __init__(self, info_cb=None):
        SectionModel.__init__(self, 'script', info_cb)
        self.code = ''

    def set_XML_element(self, element):
        require_no_children(element)
        require_no_attributes(element)
        self.set_text(element.text.lstrip('\n\r') if element is not None else '')

    # XML element that represents whole section
    def get_XML_element(self):
        res = ElementTree.Element(self.name)
        res.text = ElementTree.CDATA('\n' + self.code)
        return res

    def get_text(self):
        return self.code

    def set_text(self, text):
        self.code = text
        self.fire_changed()

