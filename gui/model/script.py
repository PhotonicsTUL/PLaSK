from model.base import SectionModel
from lxml import etree as ElementTree

class ScriptModel(SectionModel):

    def __init__(self, info_cb = None):
        SectionModel.__init__(self, 'script', info_cb)
        self.code = ''

    def set_XML_element(self, element):
        self.set_text(element.text if element is not None else '')

    # XML element that represents whole section
    def get_XML_element(self):
        res = ElementTree.Element(self.name)
        res.text = ElementTree.CDATA(self.code)
        return res
    
    def get_text(self):
        return self.code

    def set_text(self, text):
        self.code = text
        self.fire_changed()

