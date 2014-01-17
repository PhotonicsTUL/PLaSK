from model.base import SectionModel
from xml.etree import ElementTree

class ScriptModel(SectionModel):

    def __init__(self, info_cb = None):
        SectionModel.__init__(self, 'script', info_cb)
        self.code = ''

    def setXMLElement(self, element):
        self.setText(element.text if isinstance(element, ElementTree.Element) else '')

    # XML element that represents whole section
    def getXMLElement(self):
        res = ElementTree.Element(self.tagname)
        res.text = self.code
        return res
    
    def getText(self):
        return self.code

    def setText(self, text):
        self.code = text
        self.fireChanged()

