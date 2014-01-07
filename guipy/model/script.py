from model.base import SectionModel
from xml.etree import ElementTree

class ScriptModel(SectionModel):

    def __init__(self):
        self.tagname = 'script'
        self.code = ''

    def setXMLElement(self, element):
        if isinstance(element, ElementTree.Element):
            self.code = element.text
        else:
            self.code = ''

    # XML element that represents whole section
    def getXMLElement(self):
        res = ElementTree.Element(self.tagname)
        res.text = self.code
        return res
    
    def getText(self):
        return self.code

    def setText(self, text):
        self.code = text

