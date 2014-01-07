# coding: utf8

from xml.etree import ElementTree

class SectionModel(object):

    def __init__(self, tagName):
        object.__init__(self)
        self.tagname = tagName

    def getText(self):
        element = self.getXMLElement()
        text = ''
        #text = ''+element.text
        for c in element:
            text += ElementTree.tostring(c)
        if not text:
            return ''
        return text
        #return ElementTree.tostring(self.getXMLElement())

    def setText(self, text):
        self.setXMLElement(ElementTree.fromstringlist(['<?xml version="1.0"?>\n<', self.tagname, '>', text, '</', self.tagname, '>']))


class SectionModelTreeBased(SectionModel):

    def __init__(self, tagName):
        SectionModel.__init__(self, tagName)
        self.element = ElementTree.Element(tagName)

    def setXMLElement(self, element):
        if isinstance(element, ElementTree.Element):
            self.element = element
        else:
            self.element.clear()

    # XML element that represents whole section
    def getXMLElement(self):
        return self.element



