# coding: utf8

from xml.etree import ElementTree

class SectionModel(object):

    def __init__(self, name, errors_cb = None):
        """
            :param str tagName: name of section
            :param errors_cb: call when list of error has been changed with parameters: section name, list of errors
        """
        object.__init__(self)
        self.name = name
        self.errors = []    #non-critical errors in model
        self.errors_cb = errors_cb

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
        self.setXMLElement(ElementTree.fromstringlist(['<?xml version="1.0"?>\n<', self.name, '>', text, '</', self.name, '>']))
        
    def errorsChanged(self):
        if self.errors_cb: self.errors_cb(self.name, self.errors)
        
    def setErrors(self, *errors):
        self.errors = list(errors)
        self.errorsChanged()
        

class SectionModelTreeBased(SectionModel):

    def __init__(self, name):
        SectionModel.__init__(self, name)
        self.element = ElementTree.Element(name)

    def setXMLElement(self, element):
        if isinstance(element, ElementTree.Element):
            self.element = element
        else:
            self.element.clear()

    # XML element that represents whole section
    def getXMLElement(self):
        return self.element



