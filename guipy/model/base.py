# coding: utf8

from xml.etree import ElementTree
from utils import Signal

class SectionModel(object):

    def __init__(self, name, info_cb = None):
        """
            :param str tagName: name of section
            :param info_cb: call when list of error has been changed with parameters: section name, list of errors
        """
        object.__init__(self)
        self.name = name
        self.changed = Signal()
        self.info = []    #non-critical errors in model, maybe change to: Errors, Warnings and Informations
        self.infoChanged = Signal()
        if info_cb: self.infoChanged.connect(info_cb)
        
    def fireChanged(self):
        self.changed(self)

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
        
    def fireInfoChanged(self):
        self.infoChanged.call(self)
        
    def setInfo(self, *info):
        self.errors = list(info)
        self.fireInfoChanged()
        
    def isReadOnly(self):
        """
            :return: true if model is read-only (typically: has been read from external source)
        """
        return hasattr(self, 'externalSource')
        

class SectionModelTreeBased(SectionModel):

    def __init__(self, name):
        SectionModel.__init__(self, name)
        self.element = ElementTree.Element(name)

    def setXMLElement(self, element):
        if isinstance(element, ElementTree.Element):
            self.element = element
        else:
            self.element.clear()
        self.fireChanged()

    # XML element that represents whole section
    def getXMLElement(self):
        return self.element



