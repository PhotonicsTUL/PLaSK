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
        self.externalSource = None
        if info_cb: self.infoChanged.connect(info_cb)
        
    def fireChanged(self):
        self.changed(self)

    def getText(self):
        element = self.getXMLElement()
        text = element.text.lstrip('\n') if element.text else ''
        for c in element:
            text += ElementTree.tostring(c)
        return text
        #return ElementTree.tostring(self.getXMLElement())

    def setText(self, text):
        self.setXMLElement(ElementTree.fromstringlist(['<', self.name, '>', text, '</', self.name, '>']))
        
    def fireInfoChanged(self):
        self.infoChanged.call(self)
        
    def setInfo(self, *info):
        self.errors = list(info)
        self.fireInfoChanged()
        
    def isReadOnly(self):
        """
            :return: true if model is read-only (typically: has been read from external source)
        """
        return self.externalSource != None
    
    def getFileXMLElement(self):
        if self.externalSource != None:
            return ElementTree.Element(self.name, { "external": self.externalSource })
        else:
            return self.getXMLElement()
        
    def clear(self):
        self.setText('')
        self.fireChanged()
        
    def setFileXMLElement(self, element):
        if 'external' in element.attrib:
            self.externalSource = element.attrib['external']
            self.clear() #TODO element = load external content
            return       #and remove this two lines
        self.setXMLElement(element)   

class SectionModelTreeBased(SectionModel):

    def __init__(self, name):
        SectionModel.__init__(self, name)
        self.element = ElementTree.Element(name)

    def setXMLElement(self, element):
        self.element = element
        self.fireChanged()
        
    def clear(self):
        self.element.clear()
        self.fireChanged()

    # XML element that represents whole section
    def getXMLElement(self):
        return self.element



