# coding: utf8

from xml.etree import ElementTree
from utils import Signal
import os

def getSectionXMLFromFile(sectionName, fileName, oryginalFileName = None):
        """
            Load section from file.
            :param str sectionName: name of section
            :param str fileName: source file
            :param oryginalFileName: name of XPL file where fileName was given in external attribute (str or None)
            :return: XML Element without external attribute or None
        """
        usednames = set()
        if oryginalFileName:
            oryginalFileName = os.path.abspath(oryginalFileName)
            usednames.add(oryginalFileName)
            fileName = os.path.join(os.path.dirname(oryginalFileName), fileName)
        else:
            fileName = os.path.abspath(fileName)
        while True:
            el = ElementTree.parse(fileName).getroot().find(sectionName)
            if (el == None) or ('external' not in el.attrib): return el
            usednames.add(fileName)
            fileName = os.path.join(os.path.dirname(fileName), el.attrib['external'])
            if fileName in usednames: raise RuntimeError("Error while reading section \"%s\": circular reference was detected." % sectionName)

class SectionModel(object):
    
    def __init__(self, name, info_cb = None):
        """
            :param str name: name of section
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
        
    def reloadExternalSource(self, oryginalFileName = None):
        """
            Load section from external source.
            :param oryginalFileName: name of XPL file where self.externalSource was given in external attribute, used only for optimization in circular reference finding
        """
        try:
            self.setXMLElement(getSectionXMLFromFile(self.name, self.externalSourceAbs, oryginalFileName))
        except Exception as e:
            self.externalSourceLoadRaport = str(e) 
        else:
            del self.externalSourceLoadRaport
        
    def setExternalSource(self, fileName, oryginalFileName = None):
        self.externalSource = fileName
        if oryginalFileName:
            fileName = os.path.join(os.path.dirname(oryginalFileName), fileName)
        self.externalSourceAbs = os.path.abspath(fileName)
        self.reloadExternalSource(oryginalFileName)
            
    def setFileXMLElement(self, element, fileName = None):
        if 'external' in element.attrib:
            self.setExternalSource(element.attrib['external'], fileName)
            return
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



