# coding: utf8

from xml.etree import ElementTree
from utils import Signal
from info import Info
import os
from model.info import InfoListModel

def getSectionXMLFromFile(sectionName, fileName, oryginalFileName=None):
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

class ExternalSource(object):
    
    def __init__(self, fileName, oryginalFileName = None):
        object.__init__(self)
        self.fileName = fileName
        if oryginalFileName: fileName = os.path.join(os.path.dirname(oryginalFileName), fileName)
        self.fileNameAbs = os.path.abspath(fileName)

class SectionModel(object):
    
    def __init__(self, name, info_cb = None):
        """
            :param str name: name of section
            :param info_cb: call when list of error has been changed with parameters: section name, list of errors
        """
        object.__init__(self)
        self.name = name
        self.changed = Signal()
        self.__info__ = []    #model Infos: Errors, Warnings and Informations
        self.infoChanged = Signal()
        self.externalSource = None
        if info_cb: self.infoChanged.connect(info_cb)
        
    def fireChanged(self, refreshInfo = True):
        if refreshInfo: self.markInfoInvalid()
        self.changed(self)
        if refreshInfo: self.fireInfoChanged()
        
    def markInfoInvalid(self):
        """Invalidate the info and it indexes."""
        self.__info__ = None
        
    def fireInfoChanged(self):
        """
            Inform observers that info has been changed.
            You must call markInfoInvalid and prepare data to build the info before call this.
        """
        self.infoChanged(self)

    def getText(self):
        element = self.getXMLElement()
        text = element.text.lstrip('\n') if element.text else ''
        for c in element:
            text += ElementTree.tostring(c)
        return text
        #return ElementTree.tostring(self.getXMLElement())

    def setText(self, text):
        self.setXMLElement(ElementTree.fromstringlist(['<', self.name, '>', text, '</', self.name, '>']))
        
    def isReadOnly(self):
        """
            :return: true if model is read-only (typically: has been read from external source)
        """
        return self.externalSource != None
    
    def getFileXMLElement(self):
        if self.externalSource != None:
            return ElementTree.Element(self.name, { "external": self.externalSource.fileName })
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
            self.setXMLElement(getSectionXMLFromFile(self.name, self.externalSource.fileNameAbs, oryginalFileName))
        except Exception as e:
            self.externalSource.error = str(e) 
        else:
            if hasattr(self.externalSource, 'error'): del self.externalSource.error
        
    def setExternalSource(self, fileName, oryginalFileName = None):
        self.externalSource = ExternalSource(fileName, oryginalFileName)
        self.reloadExternalSource(oryginalFileName)
            
    def setFileXMLElement(self, element, fileName = None):
        if 'external' in element.attrib:
            self.setExternalSource(element.attrib['external'], fileName)
            return
        self.setXMLElement(element)
        
    def createInfo(self):
        res = []
        if self.isReadOnly():
            res.append(Info('%s section is read-only' % self.name, Info.INFO))
        if self.externalSource != None:
            res.append(Info('%s section is loaded from external file "%s" ("%s")' % (self.name, self.externalSource.fileName, self.externalSource.fileNameAbs), Info.INFO))
            if hasattr(self.externalSource, 'error'):
                res.append(Info("Can't load section from external file: %s" % self.externalSource.error, Info.ERROR))
        return res
            
    def getInfo(self, level = None):
        if self.__info__ == None:
            self.__info__ = self.createInfo()
        if level != None:
            return filter(lambda m: m.level == level, self.__info__)
        else:
            return self.__info__
    
    def getInfoListModel(self):
        return InfoListModel(self)
    
    def refreshInfo(self):
        self.__info__ = None
        self.infoChanged(self)

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



