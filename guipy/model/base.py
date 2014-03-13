# coding: utf8

from lxml import etree as ElementTree
from utils.signal import Signal
from info import Info
import os
from model.info import InfoSource
from utils.xml import print_interior

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
    """Store information about data source of section if the source is external (file name)"""
    
    def __init__(self, fileName, oryginalFileName = None):
        """
            :param str fileName: name of file with source of section (or reference to next file)
            :param str oryginalFileName: name of file, from which the XPL is read (used when fileName is relative)
        """
        object.__init__(self)
        self.fileName = fileName
        if oryginalFileName: fileName = os.path.join(os.path.dirname(oryginalFileName), fileName)
        self.fileNameAbs = os.path.abspath(fileName)

class TreeFragmentModel(InfoSource):
    """Base class for fragment of tree (with change signal and info)"""
    
    def __init__(self, info_cb = None):
        """
            :param info_cb: call when list of error has been changed with parameters: section name, list of errors
        """
        InfoSource.__init__(self, info_cb)
        self.changed = Signal()
        
    def fireChanged(self, refreshInfo = True):
        """
            Inform listeners that this section was changed.
            :param bool refreshInfo: only if True, info of this section will be refresh
        """
        if refreshInfo: self.markInfoInvalid()
        self.changed(self)
        if refreshInfo: self.fireInfoChanged()

    def getText(self):
        return print_interior(self.getXMLElement())
        #return ElementTree.tostring(self.getXMLElement())


class SectionModel(TreeFragmentModel):
    
    def __init__(self, name, info_cb = None):
        """
            :param str name: name of section
            :param info_cb: call when list of error has been changed with parameters: section name, list of errors
        """
        super(SectionModel, self).__init__(info_cb)
        self.name = name
        self.externalSource = None

    def setText(self, text):
        self.setXMLElement(ElementTree.fromstringlist(['<', self.name, '>', text.encode('utf-8'), '</', self.name, '>']))   # .encode('utf-8') wymagane (tylko) przez lxml
        
    def isReadOnly(self):
        """
            :return: true if model is read-only (typically: has been read from external source)
        """
        return self.externalSource != None
    
    def getFileXMLElement(self):
        """
            Get XML element ready to save in XPL document.
            It represents the whole section and either contains data or points to external source (has external attribute).
        """
        if self.externalSource != None:
            return ElementTree.Element(self.name, { "external": self.externalSource.fileName })
        else:
            return self.getXMLElement()
        
    def clear(self):
        """Make this section empty."""
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
        res = super(SectionModel, self).createInfo()
        if self.isReadOnly():
            res.append(Info('%s section is read-only' % self.name, Info.INFO))
        if self.externalSource != None:
            res.append(Info('%s section is loaded from external file "%s" ("%s")' % (self.name, self.externalSource.fileName, self.externalSource.fileNameAbs), Info.INFO))
            if hasattr(self.externalSource, 'error'):
                res.append(Info("Can't load section from external file: %s" % self.externalSource.error, Info.ERROR))
        return res
        
    def stubs(self):
        return ""

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



