# coding: utf8

import os
from lxml import etree as ElementTree

from ..model.info import InfoSource
from ..utils.signal import Signal
from ..utils.xml import print_interior, XML_parser
from .info import Info

def getSectionXMLFromFile(sectionName, filename, oryginalFileName=None):
        """
            Load section from file.
            :param str sectionName: name of section
            :param str filename: source file
            :param oryginalFileName: name of XPL file where filename was given in external attribute (str or None)
            :return: XML Element without external attribute or None
        """
        usednames = set()
        if oryginalFileName:
            oryginalFileName = os.path.abspath(oryginalFileName)
            usednames.add(oryginalFileName)
            filename = os.path.join(os.path.dirname(oryginalFileName), filename)
        else:
            filename = os.path.abspath(filename)
        while True:
            el = ElementTree.parse(filename).getroot().find(sectionName)
            if (el == None) or ('external' not in el.attrib): return el
            usednames.add(filename)
            filename = os.path.join(os.path.dirname(filename), el.attrib['external'])
            if filename in usednames: raise RuntimeError("Error while reading section \"%s\": circular reference was detected." % sectionName)

class ExternalSource(object):
    """Store information about data source of section if the source is external (file name)"""

    def __init__(self, filename, oryginalFileName = None):
        """
            :param str filename: name of file with source of section (or reference to next file)
            :param str oryginalFileName: name of file, from which the XPL is read (used when filename is relative)
        """
        object.__init__(self)
        self.filename = filename
        if oryginalFileName: filename = os.path.join(os.path.dirname(oryginalFileName), filename)
        self.filenameAbs = os.path.abspath(filename)

class TreeFragmentModel(InfoSource):
    """Base class for fragment of tree (with change signal and info)"""

    def __init__(self, info_cb = None):
        """
            :param info_cb: call when list of error has been changed with parameters: section name, list of errors
        """
        InfoSource.__init__(self, info_cb)
        self.changed = Signal()

    def fire_changed(self, refreshInfo = True):
        """
            Inform listeners that this section was changed.
            :param bool refreshInfo: only if True, info of this section will be refresh
        """
        if refreshInfo: self.markInfoInvalid()
        self.changed(self)
        if refreshInfo: self.fireInfoChanged()

    def get_text(self):
        return print_interior(self.get_XML_element())
        #return ElementTree.tostring(self.get_XML_element())


class SectionModel(TreeFragmentModel):

    def __init__(self, name, info_cb = None):
        """
            :param str name: name of section
            :param info_cb: call when list of error has been changed with parameters: section name, list of errors
        """
        super(SectionModel, self).__init__(info_cb)
        self.name = name
        self.externalSource = None

    def set_text(self, text):
        self.set_XML_element(ElementTree.fromstringlist(['<', self.name.encode('utf-8'), '>', text.encode('utf-8'), '</', self.name.encode('utf-8'), '>'], parser = XML_parser))   # .encode('utf-8') wymagane (tylko) przez lxml

    def is_read_only(self):
        """
            :return: true if model is read-only (typically: has been read from external source)
        """
        return self.externalSource != None

    def get_file_XML_element(self):
        """
            Get XML element ready to save in XPL document.
            It represents the whole section and either contains data or points to external source (has external attribute).
        """
        if self.externalSource != None:
            return ElementTree.Element(self.name, { "external": self.externalSource.filename })
        else:
            return self.get_XML_element()

    def clear(self):
        """Make this section empty."""
        self.set_text('')
        self.fire_changed()

    def reload_external_source(self, oryginalFileName = None):
        """
            Load section from external source.
            :param oryginalFileName: name of XPL file where self.externalSource was given in external attribute, used only for optimization in circular reference finding
        """
        try:
            self.set_XML_element(getSectionXMLFromFile(self.name, self.externalSource.filenameAbs, oryginalFileName))
        except Exception as e:
            self.externalSource.error = str(e)
        else:
            if hasattr(self.externalSource, 'error'): del self.externalSource.error

    def set_external_source(self, filename, oryginalFileName = None):
        self.externalSource = ExternalSource(filename, oryginalFileName)
        self.reload_external_source(oryginalFileName)

    def set_file_XML_element(self, element, filename = None):
        if 'external' in element.attrib:
            self.set_external_source(element.attrib['external'], filename)
            return
        self.set_XML_element(element)

    def create_info(self):
        res = super(SectionModel, self).create_info()
        if self.is_read_only():
            res.append(Info('%s section is read-only' % self.name, Info.INFO))
        if self.externalSource != None:
            res.append(Info('%s section is loaded from external file "%s" ("%s")' % (self.name, self.externalSource.filename, self.externalSource.filenameAbs), Info.INFO))
            if hasattr(self.externalSource, 'error'):
                res.append(Info("Can't load section from external file: %s" % self.externalSource.error, Info.ERROR))
        return res

    def stubs(self):
        return ""

class SectionModelTreeBased(SectionModel):

    def __init__(self, name):
        SectionModel.__init__(self, name)
        self.element = ElementTree.Element(name)

    def set_XML_element(self, element):
        self.element = element
        self.fire_changed()

    def clear(self):
        self.element.clear()
        self.fire_changed()

    # XML element that represents whole section
    def get_XML_element(self):
        return self.element



