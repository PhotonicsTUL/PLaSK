# coding: utf8
import os
from lxml import etree as ElementTree

from ..model.info import InfoSource
from ..utils.signal import Signal
from ..utils.xml import print_interior, XML_parser, AttributeReader
from .info import Info

def getSectionXMLFromFile(section_name, filename, original_filename=None):
        """
            Load section from file.
            :param str section_name: name of section
            :param str filename: source file
            :param original_filename: name of XPL file where filename was given in external attribute (str or None)
            :return: XML Element without external attribute or None
        """
        usednames = set()
        if original_filename:
            original_filename = os.path.abspath(original_filename)
            usednames.add(original_filename)
            filename = os.path.join(os.path.dirname(original_filename), filename)
        else:
            filename = os.path.abspath(filename)
        while True:
            el = ElementTree.parse(filename).getroot().find(section_name)
            if (el is None) or ('external' not in el.attrib): return el
            usednames.add(filename)
            filename = os.path.join(os.path.dirname(filename), el.attrib['external'])
            if filename in usednames: raise RuntimeError("Error while reading section \"%s\": circular reference was detected." % section_name)

class ExternalSource(object):
    """Store information about data source of section if the source is external (file name)"""

    def __init__(self, filename, original_filename=None):
        """
            :param str filename: name of file with source of section (or reference to next file)
            :param str original_filename: name of file, from which the XPL is read (used when filename is relative)
        """
        object.__init__(self)
        self.filename = filename
        if original_filename: filename = os.path.join(os.path.dirname(original_filename), filename)
        self.abs_filename = os.path.abspath(filename)

class TreeFragmentModel(InfoSource):
    """Base class for fragment of tree (with change signal and info)"""

    def __init__(self, parent=None, info_cb=None):
        """
            :param info_cb: call when list of error has been changed with parameters: section name, list of errors
        """
        InfoSource.__init__(self, info_cb)
        self.changed = Signal()
        self.parent = parent

    def fire_changed(self, refresh_info=True, *args, **kwargs):
        """
            Inform listeners that this section was changed.
            :param bool refresh_info: only if True, info of this section will be refresh
        """
        if refresh_info: self.markInfoInvalid()
        if 'src' not in kwargs: kwargs['src'] = self
        self.changed(self, *args, **kwargs)
        if refresh_info: self.fireInfoChanged()
        if self.parent is not None: self.parent.fire_changed(refresh_info, *args, **kwargs)

    def get_text(self):
        return print_interior(self.get_XML_element())
        #return ElementTree.tostring(self.get_XML_element())

    def is_read_only(self):
        """
            Check if model is read only.
            It just return parent.is_read_only(), which is valid implementation from internal section models (SectionModel overwrite it).
            :return: true if model is read-only (typically: has been read from external source)
        """
        return self.parent.is_read_only()

class SectionModel(TreeFragmentModel):

    def __init__(self, name, info_cb=None):
        """
            :param str name: name of section
            :param info_cb: call when list of error has been changed with parameters: section name, list of errors
        """
        super(SectionModel, self).__init__(info_cb=info_cb)
        self.name = name
        self.externalSource = None

    def set_text(self, text):
        self.set_XML_element(
            ElementTree.fromstringlist(['<', self.name.encode('utf-8'), '>', text.encode('utf-8'), '</',
                                        self.name.encode('utf-8'), '>'], parser=XML_parser))
                                        # .encode('utf-8') wymagane (tylko) przez lxml

    def is_read_only(self):
        return self.externalSource is not None

    def get_file_XML_element(self):
        """
            Get XML element ready to save in XPL document.
            It represents the whole section and either contains data or points to external source (has external attribute).
        """
        if self.externalSource is not None:
            return ElementTree.Element(self.name, {"external": self.externalSource.filename})
        else:
            return self.get_XML_element()

    def clear(self):
        """Make this section empty."""
        self.set_text('')
        self.fire_changed()

    def reload_external_source(self, original_filename=None):
        """
            Load section from external source.
            :param original_filename: name of XPL file where self.externalSource was given in external attribute,
                   used only for optimization in circular reference finding
        """
        try:
            self.set_XML_element(getSectionXMLFromFile(self.name, self.externalSource.filenameAbs, original_filename))
        except Exception as e:
            self.externalSource.error = str(e)
        else:
            if hasattr(self.externalSource, 'error'): del self.externalSource.error

    def set_external_source(self, filename, original_filename=None):
        self.externalSource = ExternalSource(filename, original_filename)
        self.reload_external_source(original_filename)

    def set_file_XML_element(self, element, filename=None):
        with AttributeReader(element) as a:
            if 'external' in a:
                self.set_external_source(a['external'], filename)
                return
        self.set_XML_element(element)

    def create_info(self):
        res = super(SectionModel, self).create_info()
        if self.is_read_only():
            res.append(Info('%s section is read-only' % self.name, Info.INFO))
        if self.externalSource is not None:
            res.append(Info('{} section is loaded from external file "{}" ("{}")'
                            .format(self.name,
                                    self.externalSource.filename,
                                    self.externalSource.filenameAbs), Info.INFO))
            if hasattr(self.externalSource, 'error'):
                res.append(Info(u"Cannot load section from external file: {}"
                                .format(self.externalSource.error), Info.ERROR))
        return res

    def stubs(self):
        return ""

class SectionModelTreeBased(SectionModel):
    """Model of section which just hold XML ElementTree Element"""

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



