# coding: utf8
# Copyright (C) 2014 Photonics Group, Lodz University of Technology
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of GNU General Public License as published by the
# Free Software Foundation; either version 2 of the license, or (at your
# opinion) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
import os
from lxml import etree

from ..qt.QtCore import *
from ..qt.QtWidgets import *
from ..qt.QtGui import *
from .info import InfoSource, Info
from ..utils.signal import Signal
from ..utils.xml import print_interior, XMLparser, AttributeReader


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
            el = etree.parse(filename).getroot().find(section_name)
            if (el is None) or ('external' not in el.attrib): return el
            usednames.add(filename)
            filename = os.path.join(os.path.dirname(filename), el.attrib['external'])
            if filename in usednames: raise RuntimeError("Error while reading section \"%s\": circular reference was detected." % section_name)


class ExternalSource:
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
            :param TreeFragmentModel parent: parent in models tree
            :param info_cb: call when list of error has been changed with parameters: section name, list of errors
        """
        super(TreeFragmentModel, self).__init__(info_cb)
        self.changed = Signal()
        self.tree_parent = parent   # parent is not good name, due to its common use in Qt!

    def fire_changed(self, refresh_info=True, *args, **kwargs):
        """
            Inform listeners that this section was changed.
            :param bool refresh_info: only if True, info of this section will be refresh
        """
        if refresh_info: self.mark_info_invalid()
        if 'src' not in kwargs: kwargs['src'] = self
        self.changed(self, *args, **kwargs)
        if refresh_info: self.fire_info_changed()
        if self.tree_parent is not None: self.tree_parent.fire_changed(refresh_info, *args, **kwargs)

    def get_text(self):
        return print_interior(self.make_xml_element())
        #return etree.tostring(self.make_xml_element())

    def is_read_only(self):
        """
            Check if model is read only.
            It just return parent.is_read_only(), which is valid implementation from internal section models
            (SectionModel overwrite it).
            :return: true if model is read-only (typically: has been read from external source)
        """
        return self.tree_parent.is_read_only()


class SectionModel(TreeFragmentModel):
    """Base class for model of section (defines, geometry, ...)."""

    def __init__(self, name, info_cb=None, undo_stack=None, parent=None):
        """
            :param str name: name of section
            :param info_cb: call when list of error has been changed with parameters: section name, list of errors
            :param undo_stack: undo stack for model; if None it is creates
        """
        super(SectionModel, self).__init__(parent, info_cb=info_cb)
        self.name = name
        self.externalSource = None
        self.line_in_file = None
        if undo_stack is None:
            self.undo_stack = QUndoStack()
        else:
            self.undo_stack = undo_stack

    def create_undo_action(self, parent):
        """
            :param QObject parent: qt parent
            :return QAction: undo action connected with self.undo_stack
        """
        res = self.undo_stack.createUndoAction(parent)
        res.setIcon(QIcon.fromTheme('edit-undo'))
        res.setShortcut(Qt.CTRL + Qt.Key_Z)
        return res

    def create_redo_action(self, parent):
        """
            :param QObject parent: qt parent
            :return QAction: redo action connected with self.undo_stack
        """
        res = self.undo_stack.createRedoAction(parent)
        res.setIcon(QIcon.fromTheme('edit-redo'))
        res.setShortcut(Qt.CTRL + Qt.SHIFT + Qt.Key_Z)
        return res

    def set_text(self, text):
        if not isinstance(text, str): text = text.encode('utf8')
        name = self.name if isinstance(self.name, str) else self.name.encode('utf8')
        self.load_xml_element(
            etree.fromstringlist(['<', name, '>', text, '</',
                                  name, '>'], parser=XMLparser))

    def is_read_only(self):
        return self.externalSource is not None

    def make_file_xml_element(self):
        """
            Get XML element ready to save in XPL document.
            It represents the whole section and either contains data or points to external source (has external attribute).
        """
        if self.externalSource is not None:
            return etree.Element(self.name, {"external": self.externalSource.filename})
        else:
            return self.make_xml_element()

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
            self.load_xml_element(getSectionXMLFromFile(self.name, self.externalSource.filenameAbs, original_filename))
        except Exception as e:
            self.externalSource.error = str(e)
        else:
            if hasattr(self.externalSource, 'error'): del self.externalSource.error

    def set_external_source(self, filename, original_filename=None):
        self.externalSource = ExternalSource(filename, original_filename)
        self.reload_external_source(original_filename)

    def load_file_xml_element(self, element, filename=None):
        with AttributeReader(element) as a:
            if 'external' in a:
                self.set_external_source(a['external'], filename)
                return
        self.load_xml_element(element, undoable=False)
        self.line_in_file = element.sourceline   # TODO can be wrong when the next sections will not be read correctly

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
    """Model of section which just hold XML etree Element"""

    def __init__(self, name):
        SectionModel.__init__(self, name)
        self.element = etree.Element(name)

    def load_xml_element(self, element, undoable=True):
        #TODO undo support if this is used
        self.element = element
        self.fire_changed()

    def clear(self):
        self.element.clear()
        self.fire_changed()

    # XML element that represents whole section
    def make_xml_element(self):
        return self.element



