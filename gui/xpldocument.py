# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.


import shutil
from lxml import etree

import re
import numpy

basestring = str, bytes
try:
    import plask
except ImportError:
    plask = None

from .model import SectionModelTreeBased
from .controller.source import SourceEditController
from .controller.defines import DefinesController
if plask is not None:
    from .controller.geometry import GeometryController
from .controller.script import ScriptController, LOG_LEVELS
from .controller.multi import GUIAndSourceController, GeometryGUIAndSourceController
from .controller.solvers import SolversController
from .controller.connects import ConnectsController
from .controller import materials
from .controller.grids import GridsController
from .utils.xml import XMLparser, OrderedTagReader

from .utils.config import CONFIG


class XPLDocument:

    SECTION_NAMES = ['defines', 'materials', 'geometry', 'grids', 'solvers', 'connects', 'script']
    NAME = "PLaSK structure data"
    EXT = "xpl"

    coding = 'utf-8'

    def __init__(self, window, filename=None):
        self.window = window
        self.defines = GUIAndSourceController(DefinesController(self))
        self.materials = GUIAndSourceController(materials.MaterialsController(self))
        from . import _DEBUG
        if plask is not None:
            self.geometry = GeometryGUIAndSourceController(GeometryController(self))
        else:
            self.geometry = SourceEditController(self, SectionModelTreeBased(XPLDocument.SECTION_NAMES[2]))
        self.grids = GUIAndSourceController(GridsController(self))
        self.script = ScriptController(self)
        self.solvers = GUIAndSourceController(SolversController(self))
        self.connects = GUIAndSourceController(ConnectsController(self))
        self.loglevel = 'detail'
        self.other_changes = False  # changes which are not in undo stacks

        self.controllers = (
            self.defines,
            self.materials,
            self.geometry,
            self.grids,
            self.solvers,
            self.connects,
            self.script
        )
        for c in self.controllers:
            c.model.changed.connect(self.on_model_change)
            c.model.undo_stack.cleanChanged.connect(self.update_window_changed)
        self.filename = None
        self.set_clean()
        self.comments = [None] * (len(XPLDocument.SECTION_NAMES) + 1)
        if filename: self.load_from_file(filename)

        if _DEBUG and plask is not None:  # very useful!
            from .utils.modeltest import ModelTest
            ModelTest(self.model_by_name('geometry'))
        # self.undo_group = QUndoGroup()    # do we need this?
        # for c in self.controllers:
        #     self.undo_group.addStack(c.model.undo_stack)

    def update_window_changed(self, ignored_clean_flag=None):   # ignored_clean_flag enable this to connect with cleanChanged signal
        self.window.set_changed(not self.is_clean())

    def on_model_change(self, model, *args, **kwargs):
        """Slot called by model 'changed' signals when user edits any section model, do nothing by default."""
        #self.set_changed(True)
        #self.window.set_changed(self.is_clean())
        pass

    @staticmethod
    def _undo_stack_is_clean(controller):
        try:
            return controller.model.undo_stack.isClean()
        except RuntimeError:
            return True

    def is_clean(self):
        """
            Change if document is in the clean state (the same state as in file).
            :return bool: True only if document is in the clean state
        """
        return all(self._undo_stack_is_clean(c) for c in self.controllers) and not self.other_changes

    def is_changed(self):
        return not self.is_clean()

    def set_clean(self):
        """
            Mark the document (stacks of all sections) as clean.
        """
        self.other_changes = False
        for c in self.controllers:
            c.model.undo_stack.setClean()

    # TODO remove this method, call set_clean instead (where it is needed)
    def set_changed(self, changed=True):
        if not changed:
            self.set_clean()
        else:
            self.other_changes = True
        self.window.set_changed(self.is_changed())

    def load_from_file(self, filename):
        infile = open(filename, 'rb')
        tree = etree.parse(infile, XMLparser)
        self.loglevel = tree.getroot().attrib.get('loglevel', 'detail')
        with OrderedTagReader(tree.getroot()) as r:
            for i, name in enumerate(XPLDocument.SECTION_NAMES):
                element = r.get(name)
                if element is not None:
                    self.comments[i] = element.comments
                    self.model_by_index(i).load_file_xml_element(element, filename)
                else:
                    self.model_by_index(i).clear()
            self.comments[-1] = r.get_comments()
        self.filename = filename
        self.set_clean()

    def get_contents(self, sections=None, update_lines=None):
        """
        Get document content in XPL format.
        :param sections: if given, only selected sections will be included in result, list or string (in such case all sections up to given one will be included)
        :param bool update_lines: if True number of lines in model will be updated (True by default only if all sections are stored)
        :return str: document content in XPL format
        """
        if isinstance(sections, basestring):
            sections = XPLDocument.SECTION_NAMES[:XPLDocument.SECTION_NAMES.index(sections)+1]
        if update_lines is None: update_lines = sections is None or sections == self.SECTION_NAMES
        data = '<plask loglevel="{}">\n\n'.format(self.loglevel)
        current_line_in_file = 3
        for i, c in enumerate(self.controllers):
            if self.comments[i]:
                for cmt in self.comments[i]:
                    data += "<!--{}-->\n".format(cmt)
                data += "\n"
            if update_lines: c.update_line_numbers(current_line_in_file)
            section_string = c.get_contents()
            if section_string:
                lines_count = section_string.count('\n') + 1
                current_line_in_file += lines_count
                if sections is None or c.model.name in sections:
                    data += section_string + '\n'
                else:
                    data += '\n' * lines_count
        if self.comments[-1]:
            for cmt in self.comments[-1]:
                data += "<!--{}-->\n".format(cmt)
            data += "\n"
        data += '</plask>\n'
        return data

    def save_to_file(self, filename):
        # safe write: maybe inefficient, but does not destroy the file if the error occurs
        data = self.get_contents()
        if CONFIG['main_window/make_backup']:
            try:
                shutil.copyfile(filename, filename+'.bak')
            except (IOError, OSError):
                pass
        try:
            open(filename, 'w', encoding='utf8').write(data)
        except TypeError:
            open(filename, 'w').write(data.encode('utf8'))
        self.filename = filename
        self.set_clean()

    def controller_by_index(self, index):
        return self.controllers[index]

    def controller_by_name(self, section_name):
        return self.controllers[XPLDocument.SECTION_NAMES.index(section_name)]

    def model_by_index(self, index):
        return self.controller_by_index(index).model

    def model_by_name(self, section_name):
        return self.controller_by_name(section_name).model

    def get_info(self, level=None):
        """Get messages from all models, on given level (all by default)."""
        res = []
        for c in self.controllers: res.extend(c.model.get_info(level))
        return res

    def stubs(self):
        return '\n'.join(s for s in (c.model.stubs() for c in self.controllers) if s)

    #TODO undo support, in script undo stack
    def set_loglevel(self, loglevel):
        if isinstance(loglevel, int):
            loglevel = LOG_LEVELS[loglevel]
        loglevel = loglevel.lower()
        if self.loglevel != loglevel:
            # def loglevelsetter(s, v): s.loglevel = v
            # self.script.model.undo_stack.push(    #TODO script does not use this stack
            #     UndoCommandWithSetter(self.script.model, self, loglevelsetter, loglevel, self.loglevel, 'loglevel')
            # )
            self.loglevel = loglevel
            self.set_changed()

    # def set_loglevel_undoable(self, loglevel):
    #     loglevel = loglevel.lower()
    #     if self.loglevel != loglevel:
    #         self.script.model.undo_stack.


class FieldParser:
    subst = re.compile('\{([^}]*)\}')

    def __init__(self, document):
        self.defines = dict()
        for entry in document.defines.model.entries:
            self.defines[entry.name] = eval(self.subst.sub(r'\1', entry.value), numpy.__dict__, self.defines)

    def eval(self, value):
        for sub in self.subst.findall(value):
            value = self.subst.subn(str(eval(sub, numpy.__dict__, self.defines)), value, 1)[0]
        return value
