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

import shutil
import re

from .qt.QtWidgets import *
from .controller.script import ScriptController
from .utils.config import CONFIG

coding_re_s = re.compile("(?:\\s*#[^\\n]*\\n)?\\s*#[^\\n]*coding[=:]\\s*([-\\w.]+)")
coding_re_b = re.compile(b"(?:\\s*#[^\\n]*\\n)?\\s*#[^\\n]*coding[=:]\\s*([-\\w.]+)")


class _Dummy(object):
    pass


class PyDocument(object):

    SECTION_NAMES = ["script"]
    NAME = "Python script"
    EXT = "py"

    def __init__(self, window, filename=None):
        self.window = window
        self.script = ScriptController(self)
        self.script.model.changed.connect(self.on_model_change)
        self.script.model.line_in_file = 0
        self.script.model.undo_stack.cleanChanged.connect(self.set_changed)
        self.controllers = (self.script,)
        self.materials = _Dummy()
        self.materials.model = None
        self.defines = None
        self.solvers = None
        self.filename = None
        self.coding = 'utf-8'
        self.loglevel = 'detail'
        self.set_changed(False)
        if filename: self.load_from_file(filename)

    def on_model_change(self, model, *args, **kwargs):
        """Slot called by model 'changed' signals when user edits any section model"""
        self.set_changed(True)

    def is_changed(self):
        return self.window.isWindowModified()

    def set_changed(self, changed=True):
        self.window.set_changed(changed)

    def load_from_file(self, filename):
        data = open(filename, 'rb').read()
        m = coding_re_b.match(data)
        if m:
            coding = m.group(1).decode('ascii')
        else:
            coding = 'utf8'
        self.script.model.set_text(data.decode(coding))
        self.filename = filename
        self.set_changed(False)

    def get_content(self):
        return self.script.model.get_text()

    def save_to_file(self, filename):
        text = self.get_content()
        m = coding_re_s.match(text)
        if m:
            coding = m.group(1)
            try:
                text.encode(coding)
            except UnicodeEncodeError:
                QMessageBox.critical(None, "Error while saving file.",
                                           "The file could not be saved with the specified encoding '{}'.\n\n"
                                           "Please set the proper encoding and try again.".format(coding))
                return
            self.coding = coding
        else:
            self.coding = 'utf-8'
        if CONFIG['main_window/make_backup']:
            try:
                shutil.copyfile(filename, filename+'.bak')
            except (IOError, OSError):
                pass
        try:
            open(filename, 'w', encoding=self.coding).write(text)
        except TypeError:
            open(filename, 'w', ).write(text.encode(self.coding))
        self.filename = filename
        self.set_changed(False)

    def controller_by_index(self, index):
        return self.script

    def controller_by_name(self, section_name):
        return self.script

    def get_info(self, level=None):
        """Get messages from all models, on given level (all by default)."""
        return self.script.model.get_info(level)

    def stubs(self):
        return ""

    def set_loglevel(self, loglevel):
        self.loglevel = loglevel