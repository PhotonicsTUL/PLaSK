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

from .qt import QtGui

from .controller.script import ScriptController

coding_re = re.compile(r"(?:\s*#[^\n]*\n)?\s*#[^\n]*coding[=:]\s*([-\w.]+)")

class _Dummy(object):
    pass


class PyDocument(object):

    SECTION_NAMES = ["script"]

    def __init__(self, window, filename=None):
        self.window = window
        self.script = ScriptController(self)
        self.script.model.changed.connect(self.on_model_change)
        self.script.model.line_in_file = 0
        self.controllers = (self.script,)
        self.materials = _Dummy()
        self.materials.model = None
        self.solvers = None
        self.filename = None
        self.set_changed(False)
        if filename: self.load_from_file(filename)

    def on_model_change(self, model, *args, **kwargs):
        """Slot called by model 'changed' signals when user edits any section model"""
        self.set_changed(True)

    def set_changed(self, changed=True):
        self.window.set_changed(changed)

    def load_from_file(self, filename):
        data = open(filename, 'r').read()
        m = coding_re.match(data)
        if m:
            coding = m.group(1)
        else:
            coding = 'utf8'
        self.script.model.set_text(data.decode(coding))
        self.filename = filename
        self.set_changed(False)

    def save_to_file(self, filename):
        text = self.script.model.get_text()
        m = coding_re.match(text)
        if m:
            coding = m.group(1)
            try:
                text = text.encode(coding)
            except UnicodeEncodeError:
                QtGui.QMessageBox.critical(None, "Error while saving file.",
                                           "The file could not be saved with the specified encoding '{}'.\n\n"
                                           "Please set the proper encoding and try again.".format(coding))
                return
        else:
            text = text.encode('utf8')
        try:
            shutil.copyfile(filename, filename+'.bak')
        except (IOError, OSError):
            pass
        open(filename, 'w').write(text)
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
