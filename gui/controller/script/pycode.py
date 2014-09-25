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

from ...external.pycode import pyqtfrontend


class PyCode(pyqtfrontend.PyCode):

    def __init__(self, project_folder, textedit, filename=None, prefix=""):
        super(PyCode, self).__init__(project_folder, textedit, filename)
        self.set_prefix(prefix)

    def source(self):
        src, pos = super(PyCode, self).source()
        src = self.prefix + src
        pos = pos + len(self.prefix)
        return src, pos

    def set_prefix(self, prefix):
        self.prefix = prefix
        if len(self.prefix) > 0 and self.prefix[-1] != '\n':
            self.prefix += '\n'
