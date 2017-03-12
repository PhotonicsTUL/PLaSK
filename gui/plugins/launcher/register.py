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

from uuid import getnode

import gui
from gui.qt.QtWidgets import *
from gui.qt.QtGui import *


class ShowSystemAction(QAction):

    def __init__(self, parent):
        super(ShowSystemAction, self).__init__(QIcon.fromTheme('material-external'),
                                               'Show System ID...', parent)
        self.triggered.connect(self.execute)

    def execute(self):
        systemid = "{:X}".format(getnode())
        clipboard = QApplication.clipboard()
        clipboard.setText(systemid)
        if clipboard.supportsSelection():
            clipboard.setText(systemid, QClipboard.Selection)
        msgbox = QMessageBox()
        msgbox.setWindowTitle("System ID")
        msgbox.setText("Your system ID is '{}'.                   ".format(systemid))
        msgbox.setInformativeText("The code has been copied to the clipboard.\n"
                                  "Send it to your PLaSK vendor in order to obtain a license file.")
        msgbox.setStandardButtons(QMessageBox.Ok)
        msgbox.setIcon(QMessageBox.Information)
        msgbox.exec_()

gui.HELP_ACTIONS.append(ShowSystemAction)
