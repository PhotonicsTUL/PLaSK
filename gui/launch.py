LAUNCHERS = []

from PyQt4 import QtGui

class LaunchDialog(QtGui.QDialog):

    def __init__(self, parent=None):
        super(LaunchDialog, self).__init__(parent)
        self.setWindowTitle("Launch computations")

        combo = QtGui.QComboBox()
        combo.insertItems(len(LAUNCHERS), [item[0] for item in LAUNCHERS])

        layout = QtGui.QVBoxLayout()

        self.setLayout(layout)


def launch_plask(filename):
    dialog = LaunchDialog()
    dialog.exec_()
