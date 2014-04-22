LAUNCHERS = []

from .qt import QtGui

class LaunchDialog(QtGui.QDialog):

    def __init__(self, parent=None):
        super(LaunchDialog, self).__init__(parent)
        self.setWindowTitle("Launch Computations")

        self.layout = QtGui.QVBoxLayout()
        self.setLayout(self.layout)

        combo = QtGui.QComboBox()
        combo.insertItems(len(LAUNCHERS), [item.name for item in LAUNCHERS])
        combo.currentIndexChanged.connect(self.launcher_changed)
        self.layout.addWidget(combo)

        self.launcher_widgets = [l.widget() for l in LAUNCHERS]
        for widget in self.launcher_widgets:
            widget.setVisible(False)
            self.layout.addWidget(widget)
        self.current = combo.currentIndex()
        self.launcher_widgets[self.current].setVisible(True)

        self.setMinimumWidth(1.5*QtGui.QFontMetrics(QtGui.QFont()).width(self.windowTitle()))

        buttons = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui. QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons)

    def launcher_changed(self, index):
        self.launcher_widgets[self.current].setVisible(False)
        self.current = index
        self.launcher_widgets[self.current].setVisible(True)
        self.resize(self.width(), 0)

def launch_plask(filename):
    dialog = LaunchDialog()
    if dialog.exec_() == QtGui.QDialog.Accepted:
        launcher = LAUNCHERS[dialog.current]
        launcher.launch(filename)