from ..base import Controller
from ...qt import QtGui


class AxisEdit(QtGui.QGroupBox):

    def __init__(self, title, accept_points = False):
        super(AxisEdit, self).__init__(title)
        self.start = QtGui.QLineEdit()
        self.stop = QtGui.QLineEdit()
        self.step = QtGui.QLineEdit()
        form_layout = QtGui.QFormLayout()
        form_layout.addRow("start", self.start)
        form_layout.addRow("stop", self.stop)
        form_layout.addRow("step", self.step)
        self.accept_points = accept_points
        if self.accept_points:
            self.points = QtGui.QTextEdit()
            self.points.setWordWrapMode(QtGui.QTextEdit.WidgetWidth)
            form_layout.addRow("points", self.points)
        self.setLayout(form_layout)

    def to_model(self, axis_model):
        axis_model.start = self.start.text()
        axis_model.stop = self.stop.text()
        axis_model.step = self.step.text()
        if self.accept_points:
            axis_model.points = self.points.toPlainText()


class RectilinearMeshConroller(Controller):

    def __init__(self, model):
        super(RectilinearMeshConroller, self).__init__(model=model)

        self.form = QtGui.QGroupBox()

        vbox = QtGui.QVBoxLayout()
        for i in range(0, model.dim):
            vbox.addWidget(AxisEdit(model.axis[i]))
        self.form.setLayout(vbox)


    def get_editor(self):
        return self.form
