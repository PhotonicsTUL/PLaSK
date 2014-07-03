from ..base import Controller
from ...qt import QtGui


class AxisEdit(QtGui.QGroupBox):

    def __init__(self, title, accept_non_regular = False):
        super(AxisEdit, self).__init__(title)
        form_layout = QtGui.QFormLayout()
        self.accept_non_regular = accept_non_regular
        if self.accept_non_regular:
            self.type = QtGui.QComboBox()
            self.type.addItems(['', 'ordered', 'regular'])
            form_layout.addRow("type", self.type)
        self.start = QtGui.QLineEdit()
        form_layout.addRow("start", self.start)
        self.stop = QtGui.QLineEdit()
        form_layout.addRow("stop", self.stop)
        self.step = QtGui.QLineEdit()
        form_layout.addRow("step", self.step)
        if self.accept_non_regular:
            self.points = QtGui.QTextEdit()
            self.points.setWordWrapMode(QtGui.QTextEdit.WidgetWidth)
            form_layout.addRow("points", self.points)
        self.setLayout(form_layout)

    def to_model(self, axis_model):
        axis_model.start = self.start.text()
        axis_model.stop = self.stop.text()
        axis_model.step = self.step.text()
        if self.accept_points:
            #axis_model.type = self.type.get
            axis_model.points = self.points.toPlainText()


class RectangularMeshConroller(Controller):

    def __init__(self, model):
        super(RectangularMeshConroller, self).__init__(model=model)

        self.form = QtGui.QGroupBox()

        vbox = QtGui.QVBoxLayout()
        for i in range(0, model.dim):
            vbox.addWidget(AxisEdit(model.axis[i]))
        self.form.setLayout(vbox)


    def get_editor(self):
        return self.form
