from ..base import Controller
from ...qt import QtGui

class AxisEdit(object):

    def __init__(self):
        super(AxisEdit, self).__init__()
        self.start = QtGui.QLineEdit()
        self.stop = QtGui.QLineEdit()
        self.step = QtGui.QLineEdit()
        self.points = QtGui.QTextEdit()
        self.points.setWordWrapMode(QtGui.QTextEdit.WidgetWidth)
        form_layout = QtGui.QFormLayout()
        form_layout.addRow("start", self.start)
        form_layout.addRow("stop", self.stop)
        form_layout.addRow("step", self.step)
        form_layout.addRow("points", self.points)

    def toModel(self, axis_model):
        axis_model.start = self.start.text()
        axis_model.stop = self.stop.text()
        axis_model.step = self.step.text()
        axis_model.points = self.points.toPlainText()

class RectilinearMeshConroller(Controller):

    def __init__(self, model):
        super(RectilinearMeshConroller, self).__init__(model = model)

        self.form = QtGui.QWidget()
        form_layout = QtGui.QFormLayout()
        #form_layout.addRow("Name:", self.name_edit)
        #form_layout.addRow("Type:", self.type_edit)
        #form_layout.addRow(self.method_edit_label, self.method_edit)

        self.form.setLayout(form_layout)

    def get_editor(self):
        return self.form
