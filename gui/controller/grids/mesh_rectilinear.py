from ..base import Controller
from ...qt import QtGui


class AxisEdit(QtGui.QGroupBox):

    def __init__(self, title, allow_type_select = False, accept_non_regular = False):
        super(AxisEdit, self).__init__(title)
        form_layout = QtGui.QFormLayout()
        self.allow_type_select = allow_type_select
        if not allow_type_select: self.accept_non_regular = accept_non_regular
        if self.allow_type_select:
            self.type = QtGui.QComboBox()
            self.type.addItems(['(auto-detected)', 'ordered', 'regular'])
            form_layout.addRow("type", self.type)
        self.start = QtGui.QLineEdit()
        form_layout.addRow("start", self.start)
        self.stop = QtGui.QLineEdit()
        form_layout.addRow("stop", self.stop)
        self.step = QtGui.QLineEdit()
        form_layout.addRow("step", self.step)
        if allow_type_select or accept_non_regular:
            self.points = QtGui.QTextEdit()
            #self.points.setWordWrapMode(QtGui.QTextEdit.LineWrapMode)
            form_layout.addRow("points", self.points)
        if allow_type_select:
            self.points.setVisible(self.are_points_editable())
        self.setLayout(form_layout)

    def are_points_editable(self):
        if self.allow_type_select:
            return self.type.currentText() != 'regular'
        else:
            return self.accept_non_regular

    def to_model(self, axis_model):
        if self.allow_type_select:
            if self.type.currentIndex() == 0:
                axis_model.type = ''
            else:
                axis_model.type = self.type.currentText()
        axis_model.start = self.start.text()
        axis_model.stop = self.stop.text()
        axis_model.step = self.step.text()
        if self.are_points_editable():
            #axis_model.type = self.type.get
            axis_model.points = self.points.toPlainText()
        else:
            axis_model.points = ''

class RectangularMeshConroller(Controller):
    """2D and 3D rectangular mesh controller"""

    def __init__(self, document, model):
        super(RectangularMeshConroller, self).__init__(document=document, model=model)

        self.form = QtGui.QGroupBox()

        vbox = QtGui.QVBoxLayout()
        for i in range(0, model.dim):
            vbox.addWidget(AxisEdit(model.axis_tag_name(i), allow_type_select=True))
        vbox.addStretch()
        self.form.setLayout(vbox)

    def get_editor(self):
        return self.form
