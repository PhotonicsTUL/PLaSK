from ..base import Controller
from ...qt import QtGui
from utils.str import empty_to_none

class RectilinearDivideGeneratorConroller(Controller):
    """ordered and rectangular 2D and 3D divide generator controller"""

    def __init__(self, document, model):
        super(RectilinearDivideGeneratorConroller, self).__init__(document=document, model=model)

        self.form = QtGui.QGroupBox()

        vbox = QtGui.QVBoxLayout()

        self.gradual = QtGui.QComboBox()    #not checkbox to allow put defines {}
        self.gradual.addItems(['True', 'False'])
        self.gradual.setEditable(True)
        vbox.addRow("gradual", self.gradual)

        vbox.addStretch()
        self.form.setLayout(vbox)

    def save_data_in_model(self):
        pass    #TODO

    def on_edit_enter(self):
        pass    #TODO

    def get_editor(self):
        return self.form