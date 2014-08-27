from ..base import Controller
from ...qt import QtGui
from utils.str import empty_to_none

class RectilinearDivideGeneratorConroller(Controller):
    """ordered and rectangular 2D and 3D divide generator controller"""

    def __make_div_hbox__(self, container_to_add, label):
        hbox_div = QtGui.QHBoxLayout()
        res = tuple(QtGui.QLineEdit() for _ in range(0, self.model.dim))
        for i in range(0, self.model.dim):
            hbox_div.addWidget(QtGui.QLabel('by '+str(i)+':'))
            hbox_div.addWidget(res[i])
        container_to_add.addRow(label, hbox_div)
        return res

    def __init__(self, document, model):
        super(RectilinearDivideGeneratorConroller, self).__init__(document=document, model=model)

        self.form = QtGui.QGroupBox()

        vbox = QtGui.QVBoxLayout()
        form_layout = QtGui.QFormLayout()

        self.gradual = QtGui.QComboBox()    #not checkbox to allow put defines {}
        self.gradual.addItems(['True', 'False'])
        self.gradual.setEditable(True)
        form_layout.addRow("gradual", self.gradual)

        self.prediv = self.__make_div_hbox__(form_layout, "prediv")
        self.postdiv = self.__make_div_hbox__(form_layout, "postdiv")

        vbox.addLayout(form_layout)

        vbox.addStretch()
        self.form.setLayout(vbox)

    def save_data_in_model(self):
        pass    #TODO

    def on_edit_enter(self):
        pass    #TODO

    def get_editor(self):
        return self.form