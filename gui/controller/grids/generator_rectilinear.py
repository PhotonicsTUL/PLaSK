from ..base import Controller
from ..table import table_with_manipulators
from ...model.base import TreeFragmentModel
from ...model.grids.generator_rectilinear import RectilinearDivideGenerator
from ...qt import QtGui
from ...utils.str import empty_to_none

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
        self.gradual.addItems(['', 'yes', 'no'])
        self.gradual.setEditable(True)
        form_layout.addRow("gradual", self.gradual)

        self.prediv = self.__make_div_hbox__(form_layout, "prediv")
        self.postdiv = self.__make_div_hbox__(form_layout, "postdiv")

        warnings_layout = QtGui.QHBoxLayout()
        for w in RectilinearDivideGenerator.warnings:
            cb  = QtGui.QComboBox()
            cb.addItems(['', 'yes', 'no'])
            cb.setEditable(True)
            setattr(self, 'warning_'+w, cb)
            label = QtGui.QLabel(w+':')
            label.setSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Preferred)
            warnings_layout.addWidget(label)
            cb.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
            warnings_layout.addWidget(cb)
        form_layout.addRow('warnings', warnings_layout)

        vbox.addLayout(form_layout)

        self.refinements = QtGui.QTableView()
        self.refinements.setModel(model.refinements)
        vbox.addWidget(table_with_manipulators(self.refinements, title="Refinements:"))

        #vbox.addStretch()
        self.form.setLayout(vbox)

    def save_data_in_model(self):
        for attr_name in ['gradual'] + ['warning_'+w for w in RectilinearDivideGenerator.warnings]:
            setattr(self.model, attr_name, empty_to_none(getattr(self, attr_name).currentText()))
        self.model.set_prediv([empty_to_none(self.prediv[i].text()) for i in range(0, self.model.dim)])
        self.model.set_postdiv([empty_to_none(self.postdiv[i].text()) for i in range(0, self.model.dim)])

    def on_edit_enter(self):
        for attr_name in ['gradual'] + ['warning_'+w for w in RectilinearDivideGenerator.warnings]:
            a = getattr(self.model, attr_name)
            getattr(self, attr_name).setEditText('' if a is None else a)
        for i in range(0, self.model.dim):
            self.prediv[i].setText(self.model.get_prediv(i))
            self.postdiv[i].setText(self.model.get_postdiv(i))

    def get_editor(self):
        return self.form