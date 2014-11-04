from .. import Controller
from ..defines import get_defines_completer
from ...qt import QtGui
from ...utils.str import empty_to_none, none_to_empty


class GNodeController(Controller):

    def construct_line_edit(self, row_name = None, use_defines_completer = True):
        res = QtGui.QLineEdit()
        if use_defines_completer: res.setCompleter(self.defines_completer)
        if row_name: self.form_layout.addRow(row_name, res)
        res.editingFinished.connect(self.after_field_change)
        return res

    def construct_combo_box(self, row_name = None, items = [], editable = True):
        res = QtGui.QComboBox()
        res.setEditable(editable)
        res.addItems(items)
        if row_name: self.form_layout.addRow(row_name, res)
        res.editTextChanged.connect(self.after_field_change)
        return res

    def __init__(self, document, model, node):
        super(GNodeController, self).__init__(document=document, model=model)
        self.node = node

        self.defines_completer = get_defines_completer(document.defines.model, None)

        self.form = QtGui.QGroupBox()
        #self.vbox = QtGui.QVBoxLayout()
        self.form_layout = QtGui.QFormLayout()
        self.fill_form()

        #self.vbox.addStretch()
        self.form.setLayout(self.form_layout)

    @property
    def node_index(self):
        self.model.index_for_node(self.node)

    def fill_form(self):
        pass

    def after_field_change(self):
        self.save_data_in_model()
        index = self.node_index
        self.model.dataChanged.emit(index, index)
        self.model.fire_changed()

    def save_data_in_model(self):
        pass

    def on_edit_enter(self):
        pass

    def get_widget(self):
        return self.form