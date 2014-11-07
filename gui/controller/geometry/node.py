from .. import Controller
from ..defines import get_defines_completer
from ...qt import QtGui
from ...utils.str import empty_to_none, none_to_empty


class GNodeController(Controller):

    def _get_current_form(self):
        if not hasattr(self, '_current_form'): self.construct_group()
        return self._current_form

    def construct_line_edit(self, row_name = None, use_defines_completer = True):
        res = QtGui.QLineEdit()
        if use_defines_completer: res.setCompleter(self.defines_completer)
        if row_name: self._get_current_form().addRow(row_name, res)
        res.editingFinished.connect(self.after_field_change)
        return res

    def construct_combo_box(self, row_name = None, items = [], editable = True):
        res = QtGui.QComboBox()
        res.setEditable(editable)
        res.addItems(items)
        if row_name: self._get_current_form().addRow(row_name, res)
        res.editTextChanged.connect(self.after_field_change)
        return res

    def construct_group(self, title = None, position = None):
        external = QtGui.QGroupBox(self.form)
        form_layout = QtGui.QFormLayout(external)
        if title is not None:
            external.setTitle(title)
            m = external.getContentsMargins()
            external.setContentsMargins(0, m[1], 0, m[3])
        else:
            external.setContentsMargins(0, 0, 0, 0)
        if position is None:
            self.vbox.addWidget(external)
        else:
            self.vbox.insertWidget(position, external)
        external.setLayout(form_layout)
        self._current_form = form_layout
        return form_layout

    def __init__(self, document, model, node):
        super(GNodeController, self).__init__(document=document, model=model)
        self.node = node

        self.defines_completer = get_defines_completer(document.defines.model, None)

        self.form =  QtGui.QGroupBox()
        self.form.setContentsMargins(0, 0, 0, 0)

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.setContentsMargins(0, 0, 0, 0)
        self.vbox.setSpacing(0)
        self.form.setLayout(self.vbox)
        self.fill_form()
        try:
            del self._current_form
        except:
            pass

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