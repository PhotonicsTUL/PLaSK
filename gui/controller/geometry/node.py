from .. import Controller
from ..defines import get_defines_completer
from ...model.geometry.reader import GNAligner
from ...qt import QtGui


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

    def construct_align_controllers(self, dim = None, add_to_current = True, *aligners_dir):
        ''':return: list of controllers pairs, first is combo box to select aligner type, second is line edit for its value'''
        if len(aligners_dir) == 0:
            return self.construct_align_controllers(dim, add_to_current, *self.node.aligners_dir())
        if dim is None: dim = self.node.children_dim
        positions = []
        axes_conf = self.node.get_axes_conf_dim(dim)
        for c in aligners_dir:
            position = QtGui.QComboBox()
            position.addItems(
                [('{} origin at' if i == 3 else '{} at').format(x)
                for i, x in enumerate(GNAligner.names(dim, axes_conf, c, False))]
            )
            position.currentIndexChanged.connect(self.after_field_change)
            pos_value = self.construct_line_edit()
            if add_to_current: self._get_current_form().addRow(position, pos_value)
            positions.append((position, pos_value))
        return positions

    def construct_point_controllers(self, row_name = None, dim = None):
        if dim is None: dim = self.node.dim
        hbox = QtGui.QHBoxLayout()
        res = tuple(self.construct_line_edit() for _ in range(0, dim))
        for i in range(0, dim):
            hbox.addWidget(res[i])
        group = QtGui.QGroupBox(self.form)
        group.setContentsMargins(0, 0, 0, 0)
        group.setLayout(hbox)
        if row_name:
            self._get_current_form().addRow(row_name, group)
            return res
        else:
            return res, group

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


class GNChildController(GNodeController):

    def __init__(self, document, model, node, child_node):
        super(GNChildController, self).__init__(document, model, node)
        self.child_node = child_node