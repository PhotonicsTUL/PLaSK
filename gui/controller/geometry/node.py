from .. import Controller
from ..defines import get_defines_completer
from ...qt import QtGui
from ...utils.str import empty_to_none, none_to_empty


class GNodeController(Controller):

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

    def fill_form(self):
        pass

    def save_data_in_model(self):
        pass

    def on_edit_enter(self):
        pass

    def get_widget(self):
        return self.form