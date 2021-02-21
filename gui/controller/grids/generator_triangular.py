from . import GridController

from ...qt.QtWidgets import *
from ..defines import get_defines_completer
from ...utils.widgets import EditComboBox
from ...utils.qsignals import BlockQtSignals
from ...utils.str import empty_to_none, none_to_empty
import weakref


class TriangularTriangleGeneratorController(GridController):

    def __init__(self, document, model):
        super().__init__(document=document, model=model)

        self.form = QGroupBox()
        self.defines = get_defines_completer(self.document.defines.model, self.form)
        form_layout = QFormLayout()
        weakself = weakref.proxy(self)

        self.maxarea = QLineEdit()
        self.maxarea.editingFinished.connect(
            lambda: weakself._change_attr('maxarea', empty_to_none(weakself.maxarea.text())))
        self.maxarea.setCompleter(self.defines)
        self.maxarea.setToolTip(u'&lt;options <b>maxarea</b>=""&gt;<br/>'
                                u'A maximum triangle area constraint (float [µm²])')
        form_layout.addRow("Maximum triangle area [µm²]:", self.maxarea)

        self.minangle = QLineEdit()
        self.minangle.editingFinished.connect(
            lambda: weakself._change_attr('minangle', empty_to_none(weakself.minangle.text())))
        self.minangle.setCompleter(self.defines)
        self.minangle.setToolTip(u'&lt;options <b>minangle</b>=""&gt;<br/>'
                                u'A minimum angle (float [°])')
        form_layout.addRow("Minimum angle [°]:", self.minangle)

        self.full = EditComboBox()
        self.full.addItems(('', 'yes', 'no'))
        self.full.setEditable(True)
        self.full.editingFinished.connect(
            lambda: weakself._change_attr('full', empty_to_none(weakself.full.currentText())))
        self.full.setCompleter(self.defines)
        self.full.setToolTip(u'&lt;options <b>full</b>=""&gt;<br/>'
                             u'Fill the whole bounding box of the geometry (bool)')
        form_layout.addRow("Fill whole area:", self.full)

        self.form.setLayout(form_layout)

    def fill_form(self):
        super().fill_form()
        with BlockQtSignals(self.maxarea): self.maxarea.setText(none_to_empty(self.model.maxarea))
        with BlockQtSignals(self.minangle): self.minangle.setText(none_to_empty(self.model.minangle))
        with BlockQtSignals(self.full): self.minangle.setText(none_to_empty(self.model.full))

    def get_widget(self):
        return self.form

    def select_info(self, info):
        super().select_info(info)
        getattr(self, info.property).setFocus()
