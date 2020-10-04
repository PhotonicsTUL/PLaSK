from gui.qt.QtWidgets import *
from gui.qt.QtGui import *
from gui.model.solvers import Tag
from gui.utils.widgets import MultiLineEdit


def open_editor(data, document):
    dialog = QDialog()
    layout = QVBoxLayout()
    attr = QLineEdit()
    layout.addWidget(attr)
    items = MultiLineEdit(movable=True, document=document)
    layout.addWidget(items)
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.setLayout(layout)

    attr.setText(data.attrs.get('attr'))
    items.set_values(t.name for t in data.tags)

    if dialog.exec_() == QDialog.Accepted:
        a = attr.text()
        return Tag(data.name, [Tag(i) for i in items.get_values()], {'attr': a} if a else {})
