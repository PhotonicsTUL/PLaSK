from ...qt.QtWidgets import *
from ...qt.QtCore import Qt
from ...qt import QtSignal

import json

class WatchedPanel(QWidget):

    update_watched_expressions = QtSignal(set)

    def __init__(self):
        super().__init__()
        self.watch_tree = QTreeWidget()
        self.watch_tree.setHeaderLabels(["Expression", "Value"])
        self.watch_tree.setColumnWidth(0, 250)
        self.watch_tree.setAlternatingRowColors(True)
        self.watch_tree.setRootIsDecorated(False)
        self.watch_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.watch_tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.watch_tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.watch_tree.setToolTip("Auto-evaluated watch expressions.")

        self.watch_input = QLineEdit()
        self.watch_input.setPlaceholderText("Enter watch expression…")

        self.watch_add_button = QPushButton("Add")
        self.watch_add_button.clicked.connect(self.add_expression)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.watch_tree)

        watch_input_layout = QHBoxLayout()
        watch_input_layout.addWidget(self.watch_input)
        watch_input_layout.addWidget(self.watch_add_button)

        layout.addLayout(watch_input_layout)

        self.watch_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.watch_tree.customContextMenuRequested.connect(self.open_context_menu)

    def add_expression(self):
        expr = self.watch_input.text()
        self.watch_tree.addTopLevelItem(
            QTreeWidgetItem([
                expr,
                None
            ])
        )
        self.update_expressions()

    def edit_expression(self, item):
        old_expr = item.text(0)
        new_expr, ok = QInputDialog.getText(
            self, "Edit Watch Expression", "Expression:", text=old_expr
        )

        if ok and new_expr:
            item.setText(0, new_expr)
            self.update_expressions()
        self.update_expressions()

    def delete_expression(self, item):
        index = self.watch_tree.indexOfTopLevelItem(item)
        self.watch_tree.takeTopLevelItem(index)
        self.update_expressions()

    def _get_watch_expressions(self):
        return [
            self.watch_tree.topLevelItem(i).text(0)
            for i in range(self.watch_tree.topLevelItemCount())
        ]

    def update_expressions(self):
        expressions = self._get_watch_expressions()
        self.update_watched_expressions.emit(expressions)

    def update_watch_list(self, values):
        for i in range(self.watch_tree.topLevelItemCount()):
                item = self.watch_tree.topLevelItem(i)
                expr = item.text(0)  # Column 0 = expression string
                
                if expr in values:
                    val = values[expr]
                    item.setText(1, str(val))  # Column 1 = value
                else:
                    item.setText(1, "<not available>")

    def open_context_menu(self, position):
        item = self.watch_tree.itemAt(position)
        if item is None:
            return

        menu = QMenu()

        edit_action = menu.addAction("Edit Expression")
        delete_action = menu.addAction("Delete Expression")

        action = menu.exec_(self.watch_tree.viewport().mapToGlobal(position))

        if action == edit_action:
            self.edit_expression(item)
        elif action == delete_action:
            self.delete_expression(item)
