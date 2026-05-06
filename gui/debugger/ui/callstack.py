from ...qt.QtWidgets import *
from ...qt.QtGui import QColor

class CallStackPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.call_stack_widget = QTreeWidget()
        self.call_stack_widget.setHeaderLabels(["Function", "File", "Line"])
        self.call_stack_widget.setColumnCount(3)
        self.call_stack_widget.setColumnWidth(0, 60)
        self.call_stack_widget.setColumnWidth(1, 120)
        self.call_stack_widget.setColumnWidth(2, 20)
        self.call_stack_widget.setAlternatingRowColors(True)
        self.call_stack_widget.setRootIsDecorated(True)
        self.call_stack_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.call_stack_widget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.call_stack_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.call_stack_widget.setToolTip("Shows the current call stack and frame-local variables.")

        layout = QVBoxLayout(self)
        layout.addWidget(self.call_stack_widget)

    def update_call_stack(self, stack_data):
        self.call_stack_widget.clear()

        for frame in stack_data:
            func = frame.get("function", "?")
            file = frame.get("file", "?")
            line = frame.get("line", "?")
            locals_dict = frame.get("locals", {})

            parts = file.replace("\\", "/").split("/")
            short_file = "/".join(parts[-3:]) if len(parts) > 3 else file

            top = QTreeWidgetItem([
                func,
                short_file,
                str(line)
            ])
            self.call_stack_widget.addTopLevelItem(top)

            for key, value in locals_dict.items():
                self.add_variable_item(top, key, value)

            top.setExpanded(False)

    def add_variable_item(self, parent, name, value, max_str_len=100, max_items=50):
        # Truncate long strings for display
        display_value = value
        if isinstance(value, str) and len(value) > max_str_len:
            display_value = value[:max_str_len] + "... (truncated)"
        
        # Create the current tree item
        if isinstance(value, dict):
            item = QTreeWidgetItem([str(name), f"dict ({len(value)})"])
            item.setToolTip(1, str(value))
            item.setForeground(1, QColor("darkGreen"))
            # Add children for each key/value
            for i, key in enumerate(sorted(value.keys())):
                if i >= max_items:
                    QTreeWidgetItem(item, [f"... ({len(value)-max_items} more items)", ""])
                    break
                self.add_variable_item(item, key, value[key], max_str_len, max_items)

        elif isinstance(value, (list, tuple, set)):
            type_name = type(value).__name__
            item = QTreeWidgetItem([str(name), f"{type_name} ({len(value)})"])
            item.setToolTip(1, str(value))
            item.setForeground(1, QColor("darkBlue"))
            # Add children for each element
            for i, v in enumerate(value):
                if i >= max_items:
                    QTreeWidgetItem(item, [f"... ({len(value)-max_items} more)", ""])
                    break
                self.add_variable_item(item, f"[{i}]", v, max_str_len, max_items)

        else:
            # Simple value
            item = QTreeWidgetItem([str(name), repr(display_value)])
            item.setToolTip(1, str(value))
            # Color coding
            if isinstance(value, (int, float, complex)):
                item.setForeground(1, QColor("blue"))
            elif isinstance(value, str):
                item.setForeground(1, QColor("darkRed"))
            elif isinstance(value, bool):
                item.setForeground(1, QColor("darkMagenta"))
            elif value is None:
                item.setForeground(1, QColor("gray"))

        # Add to parent
        if isinstance(parent, QTreeWidget):
            parent.addTopLevelItem(item)
        else:
            parent.addChild(item)

        # Optionally expand top-level items
        if isinstance(parent, QTreeWidget):
            item.setExpanded(True)

    def format_var(self, name, value, indent=0, max_str_len=100, max_items=10):
            spacer = "  " * indent

            # Truncate long strings
            if isinstance(value, str) and len(value) > max_str_len:
                value = value[:max_str_len] + "... (truncated)"

            # Limit length of containers
            if isinstance(value, dict):
                lines = [f"{spacer}{name}: dict{{"]
                for i, k in enumerate(sorted(value.keys())):
                    if i >= max_items:
                        lines.append(f"{spacer}  ... ({len(value) - max_items} more items)")
                        break
                    lines.append(self.format_var(k, value[k], indent + 1))
                lines.append(f"{spacer}}}")
                return "\n".join(lines)
            elif isinstance(value, (list, tuple, set)):
                type_name = type(value).__name__
                lines = [f"{spacer}{name}: {type_name}["]
                for i, v in enumerate(value):
                    if i >= max_items:
                        lines.append(f"{spacer}  ... ({len(value) - max_items} more items)")
                        break
                    lines.append(self.format_var(f"[{i}]", v, indent + 1))
                lines.append(f"{spacer}]")
                return "\n".join(lines)
            else:
                return f"{spacer}{name}: {repr(value)}"
