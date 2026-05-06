from ...qt.QtWidgets import *
from ...qt.QtGui import QColor

class VariablesPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.variables_panel = QTreeWidget()
        self.variables_panel.setHeaderLabels(["Variable", "Value"])
        self.variables_panel.setColumnWidth(0, 200)
        self.variables_panel.setToolTip("Shows all current local variables and their values.")

        layout = QVBoxLayout(self)
        layout.addWidget(self.variables_panel)

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

    def update_vars(self, locals_dict):
        self.variables_panel.clear()
        for key, value in locals_dict.items():
            self.add_variable_item(self.variables_panel, key, value)
