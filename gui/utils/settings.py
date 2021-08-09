# coding: utf-8
# Copyright (C) 2014 Photonics Group, Lodz University of Technology
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of GNU General Public License as published by the
# Free Software Foundation; either version 2 of the license, or (at your
# opinion) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import sys
import os
from collections import OrderedDict
from typing import Callable
from numpy import log10, ceil

try:
    import matplotlib
except ImportError:
    matplotlib = None

from .config import CONFIG, KEYBOARD_SHORTCUTS, parse_font, parse_highlight
from ..qt.QtCore import *
from ..qt.QtWidgets import *
from ..qt.QtGui import *
from ..qt import QT_API
from .qsignals import BlockQtSignals
from .widgets import LineEditWithClear, VerticalScrollArea, EDITOR_FONT


try:
    import yaml
except ImportError:
    yaml = None
else:
    # Disable yaml warning
    try:
        yaml.warnings({'YAMLLoadWarning': False})
    except (TypeError, NameError, AttributeError):
        pass

try:
    import plask
except ImportError:
    plask = None

basestring = str, bytes
PRESET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'presets')


def CheckBox(entry, help=None, needs_restart=False):
    return lambda parent: SettingsDialog.CheckBox(entry, help=help, parent=parent, needs_restart=needs_restart)

def Combo(entry, options, help=None, needs_restart=False):
    return lambda parent: SettingsDialog.Combo(entry, options, help=help, parent=parent, needs_restart=needs_restart)

def SpinBox(entry, min=None, max=None, help=None, needs_restart=False):
    return lambda parent: SettingsDialog.SpinBox(entry, min=min, max=max, help=help, parent=parent,
                                               needs_restart=needs_restart)

def FloatSpinBox(entry, step=None, min=None, max=None, help=None, needs_restart=False):
    return lambda parent: SettingsDialog.FloatSpinBox(entry, step=step, min=min, max=max, help=help, parent=parent,
                                                    needs_restart=needs_restart)

def Color(entry, help=None, needs_restart=False):
    return lambda parent: SettingsDialog.Color(entry, help=help, parent=parent, needs_restart=needs_restart)

def Syntax(entry, help=None, needs_restart=False):
    return lambda parent: SettingsDialog.Syntax(entry, help=help, parent=parent, needs_restart=needs_restart)

def Font(entry, help=None, needs_restart=False):
    return lambda parent: SettingsDialog.Font(entry, help=help, parent=parent, needs_restart=needs_restart)

def Path(entry, title, mask, help=None, needs_restart=False):
    return lambda parent: SettingsDialog.Path(entry, title, mask, help=help, parent=parent, needs_restart=needs_restart)


class MaterialColorsConfig(QWidget):

    entry = 'geometry/material_colors'
    caption = "Select custom colors for materials shown in geometry preview."
    needs_restart = False

    class TableModel(QAbstractTableModel):
        def __init__(self):
            super().__init__()
            self.colors = list(CONFIG[MaterialColorsConfig.entry].items())
            self.filter = None
        def flags(self, index):
            flags = super().flags(index) | Qt.ItemIsEnabled
            if index.column() == 0:
                flags |= Qt.ItemIsSelectable | Qt.ItemIsEditable
            else:
                flags &= ~Qt.ItemIsSelectable
            return flags
        def columnCount(self, parent=None):
            return 2
        def rowCount(self, parent=None):
            return len(self.colors)
        def headerData(self, section, orientation, role=Qt.DisplayRole):
            if role == Qt.DisplayRole and orientation == Qt.Horizontal:
                return ('Material', 'Color')[section]
        def data(self, index, role=Qt.DisplayRole):
            if index.column() == 0:
                if role == Qt.DisplayRole or role == Qt.EditRole:
                    return self.colors[index.row()][0]
            elif index.column() == 1:
                if role == Qt.BackgroundColorRole:
                    color = self.colors[index.row()][1]
                    return QColor(color) if color is not None else None
        def setData(self, index, value, role=Qt.EditRole):
            row = index.row()
            if index.column() == 0:
                self.colors[row] = value, self.colors[row][1]
                return True
            if index.column() == 1:
                self.colors[row] = self.colors[row][0], value
                return True
            return False
        def removeRows(self, row, count, parent=QModelIndex()):
            self.beginRemoveRows(parent, row, row+count-1)
            for i in range(count):
                del self.colors[row]
            self.endRemoveRows()
            return count > 0
        def insertRows(self, row, count, parent=QModelIndex()):
            self.beginInsertRows(parent, row, row+count-1)
            for i in range(count):
                self.colors.insert(row, ('', None))
            self.endInsertRows()
            return count > 0

    def __init__(self, items, parent=None):
        super().__init__(parent)
        toolbar = QHBoxLayout()
        add_action = QToolButton()
        add_action.setIcon(QIcon.fromTheme('list-add'))
        add_action.pressed.connect(self.add)
        toolbar.addWidget(add_action)
        remove = QToolButton()
        remove.setIcon(QIcon.fromTheme('list-remove'))
        remove.pressed.connect(self.remove)
        toolbar.addWidget(remove)
        toolbar.setAlignment(Qt.AlignLeft)

        self.model = self.TableModel()
        self.filter = QSortFilterProxyModel()
        self.filter.setSourceModel(self.model)
        self.filter.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.filter.setDynamicSortFilter(True)

        self.table = QTableView()
        self.table.setModel(self.filter)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)

        self.table.clicked.connect(self.select_color)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(self.caption)
        layout.addWidget(label)

        layout.addLayout(toolbar)
        layout.addWidget(self.table)
        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

        items[self.entry] = self

    def add(self):
        self.filter.setFilterFixedString('')
        row = self.model.rowCount()
        self.model.insertRows(row, 1)
        self.table.viewport().update()
        index = self.filter.index(row, 0)
        self.table.scrollTo(index)
        self.table.setCurrentIndex(index)
        self.table.scrollTo(index)
        self.table.edit(index)

    def remove(self):
        row = self.table.selectionModel().currentIndex().row()
        self.filter.removeRows(row, 1)
        self.filter.invalidateFilter()
        self.table.viewport().update()

    def select_color(self, index):
        if index.column() != 1: return
        row = index.row()
        dlg = QColorDialog(self.parent())
        color = self.filter.data(index, Qt.BackgroundColorRole)
        if color is not None: dlg.setCurrentColor(color)
        if dlg.exec_():
            self.filter.setData(index, dlg.currentColor().name())

    def load(self, value):
        self.model.colors = list(value.items())
        self.table.viewport().update()

    def save(self):
        CONFIG[self.entry] = {k: v for (k,v) in self.model.colors if v is not None}

    def filter_view(self, filter):
        if filter is None: filter = ''
        self.filter.setFilterFixedString(filter)
        return self.filter.rowCount() > 0


class PluginsConfig(QWidget):

    def __init__(self, config_items, parent=None):
        super().__init__(parent)
        self.plugins = []
        layout = QVBoxLayout()
        self.setLayout(layout)
        label = QLabel("Select active plugins. After making any changes here, you must restart PLaSK GUI.")
        layout.addWidget(label)
        # frame = VerticalScrollArea()
        frame = QScrollArea()
        frame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        frame.setBackgroundRole(QPalette.Base)
        frame.setAutoFillBackground(True)
        layout.addWidget(frame)
        inframe = QWidget()
        self.plugins_layout = QGridLayout()
        self.plugins_layout.setAlignment(Qt.AlignTop)
        self.plugins_layout.setHorizontalSpacing(8)
        self.plugins_layout.setVerticalSpacing(16)
        inframe.setLayout(self.plugins_layout)
        from .. import PLUGINS
        for plugin, name, desc in PLUGINS:
            entry = 'plugins/{}'.format(plugin)
            if CONFIG[entry] is None: CONFIG[entry] = True
            checkbox = SettingsDialog.CheckBox(entry, help=desc, needs_restart=True)
            checkbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            label = QLabel()
            label.setTextFormat(Qt.RichText)
            if desc is not None:
                label.setText('{}<br/><span style="font-size: small">{}</span>'.format(name, desc))
            else:
                label.setText(name)
            label.setBuddy(checkbox)
            self.plugins.append((checkbox, label))
        self.filter_view()
        frame.setWidget(inframe)

    def filter_view(self, filter=None):
        for _ in range(self.plugins_layout.count()):
            it = self.plugins_layout.takeAt(0)
            if it: it.widget().setParent(None)
        for row, (checkbox, label) in enumerate(self.plugins):
            if not filter or filter in label.text().lower():
                self.plugins_layout.addWidget(checkbox, row, 0)
                self.plugins_layout.addWidget(label, row, 1)
        return self.plugins_layout.count() > 0


def get_launchers():
    from ..launch import LAUNCHERS
    return [l.name for l in LAUNCHERS]


def keyboard_shortcut_editors():
    for entry, data in KEYBOARD_SHORTCUTS.items():
        label = data[0]
        yield label, lambda parent: SettingsDialog.KeySequence(entry, parent=parent)


CONFIG_WIDGETS = OrderedDict([
    ("General", OrderedDict([
        ("Appearance && Behavior", [
            ("Create backup files on save",
             CheckBox('main_window/make_backup',
                      "Create backup files on save. "
                      "It is recommended  this option on, to keep the backup of the "
                      "edited files in case the new one becomes corrupt or you accidentally "
                      "remove some important parts.")),
            ("Icons theme (requires restart)",
             Combo('main_window/icons_theme',
                   ['Tango', 'Breeze'] if os.name == 'nt' else ['system', 'Tango', 'Breeze'],
                   "Main window icons theme.", needs_restart=True)),
            ("Icons size (requires restart)",
             Combo('main_window/icons_size',
                   ['default', 'small', 'normal', 'large', 'huge', 'enormous'],
                   "Main windows icons size.", needs_restart=True)),
            ("Dark style",
             CheckBox('main_window/dark_style', "Use dark style for application window.", needs_restart=True)),
            ("Automatically check for updates",
             CheckBox('updates/automatic_check',
                      "If this option is checked, PLaSK will automatically check for a new version on startup.")),
            ("Do not use Unicode minus",
             CheckBox("workarounds/no_unicode_minus",
                      "Do not use Unicode minus sign. You should check if if you see some strange character in your plots "
                      "instead of the minus sign.")),
        ]),
        ("Keyboard Shortcuts", keyboard_shortcut_editors),
        ("Help", [
            ("Show only online help",
             CheckBox('help/online',
                      "If this is checked ‘Show Help’ opens online help in an external browser window.")),
            ("Help window font size",
             SpinBox('help/fontsize', 1, 512,
                     "Default font size in the help window.")),
        ]),
        ("Experimental Features", [
            "Here you can turn on experimental features. Please expect that they do not work 100% reliable.\n"
            "Turning them on may cause GUI to crash or the correct files may not load properly.\n\n"
            "You have been warned, use at your own risk!\n",
            ("Preserve XML comments",
             CheckBox('experimental/preserve_comments',
                      "If this option is checked, PLaSK will try to preserve comments in the XML source.",
                      needs_restart=True)),

        ]
         )
    ])),
    ("Graphics", OrderedDict([
        ("General", [
            ("Background color", Color('plots/face_color', "Background color of all plots.")),
            ("Edges color", Color('plots/edge_color', "Color of edges in all plots.")),
            ("Axes color", Color('plots/axes_color', "Color of zero axes in all plots.")),
            ("Grid color", Color('plots/grid_color', "Color of grid in all plots.")),
        ]),
        ("Geometry View", [
            ("Selection frame color", Color('geometry/selected_color',
                                            "Color of a frame around the selected object.")),
            ("Selection frame opacity", FloatSpinBox('geometry/selected_alpha',
                                                     step=0.1, min=0.0, max=1.0,
                                                     help="Opacity of a frame around "
                                                          "the selected object.")),
            ("Selection frame width", FloatSpinBox('geometry/selected_width',
                                                   step=0.1, min=0.1,
                                                   help="Width of a frame around "
                                                        "the selected object.")),
            ("Show local origin", CheckBox('geometry/show_origin',
                                           "Show local origin of the selected object")),
            ("Origin mark color", Color('geometry/origin_color',
                                        "Color of a local origin mark.")),
            ("Origin mark opacity", FloatSpinBox('geometry/origin_alpha',
                                                 step=0.1, min=0.0, max=1.0,
                                                 help="Opacity of a local origin mark.")),
            ("Origin mark size", FloatSpinBox('geometry/origin_size',
                                              step=0.1, min=0.1,
                                              help="Size of a local origin mark.")),
            ("Origin mark width", FloatSpinBox('geometry/origin_width',
                                               step=0.1, min=0.1,
                                               help="Line width of the local origin mark.")),
            ("Show geometry edges", CheckBox('geometry/show_edges',
                                             "Show how the current geometry is interpretted outside its boundaries.")),
            ("Edges opacity", FloatSpinBox('geometry/edges_alpha',
                                           step=0.1, min=0.0, max=1.0,
                                           help="Opacity geometry edges.")),
            ("Info lines color", Color('geometry/extra_color',
                                       "Color of info lines for the selected object.")),
            ("Info lines opacity", FloatSpinBox('geometry/extra_alpha',
                                                step=0.1, min=0.0, max=1.0,
                                                help="Opacity of info lines for the selected object.")),
            ("Info lines width", FloatSpinBox('geometry/extra_width',
                                              step=0.1, min=0.1,
                                              help="Width of info lines for the selected object.")),
            "Lattice Editor",
            ("Existing boundary color", Color('geometry/lattice_line_color',
                                              "Color of boundary lines in lattice editor.")),
            ("Edited boundary color", Color('geometry/lattice_active_color',
                                            "Color of active line in lattice editor.")),
            ("Lattice mark color", Color('geometry/lattice_mark_color',
                                         "Color of current node mark in lattice editor.")),
        ]),
        ("Mesh Preview", [
            ("Mesh color", Color('mesh/mesh_color', "Mesh color in the preview plot.")),
            ("Mesh line width", FloatSpinBox('mesh/line_width', step=0.1, min=0.1,
                                             help="Mesh lines width in the preview plot.")),
        ]),
        ("Boundary Conditions", [
            ("Marker color", Color('boundary_conditions/color',
                                   "Marker color in the boundary conditions preview plot.")),
            ("Selected color", Color('boundary_conditions/selected_color',
                                   "Marker color of the selected boundary condition.")),
        ]),
        ("Material Colors", MaterialColorsConfig)
    ])),
    ("Editor", OrderedDict([
        ("Appearance && Behavior", [
            ("Keep selection after paste", CheckBox('editor/select_after_paste',
                                                    "Keep selection of pasted text.")),
            ("Remove trailing spaces", CheckBox('editor/remove_trailing_spaces',
                                                "Remove trailing spaces from script on file save.")),
            ("Editor font", Font('editor/font', "Font in text editors.")),
            ("Foreground color", Color('editor/foreground_color', "Foreground color in text editor.")),
            ("Background color", Color('editor/background_color', "Background color in text editor.")),
            ("Current line color", Color('editor/current_line_color',
                                         "Background color of the current line.")),
            ("Find result color", Color('editor/match_color',
                                        "Background color of strings matching current search.")),
            ("Replaced result color", Color('editor/replace_color',
                                            "Background color of strings right after replace.")),
            ("Word highlight color", Color('editor/selection_color',
                                           "Highlight color for the current word.")),
            ("Matching bracket color", Color('editor/matching_bracket_color',
                                             "Highlight color for matching brackets "
                                             "in script editor.")),
            ("Unmatched bracket color", Color('editor/not_matching_bracket_color',
                                              "Highlight color for unmatched brackets "
                                              "in script editor.")),
            "Line Numbers Area",
            ("Background color", Color('editor/linenumber_background_color',
                                       "Background color in the line numbers area.")),
            ("Foreground color", Color('editor/linenumber_foreground_color',
                                       "Foreground color in the line numbers area.")),
            "Help Dock",
            ("Help font", Font('editor/help_font', "Font in script on-line help.")),
            ("Foreground color", Color('editor/help_foreground_color', "Foreground color in script on-line help.")),
            ("Background color", Color('editor/help_background_color', "Background color in script on-line help.")),
        ]),
        ("Python Syntax", [
            ("Comment", Syntax('syntax/python_comment', "Python syntax highlighting.")),
            ("String", Syntax('syntax/python_string', "Python syntax highlighting.")),
            ("Special character", Syntax('syntax/python_special', "Python syntax highlighting.")),
            ("Builtin", Syntax('syntax/python_builtin', "Python syntax highlighting.")),
            ("Keyword", Syntax('syntax/python_keyword', "Python syntax highlighting.")),
            ("Number", Syntax('syntax/python_number', "Python syntax highlighting.")),
            ("Decorator", Syntax('syntax/python_decorator', "Python syntax highlighting.")),
            ("Class member", Syntax('syntax/python_member', "Python syntax highlighting.")),
            ("PLaSK function", Syntax('syntax/python_plask', "Python syntax highlighting.")),
            ("PLaSK provider", Syntax('syntax/python_provider', "Python syntax highlighting.")),
            ("PLaSK receiver", Syntax('syntax/python_receiver', "Python syntax highlighting.")),
            ("Log level", Syntax('syntax/python_log', "Python syntax highlighting.")),
            ("Solver", Syntax('syntax/python_solver', "Python syntax highlighting.")),
            ("XPL Definition", Syntax('syntax/python_define', "Python syntax highlighting.")),
            ("PLaSK dictionary", Syntax('syntax/python_loaded', "Python syntax highlighting.")),
            ("Pylab identifier", Syntax('syntax/python_pylab', "Python syntax highlighting.")),
        ]),
        ("XML Syntax", [
            ("XML Tag", Syntax('syntax/xml_tag', "XML syntax highlighting.")),
            ("XML Attribute", Syntax('syntax/xml_attr', "XML syntax highlighting.")),
            ("XML Value", Syntax('syntax/xml_value', "XML syntax highlighting.")),
            ("XML Text", Syntax('syntax/xml_text', "XML syntax highlighting.")),
            ("XML Defined Value", Syntax('syntax/xml_define', "XML syntax highlighting.")),
            ("XML Comment", Syntax('syntax/xml_comment', "XML syntax highlighting.")),
        ]),
        ("Script Completion", [
            ("Do not complete on dot", CheckBox('workarounds/jedi_no_dot',
                                                "Do not show completion pop-up after you type a dot. This still allows "
                                                "to show the pop-up manually by pressing Ctrl+Space.")),
            ("Run in a single thread", CheckBox('workarounds/blocking_jedi',
                                                "Do not run script completion in background. This may cause the GUI "
                                                "stop responding for the couple of seconds when showing completion "
                                                "pop-up, but may be helpful if the program often crashes on "
                                                "completion.")),
            ("Use system completion library", CheckBox('workarounds/system_jedi',
                                                       "Use system jedi library. This may be newer version "
                                                       "than the one bundled with PLaSK. However, it has been "
                                                       "reported to cause GUI crashes.", True)),
            ("Disable completion", CheckBox('workarounds/no_jedi',
                                            "Disable script completion and on-line help.")),
        ]),
    ])),
    ("Launcher", OrderedDict([
        ("Settings",
         [
            ("Default launcher", Combo('launcher/default', get_launchers,
                                       "Default launcher to select in new window.")),
            ("Messages font", Font('launcher_local/font', "Font in local launcher window.")),
        ]),
        ("Colors", [
            ("Background color", Color('launcher_local/background_color', "Background color in launcher window.")),
            ("Foreground color", Color('launcher_local/color_0', "Foreground color in launcher window.")),
            ("Critical error", Color('launcher_local/color_1', "Log colors.")),
            ("Error", Color('launcher_local/color_2', "Log colors.")),
            ("Warning", Color('launcher_local/color_3', "Log colors.")),
            ("Important", Color('launcher_local/color_4', "Log colors.")),
            ("Info", Color('launcher_local/color_5', "Log colors.")),
            ("Result", Color('launcher_local/color_6', "Log colors.")),
            ("Data", Color('launcher_local/color_7', "Log colors.")),
            ("Detail", Color('launcher_local/color_8', "Log colors.")),
            ("Error detail", Color('launcher_local/color_9', "Log colors.")),
            ("Debug", Color('launcher_local/color_10', "Log colors.")),
        ]),
        ("Workarounds",
         [
            ("PLaSK executable", Path('launcher_local/program', "PLaSK executable",
                                      "PLaSK (plask{0});;Any program (*{0})".format('.exe' if sys.platform == 'win32' else ''),
                                      "Full patch to PLaSK executable (leave empty for default)")),
            ("Disable OpenMP", CheckBox('workarounds/disable_omp',
                                        "Disable parallel computations with OpenMP.")),
        ]),
    ])),
    ("Plugins", {"Plugins": PluginsConfig})
])


if os.name == 'posix':
    CONFIG_WIDGETS['Launcher']["Settings"].extend([
        "Console Launcher",
        ("Terminal program", Path('launcher_console/terminal', "Terminal program", "Executable (*)",
                                  "Full patch to terminal program on your system")),
    ])


class SettingsDialog(QDialog):

    class CheckBox(QCheckBox):
        def __init__(self, entry, parent=None, help=None, needs_restart=False):
            super().__init__(parent)
            self.entry = entry
            if help is not None: self.setWhatsThis(help)
            self.needs_restart = needs_restart
            self.load(bool(CONFIG[self.entry]))
        @property
        def changed(self):
            return CONFIG[self.entry] != self.isChecked()
        def load(self, value):
            self.setChecked(value)
        def save(self):
            CONFIG[self.entry] = self.isChecked()

    class Combo(QComboBox):
        def __init__(self, entry, options, parent=None, help=None, needs_restart=False):
            super().__init__(parent)
            self.entry = entry
            if callable(options):
                options = options()
            self.addItems(options)
            if help is not None:
                self.setWhatsThis(help)
            self.needs_restart = needs_restart
            try:
                index = options.index(CONFIG[entry])
            except ValueError:
                index = 0
            self.setCurrentIndex(index)
        @property
        def changed(self):
            return CONFIG[self.entry] != self.currentText()
        def load(self, value):
            index = self.findText(value)
            if index != -1:
                self.setCurrentIndex(index)
        def save(self):
            CONFIG[self.entry] = self.currentText()

    class SpinBox(QSpinBox):
        def __init__(self, entry, parent=None, min=None, max=None, help=None, needs_restart=False):
            super().__init__(parent)
            self.entry = entry
            if min is not None: self.setMinimum(min)
            if max is not None: self.setMaximum(max)
            if help is not None: self.setWhatsThis(help)
            self.needs_restart = needs_restart
            self.load(int(CONFIG[self.entry]))
        @property
        def changed(self):
            return CONFIG[self.entry] != self.value()
        def load(self, value):
            self.setValue(value)
        def save(self):
            CONFIG[self.entry] = self.value()

    class FloatSpinBox(QDoubleSpinBox):
        def __init__(self, entry, parent=None, step=None, min=None, max=None, help=None, needs_restart=False):
            super().__init__(parent)
            self.entry = entry
            if min is not None: self.setMinimum(min)
            if max is not None: self.setMaximum(max)
            if step is not None:
                self.setSingleStep(step)
                self.setDecimals(int(ceil(-log10(step))))
            if help is not None: self.setWhatsThis(help)
            self.needs_restart = needs_restart
            self.load(float(CONFIG[self.entry]))
        @property
        def changed(self):
            return CONFIG[self.entry] != self.value()
        def load(self, value):
            self.setValue(value)
        def save(self):
            CONFIG[self.entry] = self.value()

    class Color(QToolButton):
        def __init__(self, entry, parent=None, help=None, needs_restart=False):
            super().__init__(parent)
            self.entry = entry
            if help is not None:
                self.setWhatsThis(help)
            self.clicked.connect(self.on_press)
            self.setSizePolicy(QSizePolicy.Expanding, self.sizePolicy().verticalPolicy())
            self.needs_restart = needs_restart
            self.load(CONFIG[self.entry])
        def on_press(self):
            dlg = QColorDialog(self.parent())
            if self._color:
                dlg.setCurrentColor(QColor(self._color))
            if dlg.exec_():
                self._color = dlg.currentColor().name()
                self.setStyleSheet(u"background-color: {};".format(self._color))
        @property
        def changed(self):
            return CONFIG[self.entry] != self._color
        def load(self, value):
            self._color = value
            self.setStyleSheet(u"background-color: {};".format(value))
        def save(self):
            CONFIG[self.entry] = self._color

    class Syntax(QWidget):
        def __init__(self, entry, parent=None, help=None, needs_restart=False):
            super().__init__(parent)
            self.entry = entry
            layout = QHBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            self.setLayout(layout)
            self.color_button = QToolButton(self)
            self.color_button.setSizePolicy(QSizePolicy.Expanding, self.color_button.sizePolicy().verticalPolicy())
            self.color_button.clicked.connect(self.on_color_press)
            layout.addWidget(self.color_button)
            self.bold = QCheckBox('bold', self)
            layout.addWidget(self.bold)
            self.italic = QCheckBox('italic', self)
            layout.addWidget(self.italic)
            if help is not None: self.setWhatsThis(help)
            self.needs_restart = needs_restart
            self.load(CONFIG[self.entry])
        def on_color_press(self):
            if QApplication.keyboardModifiers() == Qt.CTRL:
                self._color = None
                self.color_button.setStyleSheet("")
                return
            dlg = QColorDialog(self.parent())
            if self._color:
                dlg.setCurrentColor(QColor(self._color))
            if dlg.exec_():
                self._color = dlg.currentColor().name()
                self.color_button.setStyleSheet(u"background-color: {};".format(self._color))
                self.changed = True
        def load(self, value):
            syntax = parse_highlight(value)
            self._color = syntax.get('color')
            if self._color is not None:
                self.color_button.setStyleSheet(u"background-color: {};".format(self._color))
            self.bold.setChecked(syntax.get('bold', False))
            self.italic.setChecked(syntax.get('italic', False))
            self.changed = False
        def save(self):
            syntax = []
            if self._color is not None: syntax.append('color=' + self._color)
            if self.bold.isChecked(): syntax.append('bold=true')
            if self.italic.isChecked(): syntax.append('italic=true')
            CONFIG[self.entry] = ', '.join(syntax)

    class Font(QPushButton):
        def __init__(self, entry, parent=None, help=None, needs_restart=False):
            super().__init__(parent)
            self.entry = entry
            self.current_font = QFont()
            if help is not None:
                self.setWhatsThis(help)
            self.clicked.connect(self.on_press)
            self.setSizePolicy(QSizePolicy.Expanding, self.sizePolicy().verticalPolicy())
            self.needs_restart = needs_restart
            self.load(parse_font(self.entry))
        def on_press(self):
            dlg = QFontDialog(self.parent())
            dlg.setCurrentFont(self.current_font)
            if dlg.exec_():
                self.current_font = dlg.selectedFont()
                self.setText("{} {}".format(self.current_font.family(), self.current_font.pointSize()))
                self.setFont(self.current_font)
        @property
        def changed(self):
            return CONFIG[self.entry] != self.current_font.toString().split(',')
        def load(self, value):
            self.current_font.fromString(value)
            family = self.current_font.family()
            size = self.current_font.pointSize()
            self.setText("{} {}".format(family, size))
            self.setFont(self.current_font)
        def save(self):
            CONFIG[self.entry] = self.current_font.toString().split(',')

    class Path(QWidget):
        def __init__(self, entry, title,  mask, parent=None, help=None, needs_restart=False):
            super().__init__(parent)
            layout = QHBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            self.edit = QLineEdit(self)
            self.select = QToolButton(self)
            self.select.setIcon(QIcon.fromTheme('document-open'))
            self.select.pressed.connect(self.pushed)
            layout.addWidget(self.edit)
            layout.addWidget(self.select)
            self.setLayout(layout)
            self.entry = entry
            self.title = title
            self.mask = mask
            if help is not None:
                self.setWhatsThis(help)
            self.needs_restart = needs_restart
            self.load(CONFIG[self.entry])
        def pushed(self):
            dirname = os.path.dirname(self.edit.text())
            if not dirname:
                dirname = os.path.dirname(sys.executable)
            filename = QFileDialog.getOpenFileName(self, "Select {}".format(self.title), dirname, self.mask)
            if type(filename) == tuple: filename = filename[0]
            if filename: self.edit.setText(filename)
        @property
        def changed(self):
            return CONFIG[self.entry] != self.edit.text()
        def load(self, value):
            if value is not None:
                self.edit.setText(str(value))
        def save(self):
            CONFIG[self.entry] = self.edit.text()

    try:
        class KeySequence(QKeySequenceEdit):
            def __init__(self, entry, parent=None):
                value = CONFIG.shortcut(entry)
                if value is not None:
                    super().__init__(value, parent)
                else:
                    super().__init__(parent)
                self.entry = entry
                self.editingFinished.connect(self.key_changed)
            needs_restart = False
            @property
            def changed(self):
                saved_key = CONFIG.shortcut(self.entry)
                current_key = self.keySequence()
                return not ((saved_key is None and current_key.isEmpty()) or (saved_key.toString() == current_key.toString()))
            def load(self, value):
                if value:
                    self.setKeySequence(QKeySequence(value, QKeySequence.PortableText))
                else:
                    self.setKeySequence(QKeySequence())
            def _set_shortcut(self, action):
                current_key = self.keySequence()
                try:
                    action.setShortcut(current_key)
                    if not current_key.isEmpty():
                        key_hint = '  ({})'.format(current_key.toString())
                    else:
                        key_hint = ''
                    try:
                        action.setToolTip(action._tooltip + key_hint)
                    except AttributeError:
                        pass
                except Exception as err:
                    return False
                else:
                    return True
            def save(self):
                if self.changed:
                    CONFIG._qshortcuts[self.entry] = \
                        set(act for act in CONFIG._qshortcuts.get(self.entry, ()) if self._set_shortcut(act))
                CONFIG['keyboard_shortcuts/' + self.entry] = self.keySequence().toString()
            def key_changed(self):
                new_key = self.keySequence()
                if new_key.isEmpty(): return
                used = CONFIG.shortcut_used_by(new_key, self.entry)
                if used is not None:
                    confirm = QMessageBox.question(self, "Shortcut Already Used",
                        "Shortcut is already used by action ‘{}’. Do you want to assign it to ‘{}’?".format(
                            KEYBOARD_SHORTCUTS[used][0], KEYBOARD_SHORTCUTS[self.entry][0]),
                        QMessageBox.Yes | QMessageBox.No,  QMessageBox.No)
                    if confirm == QMessageBox.No:
                        self.setKeySequence(CONFIG.shortcut(self.entry))
    except NameError:
        KeySequence = None

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("GUI Settings")
        vlayout = QVBoxLayout()

        self.categories = QListWidget()
        self.stack = QStackedWidget()

        self.filter = LineEditWithClear()
        self.filter.setPlaceholderText("Filter...")
        self.filter.textChanged.connect(self.build_view)

        clayout = QVBoxLayout()
        clayout.addWidget(self.filter)
        clayout.addWidget(self.categories)
        hlayout = QHBoxLayout()
        hlayout.addLayout(clayout)
        hlayout.addWidget(self.stack)
        vlayout.addLayout(hlayout)

        self.items = {}
        self.pages = []
        pages = []

        for cat, tabs in CONFIG_WIDGETS.items():
            page = QTabWidget()
            page_items = []
            page_tabs = []
            for title, items in tabs.items():
                if isinstance(items, type) and issubclass(items, QWidget):
                    widget = items(self.items)
                    widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                    page_items.append((title, widget, ()))
                    page_tabs.append((title, widget, None))
                else:
                    tab = VerticalScrollArea()
                    tab_layout = QFormLayout()
                    tab_items = []
                    if isinstance(items, Callable):
                        items = items()
                    for item in items:
                        if isinstance(item, basestring):
                            label = QLabel(item)
                            font = label.font()
                            font.setBold(True)
                            label.setFont(font)
                            tab_items.append((label,))
                        elif isinstance(item, type) and issubclass(item, QWidget):
                            widget = item(self)
                            self.items[widget.entry] = widget
                            tab_items.append((widget,))
                        else:
                            try:
                                widget = item[1](self)
                            except TypeError:
                                pass
                            else:
                                self.items[widget.entry] = widget
                                tab_items.append((item[0], widget))
                    page_items.append((title, tab_layout, tab_items))
                    page_tabs.append((title, tab, tab_layout))
            self.pages.append((cat, page, page_items))
            pages.append((page, page_tabs))

        self.build_view()
        for page, page_tabs in pages:
            for title, tab, tab_layout in page_tabs:
                if isinstance(tab, QScrollArea):
                    inner = QWidget(tab)
                    inner.setLayout(tab_layout)
                    tab.setWidget(inner)
                page.addTab(tab, title)
            self.stack.addWidget(page)
        page = QWidget()
        self.stack.addWidget(page)

        self.categories.currentRowChanged.connect(lambda i: self.stack.setCurrentIndex(self._page_index[i]))


        cwidth = self.filter.sizeHint().width()
        self.categories.setFixedWidth(cwidth)
        self.filter.setFixedWidth(cwidth)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Apply |  QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply)

        hlayout = QHBoxLayout()
        hlayout.setContentsMargins(0, 0, 0, 0)

        if yaml is not None:
            presets_menu = QMenu()
            presets = sorted(f[:-4] for f in os.listdir(PRESET_DIR) if f.endswith('.yml'))
            for preset in presets:
                preset_action = QAction(preset, self)
                preset_action.triggered.connect(
                    (lambda preset: lambda: CONFIG.load(preset, self.items))(os.path.join(PRESET_DIR, preset+'.yml')))
                presets_menu.addAction(preset_action)
            presets_button = QPushButton("&Presets")
            presets_button.setMenu(presets_menu)
            hlayout.addWidget(presets_button)

            load_button = QPushButton("&Import...")
            load_button.pressed.connect(self.load)
            hlayout.addWidget(load_button)

            save_button = QPushButton("&Export...")
            save_button.pressed.connect(self.save)
            hlayout.addWidget(save_button)

        hlayout.addWidget(buttons)
        vlayout.addLayout(hlayout)
        self.setLayout(vlayout)

        self.resize(800, 600)

    def build_view(self, filter=None):
        if filter is not None:
            filter = filter.lower()
        with BlockQtSignals(self.categories):
            self.categories.clear()
        self._page_index = []
        for page_index, (cat, page, page_items) in enumerate(self.pages):
            show_page = show_whole_page = not filter or filter in cat.lower()
            for tab_index, (title, tab, tab_items) in enumerate(page_items):
                show_whole_tab = show_whole_page or filter in title.lower()
                if tab_items:
                    for _ in range(tab.count()):
                        item = tab.itemAt(0, QFormLayout.FieldRole)
                        if item:
                            widget = item.layout() or item.widget()
                            widget.setParent(None)
                            tab.removeRow(0)
                    if show_whole_tab:
                        for item in tab_items:
                            tab.addRow(*item)
                    else:
                        show_group = False
                        for item in tab_items:
                            if len(item) == 1 and isinstance(item[0], QLabel):
                                if filter in item[0].text().lower():
                                    show_group = True
                                    tab.addRow(*item)
                                else:
                                    show_group = False
                            elif show_group or (isinstance(item[0], str) and filter in item[0].lower()):
                                    tab.addRow(*item)
                    show_tab = tab.count() > 0
                else:
                    show_tab = True
                    if show_whole_tab:
                        tab.filter_view(None)
                    else:
                        show_tab = tab.filter_view(filter)
                page.setTabVisible(tab_index, show_tab)
                if show_tab: show_page = True

            if show_page:
                self._page_index.append(page_index)
                with BlockQtSignals(self.categories):
                    self.categories.addItem(cat)

        if self._page_index:
            current = self.stack.currentIndex()
            if current in self._page_index:
                with BlockQtSignals(self.categories):
                    self.categories.setCurrentRow(self._page_index.index(current))
            else:
                self.categories.setCurrentRow(0)
        else:
            self.stack.setCurrentIndex(self.stack.count()-1)

    def load(self):
        from .. import CURRENT_DIR
        filename = QFileDialog.getOpenFileName(self, "Import settings", CURRENT_DIR, "YAML file (*.yml)")
        if type(filename) is tuple: filename = filename[0]
        if not filename: return False
        try:
            CONFIG.load(filename, self.items)
        except Exception as err:
            msgbox = QMessageBox()
            msgbox.setWindowTitle("Settings Import Error")
            msgbox.setText("The file '{}' could not be loaded from disk.".format(filename))
            msgbox.setInformativeText(unicode(err))
            msgbox.setStandardButtons(QMessageBox.Ok)
            msgbox.setIcon(QMessageBox.Critical)
            msgbox.exec_()

    def save(self):
        from .. import CURRENT_DIR
        filename = QFileDialog.getSaveFileName(self, "Export settings", CURRENT_DIR, "YAML file (*.yml)")
        if type(filename) is tuple: filename = filename[0]
        if not filename: return False
        if not filename.endswith('.yml'): filename += '.yml'
        try:
            CONFIG.save(filename)
        except Exception as err:
            msgbox = QMessageBox()
            msgbox.setWindowTitle("Settings Export Error")
            msgbox.setText("The file '{}' could not be saved to disk.".format(filename))
            msgbox.setInformativeText(unicode(err))
            msgbox.setStandardButtons(QMessageBox.Ok)
            msgbox.setIcon(QMessageBox.Critical)
            msgbox.exec_()
        else:
            msgbox = QMessageBox()
            msgbox.setWindowTitle("Settings Exported")
            msgbox.setText("Settings exported to file '{}'.".format(filename))
            msgbox.setStandardButtons(QMessageBox.Ok)
            msgbox.setIcon(QMessageBox.Information)
            msgbox.exec_()

    def apply(self):
        need_restart = False
        for item in self.items.values():
            if item.needs_restart and item.changed:
                need_restart = True
            item.save()
        CONFIG.sync()
        EDITOR_FONT.fromString(parse_font('editor/font'))
        # This should be changed before the plots are updates
        if matplotlib is not None:
            matplotlib.rcParams['axes.facecolor'] = CONFIG['plots/face_color']
            matplotlib.rcParams['axes.edgecolor'] = CONFIG['plots/edge_color']
            matplotlib.rcParams['grid.color'] = CONFIG['plots/grid_color']
            try: matplotlib.rcParams['axes.unicode_minus'] = not CONFIG['workarounds/no_unicode_minus']
            except KeyError: pass

        self.parent().config_changed.emit()
        if need_restart:
            QMessageBox.information(None,
                                    "Restart Needed",
                                    "Some of the settings you have changed require restart to take effect. "
                                    "Save your work, close PLaSK and open it again.")

    def accept(self):
        self.apply()
        super().accept()
