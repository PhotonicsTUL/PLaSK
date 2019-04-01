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
from numpy import log10, ceil

try:
    import matplotlib
except ImportError:
    matplotlib = None
else:
    import matplotlib.colors

from ..qt.QtCore import *
from ..qt.QtWidgets import *
from ..qt.QtGui import *

try:
    import yaml
except ImportError:
    yaml = None

try:
    import plask
except ImportError:
    plask = None

try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    unicode = str
    basestring = (str, bytes)
else:
    # 'unicode' exists, must be Python 2
    bytes = str


_parsed = {'true': True, 'yes': True, 'false': False, 'no': False}

if sys.platform == 'win32':
    _default_font_family = "Consolas"
    _plask_binary = "plask.exe"
elif sys.platform == 'darwin':
    _default_font_family = "Monaco"
    _plask_binary = "plask"
else:
    _default_font_family = "Monospace"
    _plask_binary = "plask"


DEFAULTS = {
    'main_window/make_backup': True,
    'main_window/use_menu': False,
    'main_window/icons_theme': 'system',
    'main_window/icons_size': 'default',
    'main_window/dark_style': False,
    'editor/select_after_paste': False,
    'help/online': False,
    'help/fontsize': 13,
    'updates/automatic_check': None,
    'editor/font': [_default_font_family, '11', '-1', '5', '50', '0', '0', '0', '0', '0'],
    'editor/help_font': [_default_font_family, '9', '-1', '5', '50', '0', '0', '0', '0', '0'],
    'editor/help_background_color': '#ffffee',
    'editor/help_foreground_color': 'black',
    'editor/background_color': 'white',
    'editor/foreground_color': 'black',
    'editor/linenumber_background_color': '#dddddd',
    'editor/linenumber_foreground_color': '#808080',
    'editor/current_line_color': '#ffffee',
    'editor/selection_color': '#ffffdd',
    'editor/match_color': '#ddffdd',
    'editor/replace_color': '#ffddff',
    'editor/matching_bracket_color': '#aaffaa',
    'editor/not_matching_bracket_color': '#ffaaaa',
    'launcher/default': 'Local Process',
    'launcher_local/font': [_default_font_family, '10', '-1', '5', '50', '0', '0', '0', '0', '0'],
    'launcher_local/background_color': 'white',
    'launcher_local/color_0': 'black',   # default
    'launcher_local/color_1': 'red',     # critical error
    'launcher_local/color_2': 'red',     # error
    'launcher_local/color_3': 'brown',   # warning
    'launcher_local/color_4': 'magenta', # important
    'launcher_local/color_5': 'blue',    # info
    'launcher_local/color_6': 'green',   # result
    'launcher_local/color_7': '#006060', # data
    'launcher_local/color_8': '#55557f', # detail
    'launcher_local/color_9': '#800000', # error detail
    'launcher_local/color_10': 'gray',   # debug
    'syntax/xml_comment': 'color=green, italic=true',
    'syntax/xml_tag': 'color=maroon, bold=true',
    'syntax/xml_attr': 'color=#888800',
    'syntax/xml_value': 'color=darkblue',
    'syntax/xml_text': 'color=black',
    'syntax/python_comment': 'color=#008000, italic=true',
    'syntax/python_string': 'color=#0000ff',
    'syntax/python_keyword': 'color=black, bold=true',
    'syntax/python_number': 'color=darkblue',
    'syntax/python_builtin': 'color=maroon',
    'syntax/python_member': 'color=#444400',
    'syntax/python_plask': 'color=#aa007f',
    'syntax/python_provider': 'color=#888800',
    'syntax/python_receiver': 'color=#888800',
    'syntax/python_log': 'color=blue',
    'syntax/python_solver': 'color=red',
    'syntax/python_loaded': 'color=#ff8800',
    'syntax/python_pylab': 'color=#440088',
    'syntax/python_define': 'color=#1c68b9, italic=true',
    'syntax/python_decorator': 'color=#009f81',
    'plots/face_color': matplotlib.colors.to_hex(matplotlib.rcParams['axes.facecolor'])
                       if matplotlib is not None else '#ffffff',
    'plots/edge_color': matplotlib.colors.to_hex(matplotlib.rcParams['axes.edgecolor'])
                       if matplotlib is not None else '#000000',
    'plots/axes_color': matplotlib.colors.to_hex(matplotlib.rcParams['axes.edgecolor'])
                       if matplotlib is not None else '#000000',
    'plots/grid_color': matplotlib.colors.to_hex(matplotlib.rcParams['grid.color'])
                       if matplotlib is not None else '#b0b0b0',
    'geometry/selected_color': '#ff4444',
    'geometry/selected_alpha': 0.7,
    'geometry/selected_width': 2.0,
    'geometry/show_origin': True,
    'geometry/origin_color': '#ffea00',
    'geometry/origin_alpha': 0.7,
    'geometry/origin_width': 2.0,
    'geometry/origin_size': 10.0,
    'geometry/extra_color': '#00aaff',
    'geometry/extra_alpha': 0.7,
    'geometry/extra_width': 1.0,
    'geometry/lattice_line_color': '#30a2da',
    'geometry/lattice_active_color': '#fc4f30',
    'geometry/lattice_mark_color': '#e5ae38',
    'geometry/material_colors': plask.MATERIAL_COLORS if plask is not None else {},
    'boundary_conditions/color': '#000088',
    'boundary_conditions/selected_color': '#ffea00',
    'mesh/mesh_color': '#00aaff',
    'mesh/line_width': 1.0,
    'workarounds/jedi_no_dot': False,
    'workarounds/no_jedi': False,
    'workarounds/blocking_jedi': False,
    'workarounds/system_jedi': False,
    'workarounds/disable_omp': False,
}

GROUPS = set(e.split('/', 1)[0] for e in DEFAULTS)

def _get_launchers():
    from ..launch import LAUNCHERS
    return [l.name for l in LAUNCHERS]


def CheckBox(entry, help=None, needs_restart=False):
    return lambda parent: ConfigDialog.CheckBox(entry, help=help, parent=parent, needs_restart=needs_restart)

def Combo(entry, options, help=None, needs_restart=False):
    return lambda parent: ConfigDialog.Combo(entry, options, help=help, parent=parent, needs_restart=needs_restart)

def SpinBox(entry, min=None, max=None, help=None, needs_restart=False):
    return lambda parent: ConfigDialog.SpinBox(entry, min=min, max=max, help=help, parent=parent,
                                               needs_restart=needs_restart)

def FloatSpinBox(entry, step=None, min=None, max=None, help=None, needs_restart=False):
    return lambda parent: ConfigDialog.FloatSpinBox(entry, step=step, min=min, max=max, help=help, parent=parent,
                                                    needs_restart=needs_restart)

def Color(entry, help=None, needs_restart=False):
    return lambda parent: ConfigDialog.Color(entry, help=help, parent=parent, needs_restart=needs_restart)

def Syntax(entry, help=None, needs_restart=False):
    return lambda parent: ConfigDialog.Syntax(entry, help=help, parent=parent, needs_restart=needs_restart)

def Font(entry, help=None, needs_restart=False):
    return lambda parent: ConfigDialog.Font(entry, help=help, parent=parent, needs_restart=needs_restart)

def Path(entry, title, mask, help=None, needs_restart=False):
    return lambda parent: ConfigDialog.Path(entry, title, mask, help=help, parent=parent, needs_restart=needs_restart)


class MaterialColorsConfig(QWidget):

    entry = 'geometry/material_colors'
    caption = "Select custom colors for materials shown in geometry preview."
    needs_restart = False

    class TableModel(QAbstractTableModel):
        def __init__(self):
            super(MaterialColorsConfig.TableModel, self).__init__()
            self.colors = list(CONFIG[MaterialColorsConfig.entry].items())
        def flags(self, index):
            flags = super(MaterialColorsConfig.TableModel, self).flags(index) | Qt.ItemIsEnabled
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
                    return QColor(self.colors[index.row()][1])
        def setData(self, index, value, role=Qt.EditRole):
            row = index.row()
            if index.column() == 0:
                self.colors[row] = value, self.colors[row][1]
                return True
            return False

    def __init__(self, parent=None):
        super(MaterialColorsConfig, self).__init__(parent)
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

        self.table = QTableView()
        self.model = self.TableModel()
        self.table.setModel(self.model)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)

        self.table.clicked.connect(self.select_color)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(toolbar)
        layout.addWidget(self.table)
        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

    def add(self):
        row = len(self.model.colors)
        self.model.beginInsertRows(QModelIndex(), row, row)
        self.model.colors.append(('', '#ffffff'))
        self.model.endInsertRows()

    def remove(self):
        row = self.table.selectionModel().currentIndex().row()
        self.model.beginRemoveRows(QModelIndex(), row, row)
        del self.model.colors[row]
        self.model.endRemoveRows()

    def select_color(self, index):
        if index.column() != 1: return
        row = index.row()
        dlg = QColorDialog(self.parent())
        dlg.setCurrentColor(QColor(self.model.colors[row][1]))
        if dlg.exec_():
            color = dlg.currentColor().name()
            self.model.colors[row] = self.model.colors[row][0], color

    def load(self, value):
        self.model.colors = list(value.items())

    def save(self):
        CONFIG[self.entry] = dict(self.model.colors)


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
        ]),
        ("Help", [
            ("Show only online help",
             CheckBox('help/online',
                      "If this is checked ‘Show Help’ opens online help in an external browser window.")),
            ("Help window font size",
             SpinBox('help/fontsize', 1, 512,
                      "Default font size in the help window.")),
        ]),
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
            "Help dock",
            ("Help font", Font('editor/help_font', "Font in script on-line help.")),
            ("Foreground color", Color('editor/help_foreground_color', "Foreground color in script on-line help.")),
            ("Background color", Color('editor/help_background_color', "Background color in script on-line help.")),
        ]),
        ("Python Syntax", [
            ("Comment", Syntax('syntax/python_comment', "Python syntax highlighting.")),
            ("String", Syntax('syntax/python_string', "Python syntax highlighting.")),
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
            ("Default launcher", Combo('launcher/default', _get_launchers,
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
                                      "PLaSK ({});;Any program ({})"
                                      .format(_plask_binary, '*.exe' if sys.platform == 'win32' else '*'),
                                      "Full patch to PLaSK executable (leave empty for default)")),
            ("Disable OpenMP", CheckBox('workarounds/disable_omp',
                                        "Disable parallel computations with OpenMP.")),
        ]),
    ]))
])


if os.name == 'posix':
    DEFAULTS['launcher_console/terminal'] = '/usr/bin/gnome-terminal'
    CONFIG_WIDGETS['Launcher']["Settings"].extend([
        "Console Launcher",
        ("Terminal program", Path('launcher_console/terminal', "Terminal program", "Executable (*)",
                                  "Full patch to terminal program on your system")),
    ])
    GROUPS.add('launcher_console')


def parse_highlight(string):
    """Parse syntax highlighting from config"""
    if isinstance(string, dict):
        return string
    result = {}
    for item in string.split(','):
        item = item.strip()
        if not item: continue
        key, val = item.split('=')
        result[key] = _parsed.get(val, val)
    return result


def parse_font(entry):
    font = CONFIG[entry]
    if isinstance(font, (str, unicode)):
        font = font.split(',')
    return ','.join(font[:-1])+',0'


class Config(object):
    """Configuration wrapper"""

    def __init__(self):
        self.qsettings = QSettings("plask", "gui")

    def __getitem__(self, key):
        current = self.qsettings.value(key)
        if current is None:
            return DEFAULTS.get(key)
        try:
            return _parsed.get(current, current)
        except TypeError:
            return current

    def get(self, key, default=None):
        current = self.qsettings.value(key)
        if current is None:
            return default
        try:
            return _parsed.get(current, current)
        except TypeError:
            return current

    def __iter__(self):
        for key in self.qsettings.childKeys():
            yield key, self.qsettings.value(key)

    class _Group(object):
        def __init__(self, config, group):
            self.config = config
            self.group = group

        def __enter__(self):
            self.config.qsettings.beginGroup(self.group)
            return self.config

        def __exit__(self, exception_type, exception_value, traceback):
            self.config.qsettings.endGroup()

    def group(self, group):
        return self._Group(self, group)

    @property
    def groups(self):
        for key in self.qsettings.childGroups():
            with self._Group(self, key) as group:
                yield key, group

    def __setitem__(self, key, value):
        self.qsettings.setValue(key, value)

    def __delitem__(self, key):
        self.qsettings.remove(key)

    def sync(self):
        """Synchronize settings"""
        self.qsettings.sync()

    def load(self, filename, widgets=None):
        data = yaml.load(open(filename))
        if not isinstance(data, dict):
            raise TypeError("Wrong YAML file contents.")
        for prefix, group in data.items():
            for key, value in group.items():
                entry = prefix + '/' + key
                try:
                    widgets[entry].load(value)
                except KeyError:
                    pass

    def save(self, filename):
        with open(filename, 'w') as out:
            for group, data in self.groups:
                if group not in GROUPS: continue
                out.write(group + ':\n')
                for key, value in data:
                    if value is None: value = 'null'
                    elif isinstance(value, (list, tuple)):
                        try: value = ','.join(value)
                        except TypeError: pass
                    if isinstance(value, str) and '#' in value: value = "'" + value + "'"
                    out.write('  ' + key + ': ' + str(value) + '\n')


CONFIG = Config()


class ConfigProxy(object):

    def __init__(self, prefix):
        self.prefix = prefix + '/'

    def __getitem__(self, key):
        return CONFIG[self.prefix + key]

    def __setitem__(self, key, value):
        CONFIG[self.prefix+key] = value

    def get(self, key, default=None):
        return CONFIG.get(self.prefix+key, default)

    def sync(self):
        CONFIG.sync()


class ConfigDialog(QDialog):

    class CheckBox(QCheckBox):
        def __init__(self, entry, parent=None, help=None, needs_restart=False):
            super(ConfigDialog.CheckBox, self).__init__(parent)
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
            super(ConfigDialog.Combo, self).__init__(parent)
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
            super(ConfigDialog.SpinBox, self).__init__(parent)
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
            super(ConfigDialog.FloatSpinBox, self).__init__(parent)
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
            super(ConfigDialog.Color, self).__init__(parent)
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
            super(ConfigDialog.Syntax, self).__init__(parent)
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
            super(ConfigDialog.Font, self).__init__(parent)
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
            super(ConfigDialog.Path, self).__init__(parent)
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

    def __init__(self, parent):
        super(ConfigDialog, self).__init__(parent)
        self.setWindowTitle("GUI Settings")
        vlayout = QVBoxLayout()

        categories = QListWidget()
        stack = QStackedWidget()
        categories.currentRowChanged.connect(stack.setCurrentIndex)

        hlayout = QHBoxLayout()
        hlayout.addWidget(categories)
        hlayout.addWidget(stack)
        vlayout.addLayout(hlayout)

        self.items = {}

        for cat, tabs in CONFIG_WIDGETS.items():
            page = QTabWidget()
            stack.addWidget(page)
            categories.addItem(cat)
            for title, items in tabs.items():
                tab = QWidget(page)
                if isinstance(items, type) and issubclass(items, QWidget):
                    tab_layout = QVBoxLayout()
                    tab.setLayout(tab_layout)
                    label = QLabel(items.caption)
                    tab_layout.addWidget(label)
                    widget = items(tab)
                    self.items[widget.entry] = widget
                    widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                    tab_layout.addWidget(widget)
                else:
                    tab_layout = QFormLayout()
                    tab.setLayout(tab_layout)
                    for item in items:
                        if isinstance(item, basestring):
                            label = QLabel(item)
                            font = label.font()
                            font.setBold(True)
                            label.setFont(font)
                            tab_layout.addRow(label)
                        elif isinstance(item, type) and issubclass(item, QWidget):
                            widget = item(self)
                            self.items[widget.entry] = widget
                            tab_layout.addRow(widget)
                        else:
                            widget = item[1](self)
                            self.items[widget.entry] = widget
                            tab_layout.addRow(item[0], widget)
                page.addTab(tab, title)

        page = QTabWidget()
        tab = QWidget()
        tab_layout = QVBoxLayout()
        tab.setLayout(tab_layout)
        label = QLabel("Select active plugins. After making any changes here, you must restart PLaSK GUI.")
        tab_layout.addWidget(label)
        # from .widgets import VerticalScrollArea
        # frame = VerticalScrollArea()
        frame = QScrollArea()
        frame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        frame.setBackgroundRole(QPalette.Base)
        frame.setAutoFillBackground(True)
        tab_layout.addWidget(frame)
        page.addTab(tab, "Plugins")
        inframe = QWidget()
        inframe_layout = QGridLayout()
        inframe_layout.setAlignment(Qt.AlignTop)
        inframe_layout.setHorizontalSpacing(8)
        inframe_layout.setVerticalSpacing(16)
        inframe.setLayout(inframe_layout)
        row = 0
        from .. import PLUGINS
        for plugin, name, desc in PLUGINS:
            entry = 'plugins/{}'.format(plugin)
            if CONFIG[entry] is None: CONFIG[entry] = True
            checkbox = ConfigDialog.CheckBox(entry, help=desc, needs_restart=True)
            checkbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            label = QLabel()
            label.setTextFormat(Qt.RichText)
            if desc is not None:
                label.setText('{}<br/><span style="font-size: small">{}</span>'.format(name, desc))
            else:
                label.setText(name)
            label.setBuddy(checkbox)
            inframe_layout.addWidget(checkbox, row, 0)
            inframe_layout.addWidget(label, row, 1)
            self.items[entry] = checkbox
            row += 1
        frame.setWidget(inframe)
        stack.addWidget(page)
        categories.addItem("Plugins")

        categories.setFixedWidth(categories.sizeHintForColumn(0) + 4)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Apply |  QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply)

        hlayout = QHBoxLayout()
        hlayout.setContentsMargins(0, 0, 0, 0)

        if yaml is not None:
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
        from .widgets import EDITOR_FONT
        EDITOR_FONT.fromString(parse_font('editor/font'))
        # This should be changed before the plots are updates
        if matplotlib is not None:
            matplotlib.rcParams['axes.facecolor'] = CONFIG['plots/face_color']
            matplotlib.rcParams['axes.edgecolor'] = CONFIG['plots/edge_color']
            matplotlib.rcParams['grid.color'] = CONFIG['plots/grid_color']
        self.parent().config_changed.emit()
        if need_restart:
            QMessageBox.information(None,
                                    "Restart Needed",
                                    "Some of the settings you have changed require restart to take effect. "
                                    "Save your work, close PLaSK and open it again.")

    def accept(self):
        self.apply()
        super(ConfigDialog, self).accept()

