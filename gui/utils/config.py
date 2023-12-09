# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# coding: utf-8

import sys
import os

try:
    import matplotlib
except ImportError:
    matplotlib = None
else:
    import matplotlib.colors

from ..qt.QtCore import *
from ..qt.QtWidgets import *
from ..qt.QtGui import *
from ..qt import QT_API

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


_parsed = {'true': True, 'yes': True, 'false': False, 'no': False}

if sys.platform == 'win32':
    _default_font_family = "Consolas"
elif sys.platform == 'darwin':
    _default_font_family = "Monaco"
else:
    _default_font_family = "Monospace"


DEFAULTS = {
    'boundary_conditions/color': '#000088',
    'boundary_conditions/selected_color': '#ffea00',
    'editor/background_color': '#ffffff',
    'editor/current_line_color': '#dff5ff',
    'editor/font': [_default_font_family, '11', '-1', '5', '50', '0', '0', '0', '0', '0'],
    'editor/foreground_color': '#000000',
    'editor/help_background_color': '#ffffee',
    'editor/help_font': [_default_font_family, '9', '-1', '5', '50', '0', '0', '0', '0', '0'],
    'editor/help_foreground_color': '#000000',
    'editor/linenumber_background_color': '#dddddd',
    'editor/linenumber_foreground_color': '#808080',
    'editor/match_color': '#ddffdd',
    'editor/matching_bracket_color': '#aaffaa',
    'editor/not_matching_bracket_color': '#ffaaaa',
    'editor/replace_color': '#ffddff',
    'editor/select_after_paste': False,
    'editor/selection_color': '#ffff7f',
    'editor/remove_trailing_spaces': False,
    'editor/find_case_sensitive': True,
    'editor/find_whole_words': False,
    'editor/find_regex': False,
    'geometry/extra_alpha': 0.7,
    'geometry/extra_color': '#00aaff',
    'geometry/extra_width': 1,
    'geometry/lattice_active_color': '#fc4f30',
    'geometry/lattice_line_color': '#30a2da',
    'geometry/lattice_mark_color': '#e5ae38',
    'geometry/material_colors': plask.MATERIAL_COLORS if plask is not None else {},
    'geometry/origin_alpha': 0.7,
    'geometry/origin_color': '#ffea00',
    'geometry/origin_size': 10,
    'geometry/origin_width': 2,
    'geometry/selected_alpha': 0.7,
    'geometry/selected_color': '#ff4444',
    'geometry/selected_width': 2,
    'geometry/show_origin': True,
    'geometry/show_edges': True,
    'geometry/edges_alpha': 0.2,
    'geometry/default_align_ordering': '',
    'help/fontsize': 13,
    'help/online': False,
    'launcher/default': 'local',
    'launcher_local/background_color': '#eeeeec',
    'launcher_local/color_0': '#2e3436',
    'launcher_local/color_1': '#ef2929',
    'launcher_local/color_10': '#888a85',
    'launcher_local/color_2': '#cc0000',
    'launcher_local/color_3': '#f57900',
    'launcher_local/color_4': '#c17d11',
    'launcher_local/color_5': '#3465a4',
    'launcher_local/color_6': '#4e9a06',
    'launcher_local/color_7': '#ad7fa8',
    'launcher_local/color_8': '#555753',
    'launcher_local/color_9': '#a40000',
    'launcher_local/font': [_default_font_family, '10', '-1', '5', '50', '0', '0', '0', '0', '0'],
    'main_window/dark_style': False,
    'main_window/icons_size': 'default',
    'main_window/icons_theme': 'Breeze',
    'main_window/make_backup': True,
    'mesh/line_width': 1,
    'mesh/mesh_color': '#00aaff',
    'plots/face_color': matplotlib.colors.to_hex(matplotlib.rcParams['axes.facecolor'])
                       if matplotlib is not None else '#ffffff',
    'plots/edge_color': matplotlib.colors.to_hex(matplotlib.rcParams['axes.edgecolor'])
                       if matplotlib is not None else '#000000',
    'plots/axes_color': matplotlib.colors.to_hex(matplotlib.rcParams['axes.edgecolor'])
                       if matplotlib is not None else '#000000',
    'plots/grid_color': matplotlib.colors.to_hex(matplotlib.rcParams['grid.color'])
                       if matplotlib is not None else '#b0b0b0',
    'plots/show_grid': True,
    'syntax/python_builtin': 'color=maroon',
    'syntax/python_comment': 'color=#5ca45c, italic=true',
    'syntax/python_decorator': 'color=#009f81',
    'syntax/python_define': 'color=#00aaff, italic=true',
    'syntax/python_keyword': 'color=#000000, bold=true',
    'syntax/python_loaded': 'color=#ff8800',
    'syntax/python_log': 'color=#5f5fc0',
    'syntax/python_member': 'color=#444400',
    'syntax/python_number': 'color=#1b3d80',
    'syntax/python_plask': 'color=#aa007f',
    'syntax/python_provider': 'color=#c08000',
    'syntax/python_pylab': 'color=#440088',
    'syntax/python_receiver': 'color=#c08000',
    'syntax/python_solver': 'color=#ff3333',
    'syntax/python_special': 'color=#268bd2',
    'syntax/python_string': 'color=#2659c0',
    'syntax/xml_attr': 'color=#c08000',
    'syntax/xml_comment': 'color=#5ca45c, italic=true',
    'syntax/xml_define': 'color=#000000',
    'syntax/xml_tag': 'color=#a42020, bold=true',
    'syntax/xml_text': 'color=#5c5c5c',
    'syntax/xml_value': 'color=#2659c0',
    'updates/automatic_check': None,
    'workarounds/blocking_jedi': False,
    'workarounds/disable_omp': False,
    'workarounds/jedi_no_dot': False,
    'workarounds/no_jedi': False,
    'workarounds/no_unicode_minus': False,
}

if os.name == 'posix':
    DEFAULTS['launcher_console/terminal'] = '/usr/bin/gnome-terminal'


KEYBOARD_SHORTCUTS = {
    'main_menu': ('Show Main Menu', 'F2'),
    'new_xpl': ('New XPL File', QKeySequence.StandardKey.New),
    'new_python': ('New Python File', None),
    'open_file': ('Open File', QKeySequence.StandardKey.Open),
    'save_file': ('Save File', QKeySequence.StandardKey.Save),
    'saveas_file': ('Save File As...', QKeySequence.StandardKey.SaveAs),
    'reload_file': ('Reload File', None),
    'goto_line': ('Go to Line...', 'Ctrl+L'),
    'show_source': ('Toggle Source View', 'F4'),
    'launch': ('Launch…', 'F5'),
    'examine_material': ('Examine Material Parameters...', 'Ctrl+Shift+M'),
    'about': ('About...', None),
    'help': ('Open Help...', 'F1'),
    'install_license': ('Install License...', None),
    'fullscreen': ('Toggle Full Screen', 'F11'),
    'settings': ('GUI Settings...', Qt.Modifier.CTRL | Qt.Key.Key_Comma),
    'quit': ('Exit', QKeySequence.StandardKey.Quit),
    'undo': ('Undo', Qt.Modifier.CTRL | Qt.Key.Key_Z),
    'redo': ('Redo', Qt.Modifier.CTRL | Qt.Modifier.SHIFT | Qt.Key.Key_Z),
    'entry_add': ('Add New Entry', Qt.Modifier.CTRL | Qt.Key.Key_Plus),
    'entry_remove': ('Remove Entry', Qt.Modifier.SHIFT | Qt.Key.Key_Delete),
    'entry_move_up': ('Move Entry Up', Qt.Modifier.CTRL | Qt.Modifier.SHIFT | Qt.Key.Key_Up),
    'entry_move_down': ('Move Entry Down', Qt.Modifier.CTRL | Qt.Modifier.SHIFT | Qt.Key.Key_Down),
    'materials_add_library': ('Materials: Add Library', None),
    'materials_add_module': ('Materials: Add Module', None),
    'entry_duplicate': ('Geometry: Duplicate', 'Ctrl+D'),
    'geometry_search': ('Geometry: Search', QKeySequence.StandardKey.Find),
    'geometry_reparent': ('Geometry: Insert into', None),
    'geometry_show_props': ('Geometry: Show Properties', None),
    'geometry_tree_expand_current': ('Expand Current Gometry', None),
    'geometry_tree_expand_all': ('Expand Geometry Tree', None),
    'geometry_tree_collapse_all': ('Collapse Geometry Tree', None),
    'mesh_make': ('Meshing: Make Mesh', None),
    'plot_plot': ('Plot: Plot', 'Alt+P'),
    'plot_refresh': ('Plot: Refresh', None),
    'plot_home': ('Plot: Home', None),
    'plot_back': ('Plot: Back', None),
    'plot_forward': ('Plot: Forward', None),
    'plot_zoom_current': ('Plot: Zoom Selected', None),
    'plot_save': ('Plot: Save', None),
    'plot_pan': ('Plot: Pan', None),
    'plot_zoom': ('Plot: Zoom', None),
    'plot_aspect': ('Plot: Aspect', None),
    'plot_plane': ('Plot: Select Plane', None),
    'plot_geometry': ('Plot: Select Geometry', None),
    'editor_find': ('Editor: Find', QKeySequence.StandardKey.Find),
    'editor_find_next': ('Editor: Find Next', QKeySequence.StandardKey.FindNext),
    'editor_find_prev': ('Editor: Find Previous', QKeySequence.StandardKey.FindPrevious),
    'editor_replace': ('Editor: Replace', QKeySequence.StandardKey.Replace),
    'python_comment': ('Python: Comment Lines', None),
    'python_uncomment': ('Python: Uncomment Lines', Qt.Modifier.CTRL | Qt.Key.Key_Question),
    'python_toggle_comment':('Python: Toggle Comment on Lines', Qt.Modifier.CTRL | Qt.Key.Key_Slash),
    'python_help': ('Python: Open Help', Qt.Modifier.CTRL | Qt.Key.Key_F1),
    'python_docstring': ('Python: Show Docstring', Qt.Modifier.SHIFT | Qt.Key.Key_F1),
    'python_hide_docstring': ('Python: Hide Docstring', Qt.Modifier.SHIFT | Qt.Key.Key_Escape),
    'python_completion': ('Python: Show Completions', Qt.Modifier.CTRL | Qt.Key.Key_Space),
    'python_join_lines': ('Python: Join Lines', Qt.Modifier.SHIFT | Qt.Modifier.CTRL | Qt.Key.Key_J),
    'launcher_copy': ('Launcher: Copy', QKeySequence.StandardKey.Copy),
    'launcher_select_all': ('Launcher: Select All', QKeySequence.StandardKey.SelectAll),
    'launcher_clear_selection': ('Launcher: Clear Selection', None),
    'launcher_show_error': ('Launcher: Show Error', Qt.Key.Key_1),
    'launcher_show_warning': ('Launcher: Show Warning', Qt.Key.Key_2),
    'launcher_show_important': ('Launcher: Show Important', Qt.Key.Key_3),
    'launcher_show_info': ('Launcher: Show Info', Qt.Key.Key_4),
    'launcher_show_result': ('Launcher: Show Result', Qt.Key.Key_5),
    'launcher_show_data': ('Launcher: Show Data', Qt.Key.Key_6),
    'launcher_show_detail': ('Launcher: Show Detail', Qt.Key.Key_7),
    'launcher_show_debug': ('Launcher: Show ebug', Qt.Key.Key_8),
    'launcher_halt': ('Launcher: Halt', 'Alt+X'),
    'launcher_cleanup': ('Launcher: Cleanup All', 'Shift+Ctrl+W'),
    'launcher_close': ('Launcher: Close', 'Ctrl+W'),
}


GROUPS = set(e.split('/', 1)[0] for e in DEFAULTS)


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


def set_font(font, entry):
    values = CONFIG[entry]
    if isinstance(values, str):
        values = values.split(',')
    font.setFamily(values[0])
    try:
        pointsize = int(values[1])
    except ValueError:
        pass
    else:
        font.setPointSize(pointsize)


class Config:
    """Configuration wrapper"""

    def __init__(self):
        self.qsettings = QSettings("plask", "gui")
        self._qshortcuts = {}

    def __getitem__(self, key):
        current = self.qsettings.value(key)
        if current is None:
            current = DEFAULTS.get(key)
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

    class _Group:
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
        data = yaml.safe_load(open(filename))
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

    def shortcut(self, entry):
        key_sequence = self.qsettings.value('keyboard_shortcuts/' + entry)
        if key_sequence is None:
            key_sequence = KEYBOARD_SHORTCUTS.get(entry, (None, None))[1]
        if key_sequence is not None and key_sequence != '':
            if isinstance(key_sequence, QKeySequence):
                return key_sequence
            elif isinstance(key_sequence, str):
                return QKeySequence(key_sequence, QKeySequence.SequenceFormat.PortableText)
            else:
                return QKeySequence(key_sequence)
        else:
            return QKeySequence()

    def shortcut_used_by(self, shortcut, ignore=None):
        if shortcut.isEmpty(): return None
        for entry in KEYBOARD_SHORTCUTS:
            if entry != ignore and self.shortcut(entry) == shortcut:
                return entry
        return None

    def set_shortcut(self, action, entry, label=None, default=None):
        if label is not None:
            KEYBOARD_SHORTCUTS[entry] = label, default
        key_sequence = self.shortcut(entry)
        try:
            action._tooltip = action.toolTip()
        except AttributeError:
            pass
        else:
            if not key_sequence.isEmpty():
                action.setShortcut(key_sequence)
                key_hint = '  ({})'.format(key_sequence.toString())
            else:
                key_hint = ''
            action.setToolTip(action._tooltip + key_hint)
        self._qshortcuts.setdefault(entry, set()).add(action)

CONFIG = Config()


def dark_style():
    palette = QPalette()
    base_color = palette.color(QPalette.ColorRole.Base)
    text_color = palette.color(QPalette.ColorRole.Text)
    return base_color.lightness() < text_color.lightness()



class ConfigProxy:

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
