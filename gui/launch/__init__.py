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
import os
import shlex
import json

from ..qt.QtGui import *
from ..qt.QtWidgets import *
from ..qt.QtCore import *
from ..qt import qt_exec
from ..utils.widgets import ComboBox, EditComboBox
from ..utils.config import CONFIG, parse_highlight
from ..utils.qsignals import BlockQtSignals
from ..utils import system
from ..lib.highlighter import Format

from .local import Launcher as LocalLauncher
from .console import Launcher as ConsoleLauncher

LAUNCH_CONFIG = {}

_DEFS_VISIBLE = False

LAUNCHERS = {
    'local': LocalLauncher(),
    'console': ConsoleLauncher()
}


class DefinesSyntaxHighlighter(QSyntaxHighlighter):

    def __init__(self, parent, defs):
        super().__init__(parent)

        self.defs = defs

        self.formats = {}

        for n, f in (
            ('inactive', parse_highlight(CONFIG['syntax/python_comment'])),
            ('key', parse_highlight(CONFIG['syntax/python_define'])),
            ('eq', parse_highlight(CONFIG['syntax/python_special'])),
        ):
            if isinstance(f, str):
                f = (f, )  # only color specified
            if isinstance(f, (tuple, list)) and len(f) > 0:
                f = Format(qFuzzyIsNull(), f[0])
            elif isinstance(f, dict):
                f = Format(n, f.get('color'))
            else:
                assert isinstance(f, Format), "Format expected, {!r} found".format(f)
            f.tcf.setFontFamily(parent.defaultFont().family())
            self.formats[f.name] = f.tcf
        self.formats['key'].setFontWeight(QFont.Weight.Bold)
        self.formats['bad key'] = Format('bad key', self.formats['key'].foreground().color(), bold=True, strikeout=True).tcf
        self.formats['bad key'].setFontFamily(parent.defaultFont().family())

    def highlightBlock(self, text):
        items = text.split('=', 1)
        if len(items) < 2:
            self.setFormat(0, len(text), self.formats['inactive'])
            return
        key, value = items
        if not value:
            self.setFormat(0, len(text), self.formats['inactive'])
            return
        kl = len(key)
        if key in self.defs:
            self.setFormat(0, kl, self.formats['key'])
        else:
            self.setFormat(0, kl, self.formats['bad key'])
        self.setFormat(kl, 1, self.formats['eq'])


class _CombolItemView(QListView):

    def __init__(self, parent=None, source_list=None, offset=0):
        super().__init__(parent)
        self.source_list = source_list
        self.offset = offset

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            index = self.currentIndex()
            if index.isValid() and index.row() >= self.offset:
                self.model().removeRow(index.row())
                if self.source_list is not None:
                    del self.source_list[index.row() - self.offset]
        super().keyReleaseEvent(event)


class LaunchDialog(QDialog):

    def __init__(self, window, launch_config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Launch Computations")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowMaximizeButtonHint)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        combo = ComboBox()
        combo.insertItems(len(LAUNCHERS), [item.name for item in LAUNCHERS.values()])
        combo.currentIndexChanged.connect(self.launcher_changed, Qt.ConnectionType.QueuedConnection)
        self.layout.addWidget(combo)

        if window.document.defines is not None:
            self.defines_button = QToolButton()
            self.defines_button.setCheckable(True)
            self.defines_button.setChecked(_DEFS_VISIBLE)
            self.defines_button.toggled.connect(self.show_defines)
            self.defines_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            self.defines_button.setIconSize(QSize(8, 8))
            self.defines_button.setStyleSheet(
                """
                border: none;
                margin-left: -2px;
                padding-left: 0px;
            """
            )
            self.defines_button.setArrowType(Qt.ArrowType.DownArrow if _DEFS_VISIBLE else Qt.ArrowType.RightArrow)
            self.defines_button.setText("Temporary de&fines:")
            self.layout.addWidget(self.defines_button)

            self.defines = QPlainTextEdit()
            self.defines.setMinimumHeight(3 * self.defines.fontMetrics().height())
            self.defines.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.layout.addWidget(self.defines)
            self.defines.setVisible(_DEFS_VISIBLE)
            self.current_defines = [e.name for e in window.document.defines.model.entries]
            self.edited_defines = '\n'.join(d + '=' for d in self.current_defines)
            self.defines.setPlainText(self.edited_defines)
            self.highlighter = DefinesSyntaxHighlighter(self.defines.document(), [d for d in self.current_defines])
            self.highlighter.rehighlight()
            self.recent_defines = launch_config.setdefault('defines', [])
            self.recent_defines_combo = ComboBox()
            self.recent_defines_combo.setView(_CombolItemView(self.recent_defines_combo, self.recent_defines, 1))
            self.recent_defines_combo.addItems([''] + ['; '.join("{}={}".format(*i) for i in defs.items())
                                                       for defs in self.recent_defines])
            self.layout.addWidget(self.recent_defines_combo)
            self.recent_defines_combo.setVisible(_DEFS_VISIBLE)
            self.recent_defines_combo.currentIndexChanged.connect(self.recent_defines_selected)
            self.defines.textChanged.connect(self.defines_edited)
        else:
            self.defines = None

        self.args_edit = EditComboBox()
        args = launch_config.get('args', [''])
        self.args_edit.setView(_CombolItemView(self.args_edit, args))
        self.args_edit.setEditable(True)
        self.args_edit.addItems(args)
        args_label = QLabel("Command line &arguments:", self)
        args_label.setBuddy(self.args_edit)
        self.layout.addWidget(args_label)
        self.layout.addWidget(self.args_edit)

        self.launch_config = launch_config

        self.launchers = list(LAUNCHERS.keys())
        self.launcher_widgets = [l.widget(window) for l in LAUNCHERS.values()]
        current_launcher = launch_config.get('current_launcher', CONFIG['launcher/default'])
        try:
            self.current_index = self.launchers.index(current_launcher)
        except ValueError:
            try:
                self.current_index = self.launchers.index('local')
            except ValueError:
                self.current_index = 0
        for i, widget in enumerate(self.launcher_widgets):
            widget.setVisible(i == self.current_index)
            self.layout.addWidget(widget)
        combo.setCurrentIndex(self.current_index)
        launch_config['current_launcher'] = self.current_launcher

        self.setFixedWidth(5 * QFontMetrics(QFont()).horizontalAdvance(self.windowTitle()))

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons)

    def showEvent(self, event):
        super().showEvent(event)
        self._needs_adjusting = True

    # HACK for X11 wrongness
    def paintEvent(self, event):
        if self._needs_adjusting:
            self.adjust_height()
            self._needs_adjusting = False
        super().paintEvent(event)

    def adjust_height(self):
        height = self.sizeHint().height()
        try:
            screen = self.screen().availableGeometry()
        except AttributeError:
            screen = QDesktopWidget().availableGeometry(self)
        screen_height = screen.size().height() - (self.frameGeometry().height() - self.geometry().height())
        if height > screen_height:
            height = screen_height
        self.setMinimumHeight(height)
        try:
            allow_expand = LAUNCHERS[self.current_launchers].allow_expand
        except AttributeError:
            allow_expand = False
        self.setMaximumHeight(
            screen_height if (allow_expand or (self.defines is not None and self.defines.isVisible())) else height
        )
        self.adjustSize()

    def recent_defines_selected(self, i):
        with BlockQtSignals(self.defines):
            if i == 0:
                self.defines.setPlainText(self.edited_defines)
            else:
                defines = '\n'.join(d + '=' + self.recent_defines[i - 1].get(d, '') for d in self.current_defines) + \
                    ''.join("\n{}={}".format(k, v) for k, v in self.recent_defines[i-1].items() if k not in self.current_defines)
                self.defines.setPlainText(defines)

    def defines_edited(self):
        self.edited_defines = self.defines.toPlainText()
        with BlockQtSignals(self.recent_defines_combo):
            self.recent_defines_combo.setItemText(0, "(custom)")
            self.recent_defines_combo.setCurrentIndex(0)

    def show_defines(self, visible):
        self.defines_button.setArrowType(Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow)
        global _DEFS_VISIBLE
        _DEFS_VISIBLE = visible
        self.defines.setVisible(visible)
        self.recent_defines_combo.setVisible(visible)
        self.adjust_height()

    def launcher_changed(self, index):
        self.launcher_widgets[self.current_index].setVisible(False)
        self.current_index = index
        self.launch_config['current_launcher'] = self.current_launcher
        self.launcher_widgets[self.current_index].setVisible(True)
        self.adjust_height()

    @property
    def current_launcher(self):
        return self.launchers[self.current_index]

def _get_config_filename(filename):
    dirname, basename = os.path.split(filename)
    return os.path.join(dirname, '.' + basename + '.json')


def load_launch_config(filename):
    with open(_get_config_filename(filename)) as config_file:
        config = json.load(config_file).get('launch', {})
    if 'defines' in config:
        config['defines'] = [
            defs if isinstance(defs, dict) else
            {it[0]: it[1] for it in ([i.strip() for i in ln.split('=', 1)] for ln in defs.splitlines()) if len(it) == 2 and it[1]}
            for defs in config['defines']
        ]
    return config


def save_launch_config(filename):
    try:
        config = LAUNCH_CONFIG[filename]
        filename = _get_config_filename(filename)
        if config:
            if os.name == 'nt':
                # workaround for hidden files
                try:
                    with open(filename, 'r+') as config_file:
                        json.dump({'launch': config}, config_file, indent=1)
                        config_file.truncate()
                except FileNotFoundError:
                    with open(filename, 'w') as config_file:
                        json.dump({'launch': config}, config_file, indent=1)
                system.set_file_attributes(filename, system.FILE_ATTRIBUTE_HIDDEN)
            else:
                with open(filename, 'w') as config_file:
                    json.dump({'launch': config}, config_file, indent=1)
    except:
        from .. import _DEBUG
        if _DEBUG:
            import traceback
            traceback.print_exc()


def launch_plask(window):
    try:
        launch_config = load_launch_config(window.document.filename)
    except:
        launch_config = LAUNCH_CONFIG.setdefault(window.document.filename, {})
    else:
        if not isinstance(launch_config, dict): launch_config = {}
        LAUNCH_CONFIG[window.document.filename] = launch_config
    try:
        dialog = LaunchDialog(window, launch_config)
        result = qt_exec(dialog)
        launcher = LAUNCHERS[dialog.current_launcher]
        launch_args = dialog.args_edit.currentText().strip()
        if result == QDialog.DialogCode.Accepted:
            try:
                launch_config['args'].remove(launch_args)
            except KeyError:
                launch_config['args'] = []
            except ValueError:
                pass
            launch_config['args'].insert(0, launch_args)
            if launch_config['args'] == ['']:
                del launch_config['args']
            launch_defs = []
            save_defs = {}
            if window.document.defines is not None:
                defines = dialog.defines.toPlainText()
                for line in defines.split('\n'):
                    if not line.strip(): continue
                    if '=' not in line or line.startswith('-'):
                        msgbox = QMessageBox()
                        msgbox.setWindowTitle("Wrong Defines")
                        msgbox.setText("Wrong define: '{}'".format(line))
                        msgbox.setStandardButtons(QMessageBox.StandardButton.Ok)
                        msgbox.setIcon(QMessageBox.Icon.Critical)
                        qt_exec(msgbox)
                        return
                    items = line.split('=', 1)
                    name = items[0].strip()
                    value = items[1].strip()
                    if value:
                        launch_defs.append('-D{}={}'.format(name, value))
                        save_defs[name] = value
                try:
                    launch_config['defines'].remove(save_defs)
                except ValueError:
                    pass
                if launch_defs:
                    launch_config['defines'].insert(0, save_defs)
            launcher.launch(window, shlex.split(launch_args), launch_defs)
    finally:
        if launch_config.get('defines') == []:  # None != []
            del launch_config['defines']
        for launch in LAUNCHERS.values():
            try:
                launch.exit(window, launch is launcher)
            except AttributeError:
                pass
        save_launch_config(window.document.filename)
