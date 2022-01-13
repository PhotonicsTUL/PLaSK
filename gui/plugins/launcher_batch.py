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
# coding: utf8

# plugin: Batch Launcher
# description: Launcher allowing to submit computations to a remote batch system (SLURM or Torque).
# description: Requires Paramiko.


from __future__ import print_function, unicode_literals

import sys
import os
import re
from stat import S_ISDIR
import json

from gui.qt.QtCore import Qt, QSize
from gui.qt.QtGui import *
from gui.qt.QtWidgets import *
from gui.qt import qt_exec
from gui.launch import LAUNCHERS, LAUNCH_CONFIG
from gui.utils.widgets import MultiLineEdit, ComboBox
from gui.utils.qsignals import BlockQtSignals

try:
    import cPickle as pickle
except ImportError:
    import pickle

try:
    import paramiko

except ImportError:
    import webbrowser
    import subprocess
    import platform


    class Launcher:
        name = "Remote Batch Job"

        def widget(self, main_window):
            message = QTextBrowser()
            message.setText("Remote batch job launcher cannot be used because Python module "
                            "Paramiko is missing. Either install it manually from its "
                            "<a href=\"http://www.paramiko.org/\">webpage</a> or press "
                            "Ok to try to launch the installation for you.")
            message.setReadOnly(True)
            message.anchorClicked.connect(self._open_link)
            message.setOpenLinks(False)
            pal = message.palette()
            pal.setColor(QPalette.ColorRole.Base, pal.color(QPalette.ColorRole.Window))
            message.setPalette(pal)
            return message

        def _open_link(self, url):
            webbrowser.open(url.toString())

        def launch(self, main_window, args, defs):
            if os.name == 'nt':
                try:
                    import _winreg as winreg
                except ImportError:
                    import winreg
                if 'conda' in sys.version or 'Continuum' in sys.version:
                    command = 'conda.exe'
                else:
                    command = 'pip.exe'
                try:
                    path = winreg.QueryValue(winreg.HKEY_LOCAL_MACHINE,
                                             r"SOFTWARE\Python\PythonCore\{}.{}\InstallPath"
                                             .format(sys.version_info.major, sys.version_info.minor))
                    path = os.path.join(path, 'Scripts', command)
                    if os.path.exists(path): command = path
                except:
                    pass
                subprocess.Popen([command, 'install', 'paramiko'])
            else:
                dist = platform.dist()[0].lower()
                if dist in ('ubuntu', 'debian', 'mint'):
                    term = 'xterm'
                    cmd = 'apt-get'
                    pkg = 'python3-paramiko' if sys.version_info.major == 3 else 'python-paramiko'
                elif dist in ('redhat', 'centos'):
                    term = 'gnome-terminal'
                    cmd = 'yum'
                    pkg = 'python3{}-paramiko'.format(sys.version_info.minor) if sys.version_info.major == 3 else \
                          'python-paramiko'
                else:
                    term = 'xterm'
                    cmd = 'pip3' if sys.version_info.major == 3 else 'pip'
                    pkg = 'paramiko' if sys.version_info.major == 3 else 'python'
                subprocess.Popen([term, '-e', 'sudo {} install {}'.format(cmd, pkg)])
            QMessageBox.information(None, "{} Launcher".format(self.name),
                                          "Once you have successfully installed Paramiko, please restart PLaSK "
                                          "to use the remote batch launcher.")


else:
    import logging
    logging.raiseExceptions = False

    import paramiko.hostkeys

    import os.path
    from collections import OrderedDict

    try:
        from shlex import quote
    except ImportError:
        from pipes import quote

    from gui.xpldocument import XPLDocument
    from gui.utils.config import CONFIG

    from gui.utils.config import DEFAULTS as CONFIG_DEFAULTS
    CONFIG_DEFAULTS['launcher_batch/email'] = ""

    from socket import timeout as TimeoutException

    from gzip import GzipFile
    try:
        from base64 import encodebytes as base64
    except ImportError:
        from base64 import encodestring as base64
    try:
        from StringIO import StringIO as BytesIO
    except ImportError:
        from io import BytesIO

    def hexlify(data):
        if isinstance(data, str):
            return ':'.join('{:02x}'.format(ord(d)) for d in data)
        else:
            return ':'.join('{:02x}'.format(d) for d in data)

    def _parse_bool(value, default):
        try:
            return bool(eval(value))
        except (SyntaxError, ValueError, TypeError):
            return default

    def _parse_int(value, default):
        try:
            return int(value)
        except ValueError:
            return default


    SYSTEMS = OrderedDict()

    def batchsystem(cls):
        SYSTEMS[cls.SYSTEM] = cls
        return cls


    EMAIL = ''


    class Account:
        """
        Base class for account data.
        """

        SUBMIT_OPTION = None
        SYSTEM = None
        RUN1 = ''
        RUNN = 'mpirun'

        def __init__(self, name, userhost=None, port=None, program='', color=None, keep=None, compress=None, bp='',
                     run1='', runn='', params=None):
            self.name = name
            self.userhost = userhost
            self.port = _parse_int(port, 22)
            self.program = program
            self.color = _parse_bool(color, False)
            self.keep = _parse_bool(keep, False)
            self.compress = _parse_bool(compress, True)
            self.bp = bp
            self.run1 = run1
            self.runn = runn
            self._widget = None
            if params is None:
                self.params = {}
            else:
                self.params = params

        def update(self, source):
            self.userhost = source.userhost
            self.port = source.port
            self.program = source.program
            self.color = source.color
            self.keep = source.keep
            self.compress = source.compress
            self.bp = source.bp
            self.run1 = source.run1
            self.runn = source.runn

        @staticmethod
        def load_old(data):
            #      1         2         3         4        5        6        7         8
            # 'userhost', 'system', 'queues', 'color', 'program', 'bp', 'port', 'compress'
            if len(data) == 7: data.append(22)
            return SYSTEMS[data[2]](data[0], data[1], data[7], data[5], False, False, True, data[6], '', data[3])

        @staticmethod
        def load(name, config):
            kwargs = dict(config)
            system = kwargs.pop('system')
            if 'params' in kwargs:
                try:
                    conf = CONFIG.get('params', '')
                    try:
                        kwargs['params'] = json.loads(conf)
                    except json.JSONDecodeError:
                        kwargs['params'] = pickle.loads(conf.encode('iso-8859-1'))
                except (pickle.PickleError, EOFError):
                    del kwargs['params']
            if 'mpirun' in kwargs:
                kwargs['runn'] = kwargs.pop('mpirun')
            return SYSTEMS[system](name, **kwargs)

        def save(self):
            return dict(userhost=self.userhost, port=self.port, system=self.__class__.SYSTEM, program=self.program,
                        color=int(self.color), keep=int(self.keep), compress=int(self.compress), bp=self.bp,
                        run1=self.run1, runn=self.runn)

        class EditDialog(QDialog):
            def __init__(self, account=None, name=None, parent=None):
                super(Account.EditDialog, self).__init__(parent)

                if account is not None and account.userhost:
                    user, host = account.userhost.split('@')
                else:
                    user = host = None

                self.setWindowTitle("Add" if name is None else "Edit" + " Remote Server")

                layout = QFormLayout()
                self.setLayout(layout)

                self.name_edit = QLineEdit()
                self.name_edit.setToolTip("Friendly name of the account.")
                if name is not None:
                    self.name_edit.setText(name)
                    self.autoname = False
                else:
                    self.autoname = True
                self.name_edit.textEdited.connect(self.name_edited)
                layout.addRow("&Name:", self.name_edit)

                self.host_edit = QLineEdit()
                self.host_edit.setToolTip("Hostname to execute the batch job at.")
                if host is not None:
                    self.host_edit.setText(host)
                self.host_edit.textEdited.connect(self.userhost_edited)
                layout.addRow("&Host:", self.host_edit)

                self.port_input = QSpinBox()
                self.port_input.setMinimum(1)
                self.port_input.setMaximum(65536)
                self.port_input.setToolTip("TCP port to connect to (usually 22).")
                if account is not None:
                    self.port_input.setValue(account.port)
                else:
                    self.port_input.setValue(22)
                layout.addRow("&Port:", self.port_input)

                self.user_edit = QLineEdit()
                self.user_edit.setToolTip("Username at the execution host.")
                if user is not None:
                    self.user_edit.setText(user)
                self.user_edit.textEdited.connect(self.userhost_edited)
                layout.addRow("&User:", self.user_edit)

                self.systems_combo = ComboBox()
                self.systems_combo.setToolTip("Batch job scheduling system at the execution host.\n"
                                              "If you are not sure about the correct value, contact\n"
                                              "the host administrator.")
                systems = list(SYSTEMS.keys())
                self.systems_combo.addItems(systems)
                try:
                    if account is not None:
                        systems_index = systems.index(account.SYSTEM)
                        self.systems_combo.setCurrentIndex(systems_index)
                    else:
                        systems_index = 0
                except ValueError:
                    systems_index = 0
                self.systems_combo.currentIndexChanged.connect(self.system_changed)
                layout.addRow("&Batch system:", self.systems_combo)

                self._system_widgets = []
                self._getters = {}
                self.systems = list(SYSTEMS.values())
                for cls in self.systems:
                    widgets = []
                    for label, widget, name, getter in cls.config_widgets(account, self, self):
                        self._getters[name] = getter
                        layout.addRow(label, widget)
                        widget.setVisible(False)
                        widgets.append(widget)
                        layout.labelForField(widget).setVisible(False)
                    self._system_widgets.append(widgets)
                self._set_rows_visibility(self._system_widgets[systems_index], True, layout)

                self.color_checkbox = QCheckBox()
                self.color_checkbox.setToolTip("Add color codes to log messages.")
                if account is not None:
                    self.color_checkbox.setChecked(account.color)
                layout.addRow("Co&lor Output:", self.color_checkbox)

                self.keep_checkbox = QCheckBox()
                self.keep_checkbox.setToolTip("Keep the executed PLaSK script after execution.")
                if account is not None:
                    self.keep_checkbox.setChecked(account.keep)
                layout.addRow("&Keep script:", self.keep_checkbox)

                self._advanced_widgets = []

                self.program_edit = QLineEdit()
                self.program_edit.setToolTip("Path to PLaSK executable. If left blank 'plask' will be used.")
                self.program_edit.setPlaceholderText("plask")
                if account is not None and account.program:
                    self.program_edit.setText(account.program)
                layout.addRow("&Command:", self.program_edit)
                self._advanced_widgets.append(self.program_edit)

                self.run1_edit = QLineEdit()
                self.run1_edit.setToolTip("Command to start single-node tasks (e.g. 'prun', 'srun'). Can be blank.")
                self.run1_edit.setPlaceholderText(self.systems[systems_index].RUN1)
                if account is not None and account.run1:
                    self.run1_edit.setText(account.run1)
                layout.addRow("&Single runner:", self.run1_edit)
                self._advanced_widgets.append(self.run1_edit)

                self.runn_edit = QLineEdit()
                self.runn_edit.setToolTip("Command to start multi-node tasks (e.g. 'prun', 'srun', 'runn'). "
                                          "Can be blank.")
                self.runn_edit.setPlaceholderText(self.systems[systems_index].RUNN)
                if account is not None and account.runn:
                    self.runn_edit.setText(account.runn)
                layout.addRow("&Multi runner:", self.runn_edit)
                self._advanced_widgets.append(self.runn_edit)

                self.bp_edit = QLineEdit()
                self.bp_edit.setToolTip("Path to directory with batch system utilities. Normally you don't need to\n"
                                        "set this, however you may need to specify it, if the submit command is\n"
                                        "located in a non-standard directory.")
                if account is not None and account.bp:
                    self.bp_edit.setText(account.bp)
                layout.addRow("Batch system pat&h:", self.bp_edit)
                self._advanced_widgets.append(self.bp_edit)

                self.compress_checkbox = QCheckBox()
                self.compress_checkbox.setToolTip(
                    "Compress script on sending to the batch system. This can make the scripts\n"
                    "stored in batch system queues smaller, however, it may be harder to track\n"
                    "possible errors.")
                if account is not None:
                    self.compress_checkbox.setChecked(account.compress)
                else:
                    self.compress_checkbox.setChecked(True)
                layout.addRow("Compr&ess Script:", self.compress_checkbox)
                self._advanced_widgets.append(self.compress_checkbox)

                self._set_rows_visibility(self._advanced_widgets, False)

                abutton = QPushButton("Ad&vanced...")
                abutton.setCheckable(True)
                abutton.toggled.connect(self.show_advanced)

                buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
                buttons.addButton(abutton, QDialogButtonBox.ButtonRole.ActionRole)
                buttons.accepted.connect(self.accept)
                buttons.rejected.connect(self.reject)
                layout.addRow(buttons)

                self.host_edit.setFocus()

            def _set_rows_visibility(self, widgets, state, layout=None):
                if layout is None:
                    layout = self.layout()
                for widget in widgets:
                    widget.setVisible(state)
                    layout.labelForField(widget).setVisible(state)

            def system_changed(self, index):
                for widgets in self._system_widgets:
                    self._set_rows_visibility(widgets, False)
                self._set_rows_visibility(self._system_widgets[index], True)
                self.setFixedHeight(self.sizeHint().height())
                self.run1_edit.setPlaceholderText(self.systems[index].RUN1)
                self.runn_edit.setPlaceholderText(self.systems[index].RUNN)
                self.adjustSize()

            def show_advanced(self, show):
                self._set_rows_visibility(self._advanced_widgets, show)
                self.setFixedHeight(self.sizeHint().height())
                self.adjustSize()

            def name_edited(self):
                self.autoname = False

            def userhost_edited(self):
                if self.autoname:
                    if self.user:
                        self.name_edit.setText("{}@{}".format(self.user, self.host))
                    else:
                        self.name_edit.setText(self.host)

            # def accept(self):
            #     dynamic = ' '.join(str(g()) for g in self._getters.values())
            #     if any('`:' in s for s in (self.name, self.host, self.user, self.bp, self.program, self.runn,
            #                               dynamic)):
            #         QMessageBox.critical(None, "Error", "Entered data contain illegal characters (:,).")
            #     else:
            #         super(Account.EditDialog, self).accept()

            def __getattr__(self, attr):
                if attr in self._getters:
                    return self._getters[attr]()
                else:
                    super(Account.EditDialog, self).__getattr__(attr)

            @property
            def name(self):
                return self.name_edit.text()

            @property
            def userhost(self):
                return "{}@{}".format(self.user, self.host)

            @property
            def host(self):
                return self.host_edit.text()

            @property
            def user(self):
                return self.user_edit.text()

            @property
            def port(self):
                return self.port_input.value()

            @property
            def system(self):
                return self.systems_combo.currentText()

            @property
            def run1(self):
                return self.run1_edit.text()

            @property
            def runn(self):
                return self.runn_edit.text()

            @property
            def color(self):
                return self.color_checkbox.isChecked()

            @property
            def keep(self):
                return self.keep_checkbox.isChecked()

            @property
            def compress(self):
                return self.compress_checkbox.isChecked()

            @property
            def program(self):
                return self.program_edit.text()

            @property
            def bp(self):
                return self.bp_edit.text()

        def get_params(self, filename):
            return {}

        def update_widget(self, params):
            return

        def save_params(self, filename):
            if filename in self.params:
                config = LAUNCH_CONFIG[filename].setdefault('batch', {})
                config[self.name] = self.params[filename]
            CONFIG['launcher_batch/email'] = EMAIL
            CONFIG.sync()

        def batch(self, name, params, path):
            return ''

        def tmpfile(self, fname, array):
            return fname

        @staticmethod
        def _compress(document, fname, stdin):
            gzipped = BytesIO()
            with GzipFile(fileobj=gzipped, filename=fname, mode='wb') as gzip:
                gzip.write(document.get_contents().encode('utf8'))
            stdin.write(base64(gzipped.getvalue()).decode('ascii'))

        def submit(self, ssh, document, args, defs, loglevel, name, params):
            fname = os.path.basename(document.filename) if document.filename is not None else 'unnamed'
            bp = self.bp
            if bp:
                if not bp.endswith('/'): bp += '/'
                bp = quote(bp)
            command = self.program or 'plask'
            stdin, stdout, stderr = ssh.exec_command("/bin/bash -l -c " + quote(self.batch(name, params, bp)))
            try:
                print("#!/bin/bash", file=stdin)
                if params['other']:
                    print(self.SUBMIT_OPTION, params['other'], file=stdin)
                for mod in params['modules'].splitlines():
                    if mod.startswith('-'):
                        print('module del', mod[1:], file=stdin)
                    elif mod.startswith('+'):
                        print('module add', mod[1:], file=stdin)
                    else:
                        print('module add', mod, file=stdin)
                if params['nodes'] == 1:
                    run = self.run1 or self.RUN1
                    if self.keep:
                        fname2 = self.tmpfile(fname, params['array'] is not None)
                    else:
                        fname2 = '-:' + fname
                else:
                    run = self.runn or self.RUNN
                    fname2 = self.tmpfile(fname, params['array'] is not None)
                command_line = "{run}{cmd} -u -{ft} -l{ll}{lc} {defs} {fname} {args}".format(
                    run=(run+' ') if run else '',
                    cmd=command,
                    fname=fname2,
                    defs=' '.join(quote(d) for d in defs), args=' '.join(quote(a) for a in args),
                    ll=loglevel,
                    lc=' -lansi' if self.color else '',
                    ft='x' if isinstance(document, XPLDocument) else 'p')
                if self.keep or params['nodes'] > 1:
                    if self.compress:
                        print("base64 -d <<\\_EOF_ | gunzip >", fname2, file=stdin)
                        self._compress(document, fname, stdin)
                    else:
                        print("cat <<\\_EOF_ >", fname2, file=stdin)
                        stdin.write(document.get_contents())
                    print("_EOF_", file=stdin)
                    print(command_line, file=stdin)
                    if not self.keep:
                        print("rm", fname2, file=stdin)
                else:
                    if self.compress:
                        print("base64 -d <<\\_EOF_ | gunzip |", command_line, file=stdin)
                        self._compress(document, fname, stdin)
                    else:
                        print(command_line, "<<\\_EOF_", file=stdin)
                        stdin.write(document.get_contents())
                    print("_EOF_", file=stdin)
                stdin.flush()
                stdin.channel.shutdown_write()
            except (OSError, IOError):
                errors = stderr.read().decode('utf8').strip()
                return False, errors
            else:
                if stderr.channel.recv_exit_status() != 0:
                    errors = stderr.read().decode('utf8').strip()
                    return False, errors
                else:
                    output = stdout.read().decode('utf8').strip()
                    return True, "Submitted job(s):\n" + output

        def widget(self):
            if self._widget is None:
                self._widget = QWidget()
            return self._widget

        def exit(self, filename):
            self._widget = None
            self.save_params(filename)

        @classmethod
        def config_widgets(cls, self, dialog, parent=None):
            return []

        @classmethod
        def _config_list(cls, account, dialog, attr, name, label, command):
            box = QVBoxLayout()
            box.setContentsMargins(0, 0, 0, 0)
            list_edit = MultiLineEdit(movable=True, compact=False)
            list_edit.setToolTip("List of available {} at the execution host.\n"
                                 "If you are not sure about the correct value, contact\n"
                                 "the host administrator.".format(name))
            list_edit.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
            box.addWidget(list_edit)
            if account is not None:
                if getattr(account, attr, None):
                    list_edit.set_values(getattr(account, attr))
            get_partitions = QPushButton("&Retrieve")
            get_partitions.setToolTip("Retrieve the list of {} automatically. To use this,\n"
                                      "you must first correctly fill-in host, user, and system fields.".format(name))
            get_partitions.pressed.connect(lambda: cls._retrieve_list(dialog, name, list_edit, command))
            box.addWidget(get_partitions)
            widget = QWidget()
            widget.setLayout(box)

            return label, widget, attr, lambda: list_edit.get_values()

        @classmethod
        def _retrieve_list(cls, dialog, what, list_edit, command):
            ssh = Launcher.connect(dialog.host, dialog.user, dialog.port)
            if ssh is None: return

            what = what[0].upper() + what[1:]

            bp = dialog.bp
            if bp:
                if not bp.endswith('/'): bp += '/'
                bp = quote(bp)
            _, stdout, stderr = ssh.exec_command(command.format(bp))
            if stdout.channel.recv_exit_status() == 0:
                list_edit.set_values(
                    sorted(line.strip() for line in stdout.read().decode('utf8').split("\n")[:-1]))
            else:
                errors = stderr.read().decode('utf8').strip()
                QMessageBox.critical(None, "Error Retrieving {}".format(what),
                                           "{} list could not be retrieved.".format(what) +
                                           ("\n\n" + errors) if errors else "")


    @batchsystem
    class Slurm(Account):

        SUBMIT_OPTION = "#SBATCH"
        SYSTEM = 'SLURM'
        RUNN = 'srun'

        _NOTIFICATIONS = {
            'b': 'BEGIN',
            'e': 'END',
            'a': 'FAIL'
        }

        _value_suffix_re = re.compile("([\d.]*)(\D*)")

        def __init__(self, name, userhost=None, port=22, program='', color=False, keep=None, compress=True, bp='',
                     run1='', runn='', partitions=None, qos=None, params=None):
            super().__init__(name, userhost, port, program, color, keep, compress, bp, run1, runn, params)
            if partitions:
                self.partitions = partitions if isinstance(partitions, list) else partitions.split(',')
            else:
                self.partitions = []
            if qos:
                self.qos = qos if isinstance(qos, list) else qos.split(',')
            else:
                self.qos = []
            self.partition_combo = self.qos_combo = None

        def update(self, source):
            super().update(source)
            self.partitions = source.partitions
            self.qos = source.qos
            if self._widget is not None:
                partition = self.partition_combo.currentText()
                self.partition_combo.clear()
                self.partition_combo.addItems(self.partitions)
                self.partition_label.setVisible(bool(self.partitions))
                self.partition_combo.setVisible(bool(self.partitions))
                try:
                    self.partition_combo.setCurrentIndex(self.partitions.index(partition))
                except ValueError:
                    pass
                qos = self.qos_combo.currentText()
                self.qos_combo.clear()
                self.qos_combo.addItems(self.qos)
                self.qos_label.setVisible(bool(self.qos))
                self.qos_combo.setVisible(bool(self.qos))
                try:
                    self.qos_combo.setCurrentIndex(self.qos.index(qos))
                except ValueError:
                    pass

        def save(self):
            data = super().save()
            data['partitions'] = ','.join(self.partitions)
            data['qos'] = ','.join(self.qos)
            return data

        def update_widget(self, params):
            if self._widget is not None:
                try: self.partition_combo.setCurrentIndex(self.partitions.index(params.get('partition')))
                except (ValueError, AttributeError): pass
                try: self.qos_combo.setCurrentIndex(self.qos.index(params.get('qos')))
                except (ValueError, AttributeError): pass

        def get_params(self):
            params = {}
            if self._widget is not None:
                if self.partitions:
                    params['partition'] = self.partitions[self.partition_combo.currentIndex()]
                if self.qos:
                    params['qos'] = self.qos[self.qos_combo.currentIndex()]
            return params

        def batch(self, name, params, path):
            name = quote(name)
            partition = '' if self.partition_combo is None or not self.partitions else \
                ' ' + quote('-p' + self.partitions[self.partition_combo.currentIndex()])
            qos = '' if self.qos_combo is None or not self.qos else \
                ' ' + quote('--qos=' + self.qos[self.qos_combo.currentIndex()])
            wall = (' ' + quote('--time=' + params['wall'])) if params['wall'] else ''
            cpus = ' -c {}'.format(params['cpus']) if params['cpus'] else ''
            nodes = ' -n {}'.format(params['nodes']) if params['nodes'] else ''
            dir = quote(params['workdir'])

            if params['mem']:
                m = Slurm._value_suffix_re.match(params['mem'])
                if params['nodes'] > 1 and m is not None:
                    val, suf = m.groups()
                    val = float(val)
                    try:
                        rank = ['K', 'M', 'G', 'T'].index(suf[0].upper())
                    except ValueError:
                        pass
                    else:
                        val *= 1024**rank
                        suf = 'K'
                    c = params['cpus'] or 1
                    mem = ' ' + quote('--mem-per-cpu={:.0f}{}'.format(val/c, suf))
                else:
                    mem = ' ' + quote('--mem=' + params['mem'])
            else:
                mem = ''

            if params['array']:
                output = "{0} -a {1[0]}-{1[1]}".format(quote(name+"-%A_%a.out"), params['array'])
            else:
                output = quote("{}-%j.out".format(name))

            if params['notify'] and EMAIL:
                email = ' ' + quote('--mail-user=' + EMAIL) + ' --mail-type=' + \
                    ','.join(self._NOTIFICATIONS[n] for n in params['notify'])
            else:
                email = ''

            return "{path}sbatch -J {name}{partition}{qos}{wall}{mem}{cpus}{nodes}{email} -o {output} -D {dir}" \
                .format(**locals())

        def tmpfile(self, fname, array):
            if array:
                return "{}-${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}{}".format(*os.path.splitext(fname))
            else:
                return "{}-${{SLURM_JOB_ID}}{}".format(*os.path.splitext(fname))

        def widget(self):
            if self._widget is None:
                self._widget = QWidget()
                layout = QVBoxLayout()
                # layout = QGridLayout()
                # layout.setColumnStretch(0, 1)
                # layout.setColumnStretch(1, 1000)
                layout.setContentsMargins(0, 0, 0, 0)
                self.partition_label = QLabel("&Partition:", self._widget)
                layout.addWidget(self.partition_label)#, 0, 0)
                self.partition_combo = ComboBox(self._widget)
                self.partition_combo.setToolTip("Select the partition to send your job to.")
                self.partition_combo.addItems(self.partitions)
                self.partition_label.setBuddy(self.partition_combo)
                layout.addWidget(self.partition_combo)#, 0, 1)
                if not self.partitions:
                    self.partition_label.setVisible(False)
                    self.partition_combo.setVisible(False)
                self.qos_label = QLabel("&QOS:", self._widget)
                layout.addWidget(self.qos_label)#, 1, 0)
                self.qos_combo = ComboBox(self._widget)
                self.qos_combo.setToolTip("Select the QOS to send your job to.")
                self.qos_combo.addItems(self.qos)
                self.qos_label.setBuddy(self.qos_combo)
                layout.addWidget(self.qos_combo)#, 1, 1)
                if not self.qos:
                    self.qos_label.setVisible(False)
                    self.qos_combo.setVisible(False)
                self._widget.setLayout(layout)
            return self._widget

        @classmethod
        def config_widgets(cls, self, dialog, parent=None):
            return [cls._config_list(self, dialog, 'partitions', 'partitions', '&Partitions:', '{}sinfo -h -o "%R"'),
                    cls._config_list(self, dialog, 'qos', 'QOS', '&QOS:', '{}sacctmgr -Pn list qos format=name')]


    @batchsystem
    class Torque(Account):

        SUBMIT_OPTION = "#PBS"
        SYSTEM = 'PBS/Torque'
        RUNN = 'pbsdsh'

        def __init__(self, name, userhost=None, port=22, program='', color=False, keep=None, compress=True, bp='',
                     run1='', runn='', queues=None, params=None):
            super().__init__(name, userhost, port, program, color, keep, compress, bp, run1, runn, params)
            if queues:
                self.queues = queues if isinstance(queues, list) else queues.split(',')
            else:
                self.queues = []
            self.queue_combo = None

        def update(self, source):
            super().update(source)
            self.queues = source.queues
            if self._widget is not None:
                queue = self.queue_combo.currentText()
                self.queue_combo.clear()
                self.queue_combo.addItems(self.queues)
                try:
                    self.queue_combo.setCurrentIndex(self.queues.index(queue))
                except ValueError:
                    pass

        def save(self):
            data = super().save()
            data['queues'] = ','.join(self.queues)
            return data

        def update_widget(self, params):
            if self._widget is not None:
                try: self.queue_combo.setCurrentIndex(self.queues.index(params.get('queue')))
                except ValueError: pass

        def get_params(self):
            params = {}
            if self._widget is not None and self.queues:
                params['queue'] = self.queues[self.queue_combo.currentIndex()]
            return params

        def batch(self, name, params, path):
            name = quote(name)
            queue = '' if self.queue_combo is None or not self.queues else \
                ' -q ' + quote(self.queues[self.queue_combo.currentIndex()])
            res = ['walltime=' + params['wall']] if params['wall'] else []
            if params['mem']: res.append('pmem=' + params['mem'])
            if params['nodes']:
                nodes = 'nodes={}'.format(params['nodes'])
                if params['cpus']:
                    nodes += ':ppn={}'.format(params['cpus'])
                res.append(nodes)
            elif params['cpus']:
                res.append('nodes=1:ppn={}'.format(params['cpus']))
            if res:
                res = ' -l ' + quote(','.join(res))

            if params['notify'] and EMAIL:
                email = ' -M ' + quote(EMAIL) + ' -m ' + params['notify']
            else:
                email = ''

            return "{path}qsub -N {name}{queue}{res}{array}{email} -d {dir}".format(
                name=quote(name),
                queue=queue,
                res=res,
                dir=quote(params['workdir']),
                array='' if params['array'] is None else " -t {}-{}".format(*params['array']),
                path=path)

        def tmpfile(self, fname, array):
            if array:
                return "{}.${{PBS_JOBID}}-${{PBS_ARRAYID}}{}".format(*os.path.splitext(os.fname))
            else:
                return "{}.${{PBS_JOBID}}{}".format(*os.path.splitext(fname))

        def widget(self):
            if self._widget is None:
                self._widget = QWidget()
                layout = QVBoxLayout()
                layout.setContentsMargins(0, 0, 0, 0)
                label = QLabel("Execution &queue:", self._widget)
                layout.addWidget(label)
                self.queue_combo = ComboBox(self._widget)
                self.queue_combo.setToolTip("Select the execution queue to send your job to.")
                self.queue_combo.addItems(self.queues)
                # if self._saved_queue is not None:
                #     try:
                #         qi = queues.index(self._saved_queue)
                #     except (IndexError, ValueError):
                #         pass
                #     else:
                #         self.queue.setCurrentIndex(qi)
                layout.addWidget(self.queue_combo)
                label.setBuddy(self.queue_combo)
                self._widget.setLayout(layout)
            return self._widget

        @classmethod
        def config_widgets(cls, self, dialog, parent=None):
            return [cls._config_list(self, dialog, 'queues', 'queues', '&Queues:',
                                     "{}qstat -Q | cut -d' ' -f1 | tail -n+3")]


    class Launcher:
        name = "Remote Batch Job"

        _passwd_cache = {}

        class ModulesPlainTextEdit(QPlainTextEdit):
            def sizeHint(self):
                size = super().sizeHint()
                size.setHeight(int(2.5 * self.fontMetrics().height()))
                return size

        def __init__(self):
            self.current_account = None
            self.load_accounts()
            global EMAIL
            EMAIL = CONFIG['launcher_batch/email']

        class Widget(QWidget):
            def __init__(self, launcher, parent=None):
                self.launcher = launcher
                super(Launcher.Widget, self).__init__(parent)
            def hideEvent(self, event):
                if self.launcher.accounts and self.launcher.current_account is not None:
                    self.launcher.accounts[self.launcher.current_account].params[self.launcher.filename] = \
                        self.launcher.get_params()
                super(Launcher.Widget, self).hideEvent(event)

        def widget(self, main_window, parent=None):
            widget = Launcher.Widget(self, parent)
            layout = QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            widget.setLayout(layout)

            self.filename = main_window.document.filename

            label = QLabel("E&xecution account:")
            layout.addWidget(label)
            accounts_layout = QHBoxLayout()
            accounts_layout.setContentsMargins(0, 0, 0, 0)
            self.accounts_combo = ComboBox()
            self.accounts_combo.addItems([a.name for a in self.accounts])
            if self.current_account is not None:
                self.accounts_combo.setCurrentIndex(self.current_account)
            else:
                self.current_account = self.accounts_combo.currentIndex()
            self.accounts_combo.currentIndexChanged.connect(self.account_changed)
            self.accounts_combo.setToolTip("Select the remote server and user to send the job to.")
            accounts_layout.addWidget(self.accounts_combo)
            account_add = QToolButton()
            account_add.setIcon(QIcon.fromTheme('list-add'))
            account_add.setToolTip("Add new remote server.")
            account_add.pressed.connect(self.account_add)
            accounts_layout.addWidget(account_add)
            self.account_edit_button = QToolButton()
            self.account_edit_button.setIcon(QIcon.fromTheme('document-edit'))
            self.account_edit_button.setToolTip("Edit the current remote server.")
            self.account_edit_button.pressed.connect(self.account_edit)
            self.account_edit_button.setEnabled(bool(self.accounts))
            accounts_layout.addWidget(self.account_edit_button)
            self.account_remove_button = QToolButton()
            self.account_remove_button.setIcon(QIcon.fromTheme('list-remove'))
            self.account_remove_button.setToolTip("Remove the current remote server.")
            self.account_remove_button.pressed.connect(self.account_remove)
            self.account_remove_button.setEnabled(bool(self.accounts))
            accounts_layout.addWidget(self.account_remove_button)
            layout.addLayout(accounts_layout)
            label.setBuddy(self.accounts_combo)

            label = QLabel("Job &name:")
            layout.addWidget(label)
            self.jobname = QLineEdit()
            self.jobname.setToolTip("Type a job name to use in the batch system.")
            self.jobname.setPlaceholderText(os.path.splitext(os.path.basename(self.filename))[0]
                                            if self.filename is not None else 'unnamed')
            layout.addWidget(self.jobname)
            label.setBuddy(self.jobname)

            self.accounts_layout = QVBoxLayout()
            self.accounts_layout.setContentsMargins(0, 0, 0, 0)
            accounts_widget = QWidget()
            accounts_widget.setLayout(self.accounts_layout)
            self.account_widgets = []
            for account in self.accounts:
                aw = account.widget()
                self.account_widgets.append(aw)
                self.accounts_layout.addWidget(aw)
                aw.setVisible(False)
            if self.accounts:
                self.account_widgets[self.accounts_combo.currentIndex()].setVisible(True)
            layout.addWidget(accounts_widget)

            grid_layout = QGridLayout()
            label = QLabel("Wall &time:")
            grid_layout.addWidget(label, 0, 0, Qt.AlignmentFlag.AlignRight)
            self.wall_time = QLineEdit()
            self.wall_time.setToolTip("Total time your computation will run. Refer to the batch system\n"
                                      "documentation for the allowed format of this field.")
            label.setBuddy(self.wall_time)
            grid_layout.addWidget(self.wall_time, 0, 1)
            label = QLabel("M&em limit:")
            grid_layout.addWidget(label, 1, 0, Qt.AlignmentFlag.AlignRight)
            self.memory = QLineEdit()
            self.memory.setToolTip("Total computer memory on each node your job will require. Refer to\n"
                                   "the batch system documentation for the allowed format of this field.")
            label.setBuddy(self.memory)
            grid_layout.addWidget(self.memory, 1, 1)
            label = QLabel("&Nodes:")
            grid_layout.addWidget(label, 0, 2, Qt.AlignmentFlag.AlignRight)
            self.nodes = QSpinBox()
            self.nodes.setMinimum(1)
            # self.nodes.setSpecialValueText("default")
            self.nodes.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.nodes.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.nodes.setToolTip("Number of independent nodes your job will allocate. By general PLaSK\n"
                                  "does not operate on multiple nodes, so 1 is usually the best choice.\n"
                                  "However, you may have written your Python script in such way that it\n"
                                  "performs calculations on multiple nodes e.g. using MPI (MPI4Py).\n"
                                  "Note: if you set it to anything larger than 1, make sure, the working\n"
                                  "directory is accessible from all the nodes.")
            label.setBuddy(self.nodes)
            grid_layout.addWidget(self.nodes, 0, 3)
            label = QLabel("&CPUs/node:")
            grid_layout.addWidget(label, 1, 2, Qt.AlignmentFlag.AlignRight)
            self.cpus = QSpinBox()
            self.cpus.setMinimum(0)
            self.cpus.setSpecialValueText("default")
            self.cpus.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.cpus.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.cpus.setToolTip("Number of CPUs on each node your job will utilize.")
            label.setBuddy(self.cpus)
            grid_layout.addWidget(self.cpus, 1, 3)
            layout.addLayout(grid_layout)

            array_layout = QHBoxLayout()
            array_layout.setContentsMargins(0, 0, 0, 0)
            self.array = QCheckBox()
            self.array.setText("A&rray")
            self.array.setToolTip("Check this if you want to launch computations multiple times using an array.\n"
                                  "Refer to your batch system documentation for details on job arrays.")
            array_layout.addWidget(self.array)
            self.array_widget = QWidget()
            array_widget_layout = QHBoxLayout()
            array_widget_layout.setContentsMargins(0, 0, 0, 0)
            self.array_from = QSpinBox()
            self.array_from.setMaximum(99999)
            self.array_from.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.array_from.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.array_from.setToolTip("First job array index.")
            label = QLabel("&start:")
            label.setBuddy(self.array_from)
            array_widget_layout.addWidget(label)
            array_widget_layout.addWidget(self.array_from)
            self.array_to = QSpinBox()
            self.array_to.setMaximum(99999)
            self.array_to.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.array_to.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.array_to.setToolTip("Last job array index.")
            label = QLabel(" en&d:")
            label.setBuddy(self.array_to)
            array_widget_layout.addWidget(label)
            array_widget_layout.addWidget(self.array_to)
            self.array.setChecked(False)
            self.array_from.setValue(0)
            self.array_to.setValue(1)
            self.array.stateChanged.connect(lambda: self.array_widget.setEnabled(self.array.isChecked()))
            self.array_widget.setEnabled(self.array.isChecked())
            self.array_widget.setLayout(array_widget_layout)
            array_layout.addWidget(self.array_widget)
            layout.addLayout(array_layout)

            label = QLabel("&Working directory:")
            layout.addWidget(label)
            self.workdir = QLineEdit()
            self.workdir.setToolTip("Type a directory at the execution server in which the job will run.\n"
                                    "If the directory starts with / it is consider as an absolute path,\n"
                                    "otherwise it is relative to your home directory. If the directory\n"
                                    "does not exists, it is automatically created.")
            label.setBuddy(self.workdir)
            dirbutton = QPushButton()
            dirbutton.setIcon(QIcon.fromTheme('folder-open'))
            dirbutton.pressed.connect(self.select_workdir)
            dirlayout = QHBoxLayout()
            dirlayout.addWidget(self.workdir)
            dirlayout.addWidget(dirbutton)
            layout.addLayout(dirlayout)

            notify_layout = QHBoxLayout()
            notify_layout.setContentsMargins(0, 0, 0, 0)
            label = QLabel("Notify:  ")
            notify_layout.addWidget(label)
            self.notify_begin = QCheckBox()
            self.notify_begin.setText("&Begin")
            self.notify_begin.setToolTip("Notify about job the execution beginning.")
            notify_layout.addWidget(self.notify_begin)
            self.notify_end = QCheckBox()
            self.notify_end.setText("Finis&h")
            self.notify_end.setToolTip("Notify about completion of the job execution.")
            notify_layout.addWidget(self.notify_end)
            self.notify_fail = QCheckBox()
            self.notify_fail.setText("Fa&il")
            self.notify_fail.setToolTip("Notify about job failure.")
            notify_layout.addWidget(self.notify_fail)
            label = QLabel("  E-&mail:")
            notify_layout.addWidget(label)
            self.email = QLineEdit()
            label.setBuddy(self.email)
            notify_layout.addWidget(self.email)
            layout.addLayout(notify_layout)

            self.params_button = QToolButton()
            self.params_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            self.params_button.setIconSize(QSize(8, 8))
            self.params_button.setStyleSheet("""
                border: none;
                margin-left: -2px;
                padding-left: 0px;
            """)
            self.params_button.setCheckable(True)
            self.params_button.setChecked(False)
            self.params_button.toggled.connect(
                lambda visible: self.show_optional(self.other_params, visible, self.params_button))
            self.params_button.setText("Other submit &parameters:")
            layout.addWidget(self.params_button)
            self.other_params = QLineEdit()
            self.other_params.setVisible(False)
            self.other_params.setToolTip("Other submit parameters. You can use them to precisely specify\n"
                                         "requested resources etc. Please refer to batch system documentation\n"
                                         "for details.")
            layout.addWidget(self.other_params)

            self.modules_button = QToolButton()
            self.modules_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            self.modules_button.setIconSize(QSize(8, 8))
            self.modules_button.setStyleSheet("""
                border: none;
                margin-left: -2px;
                padding-left: 0px;
            """)
            self.modules_button.setArrowType(Qt.ArrowType.RightArrow)
            self.modules_button.setCheckable(True)
            self.modules_button.setChecked(False)
            self.modules_button.toggled.connect(
                lambda visible: self.show_optional(self.modules, visible, self.modules_button))
            self.modules_button.setText("En&vironmental Modules:")
            layout.addWidget(self.modules_button)
            self.modules = self.ModulesPlainTextEdit()
            self.modules.setMinimumHeight(self.modules.sizeHint().height())
            self.modules.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.modules.setVisible(False)
            self.modules.setToolTip("Many HPC clusters use LMOD modules to set-up computation environment.\n"
                                    "Here you can specify a list of such modules to load before launching PLaSK.\n"
                                    "Just put one module name or module/version (e.g. 'openblas/0.2.19') per line.\n"
                                    "You can also remove default modules by adding '-' before the module name\n"
                                    "(e.g. '-mpich').")
            layout.addWidget(self.modules)

            label = QLabel("&Log level:")
            layout.addWidget(label)
            self.loglevel = ComboBox()
            loglevels = ["Error", "Warning", "Important", "Info", "Result", "Data", "Detail", "Debug"]
            self.loglevel.addItems(loglevels)
            self.loglevel.setToolTip("Logging level of the executed script.")
            if isinstance(main_window.document, XPLDocument):
                try:
                    self.loglevel.setCurrentIndex(loglevels.index(main_window.document.loglevel.title()))
                except ValueError:
                    self.loglevel.setCurrentIndex(5)
            else:
                self.loglevel.setCurrentIndex(5)
            label.setBuddy(self.loglevel)
            layout.addWidget(self.loglevel)

            self._widget = widget

            if self.accounts:
                self.update_params()
                for account in self.accounts:
                    account.update_widget(account.params.get(self.filename, {}))

            return widget

        @property
        def allow_expand(self):
            return self.modules.isVisible()

        def exit(self, window, visible):
            filename = window.document.filename
            for account in self.accounts:
                account.exit(filename)

        def get_params(self):
            params = {'workdir': self.workdir.text(),
                      'wall': self.wall_time.text(),
                      'mem': self.memory.text(),
                      'cpus': self.cpus.value(),
                      'nodes': self.nodes.value(),
                      'array': (self.array_from.value(), self.array_to.value()) if self.array.isChecked() else None,
                      'other': self.other_params.text(),
                      'modules': self.modules.toPlainText(),
                      'notify': ''}
            global EMAIL
            EMAIL = self.email.text()
            if self.notify_begin.isChecked(): params['notify'] += 'b'
            if self.notify_end.isChecked(): params['notify'] += 'e'
            if self.notify_fail.isChecked(): params['notify'] += 'a'
            if self.accounts and self.current_account is not None:
                params.update(self.accounts[self.current_account].get_params())
            return params

        def update_params(self):
            if not self.accounts or self.current_account is None:
                return
            account = self.accounts[self.current_account]
            params = account.params.get(self.filename, {})
            params.update(LAUNCH_CONFIG[self.filename].get('batch', {}).get(account.name, {}))
            with BlockQtSignals(self._widget):
                self.workdir.setText(params.get('workdir', ''))
                self.wall_time.setText(params.get('wall', ''))
                self.memory.setText(params.get('mem', ''))
                self.cpus.setValue(int(params.get('cpus', 1)))
                self.nodes.setValue(int(params.get('nodes', 1)))
                if params.get('array') is None:
                    self.array.setChecked(False)
                else:
                    self.array.setChecked(True)
                    self.array_from.setValue(int(params.get('array', (0,1))[0]))
                    self.array_to.setValue(int(params.get('array', (0,1))[1]))
                self.other_params.setText(params.get('other', ''))
                self.modules.setPlainText(params.get('modules', ''))
                self.show_optional(self.other_params, bool(params.get('other')), self.params_button)
                self.show_optional(self.modules, bool(params.get('modules')), self.modules_button)
                self.email.setText(EMAIL)
                self.notify_begin.setChecked('b' in params.get('notify', ''))
                self.notify_end.setChecked('e' in params.get('notify', ''))
                self.notify_fail.setChecked('a' in params.get('notify', ''))

        def load_accounts(self):
            self.accounts = []
            accounts = CONFIG['launcher_batch/accounts']
            if accounts is not None:
                accounts = accounts.split('\n')
                if not isinstance(accounts, list):
                    accounts = [accounts]
                for account in accounts:
                    account = account.split(':')
                    self.accounts.append(Account.load_old(account))
            else:
                with CONFIG.group('launcher_batch/accounts') as config:
                    for name, account in config.groups:
                        self.accounts.append(Account.load(name, account))

        def save_accounts(self):
            del CONFIG['launcher_batch/accounts']
            with CONFIG.group('launcher_batch/accounts') as config:
                for account in self.accounts:
                    with config.group(account.name) as group:
                        for k, v in account.save().items():
                            group[k] = v
            CONFIG.sync()

        def account_add(self):
            dialog = Account.EditDialog()
            if qt_exec(dialog) == QDialog.DialogCode.Accepted:
                name = dialog.name
                if name not in self.accounts:
                    account = SYSTEMS[dialog.system](name)
                    account.update(dialog)
                    self.accounts.append(account)
                    self.accounts_combo.addItem(name)
                    widget = account.widget()
                    self.account_widgets.append(widget)
                    self.accounts_layout.addWidget(widget)
                    index = self.accounts_combo.count() - 1
                    self.accounts_combo.setCurrentIndex(index)
                    self.account_edit_button.setEnabled(True)
                    self.account_remove_button.setEnabled(True)
                    self.account_changed(index)
                else:
                    QMessageBox.critical(None, "Add Error",
                                               "Execution account '{}' already in the list.".format(name))
                self.save_accounts()

        def account_edit(self):
            old = self.accounts_combo.currentText()
            idx = self.accounts_combo.currentIndex()
            dialog = Account.EditDialog(self.accounts[idx], old)
            if qt_exec(dialog) == QDialog.DialogCode.Accepted:
                new = dialog.name
                if new != old and new in (a.name for a in self.accounts):
                    QMessageBox.critical(None, "Edit Error",
                                               "Execution account '{}' already in the list.".format(new))
                else:
                    if dialog.system != self.accounts[idx].SYSTEM:
                        self.accounts[idx] = SYSTEMS[dialog.system](new)
                    elif new != old:
                        self.accounts[idx].name = new
                    self.accounts[idx].update(dialog)
                    widget = self.accounts[idx].widget()
                    if self.account_widgets[idx] != widget:
                        self.accounts_layout.removeWidget(self.account_widgets[idx])
                        self.account_widgets[idx].setParent(None)  # delete the widget
                        self.account_widgets[idx] = widget
                        self.accounts_layout.insertWidget(idx, widget)
                    self.accounts_combo.setItemText(idx, new)
                    self.account_changed(new)
                    self.save_accounts()

        def account_remove(self):
            confirm = QMessageBox.warning(None, "Remove Account?",
                                          "Do you really want to remove the account '{}'?"
                                          .format(self.accounts[self.current_account].name),
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if confirm == QMessageBox.StandardButton.Yes:
                self.accounts_layout.removeWidget(self.account_widgets[self.current_account])
                self.account_widgets[self.current_account].setParent(None)  # delete the widget
                del self.account_widgets[self.current_account]
                del self.accounts[self.current_account]
                idx = self.current_account
                self.current_account = None
                self.accounts_combo.removeItem(idx)
                self.account_changed(self.accounts_combo.currentIndex())
                self.save_accounts()
                if not self.accounts:
                    self.account_edit_button.setEnabled(False)
                    self.account_remove_button.setEnabled(False)
                self._adjust_window_size()

        def account_changed(self, index):
            if self.accounts and self.current_account is not None:
                self.accounts[self.current_account].params[self.filename] = self.get_params()
            if isinstance(index, int):
                self.current_account = index
            else:
                self.current_account = self.accounts_combo.currentIndex()
            for aw in self.account_widgets:
                aw.setVisible(False)
            if self.account_widgets:
                self.account_widgets[self.current_account].setVisible(True)
            self.update_params()
            self._adjust_window_size()

        def show_optional(self, field, visible, button=None):
            field.setVisible(visible)
            if button is not None:
                button.setChecked(visible)
                button.setArrowType(Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow)
            self._adjust_window_size()

        def _adjust_window_size(self):
            dialog = self._widget.parent()
            if dialog is None: return
            width = self._widget.width()
            self._widget.adjustSize()
            self._widget.setFixedWidth(width)
            dialog.adjust_height()

        class AbortException(Exception):
            pass

        @staticmethod
        def _save_host_keys(host_keys):
            keylist = []
            for host, keys in host_keys.items():
                for keytype, key in keys.items():
                    keylist.append('{} {} {}'.format(host, keytype, key.get_base64()))
            CONFIG['launcher_batch/ssh_host_keys'] = '\n'.join(keylist)
            CONFIG.sync()

        class AskAddPolicy(paramiko.MissingHostKeyPolicy):
            def missing_host_key(self, client, hostname, key):
                add = QMessageBox.warning(None, "Unknown Host Key",
                                                "The host key for {} is not cached "
                                                "in the registry. You have no guarantee that the "
                                                "server is the computer you think it is.\n\n"
                                                "The server's {} key fingerprint is:\n"
                                                "{}\n\n"
                                                "If you trust this host, hit Yes to add the key "
                                                "to the cache and carry on connecting.\n\n"
                                                "If you want to carry on connecting just once, "
                                                "without adding the key to the cache, hit No.\n\n"
                                                "If you do not trust this host, hit Cancel to "
                                                "abandon the connection."
                                                .format(hostname, key.get_name()[4:], str(hexlify(key.get_fingerprint()))),
                                                 QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
                if add == QMessageBox.StandardButton.Cancel:
                    raise Launcher.AbortException('Server {} not found in known_hosts'.format(hostname))
                client.get_host_keys().add(hostname, key.get_name(), key)
                if add == QMessageBox.StandardButton.Yes:
                    Launcher._save_host_keys(client.get_host_keys())

        @classmethod
        def connect(cls, host, user, port):
            ssh = paramiko.SSHClient()
            ssh.load_system_host_keys()
            host_keys = ssh.get_host_keys()
            saved_keys = CONFIG['launcher_batch/ssh_host_keys']
            if saved_keys is not None:
                saved_keys = saved_keys.split('\n')
                for key in saved_keys:
                    try:
                        e = paramiko.hostkeys.HostKeyEntry.from_line(key)
                    except paramiko.SSHException:
                        continue
                    if e is not None:
                        for h in e.hostnames:
                            host_keys.add(h, e.key.get_name(), e.key)
            ssh.set_missing_host_key_policy(cls.AskAddPolicy())

            passwd = cls._passwd_cache.get((host, user), '')

            while True:
                try:
                    ssh.connect(host, username=user, password=passwd, port=port, compress=True, timeout=15)
                except Launcher.AbortException:
                    return
                except paramiko.BadHostKeyException as err:
                    add = QMessageBox.warning(None, "Bad Host Key",
                                                    "WARNING - POTENTIAL SECURITY BREACH!\n\n"
                                                    "The host key for {} does not "
                                                    "match the one cached in the registry. This means "
                                                    "that either the server administrator has changed "
                                                    "the host key, or you have actually connected to "
                                                    "another computer pretending to be the server.\n\n"
                                                    "The new server's {} key fingerprint is:\n"
                                                    "{}\n\n"
                                                    "If you trust this host, hit Yes to add the key to "
                                                    "the cache and carry on connecting.\n\n"
                                                    "If you want to carry on connecting just once, "
                                                    "without adding the key to the cache, hit No.\n\n"
                                                    "If you do not trust this host, hit Cancel to "
                                                    "abandon the connection."
                                                    .format(err.hostname, err.key.get_name()[4:],
                                                            str(hexlify(err.key.get_fingerprint()))),
                                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
                    if add == QMessageBox.StandardButton.Cancel:
                        return
                    ssh.get_host_keys().add(err.hostname, err.key.get_name(), err.key)
                    if add == QMessageBox.StandardButton.Yes:
                        cls._save_host_keys(ssh.get_host_keys())
                except paramiko.AuthenticationException:
                    dialog = QInputDialog()
                    dialog.setLabelText("Password required for {}@{}. Please enter valid password:"
                                        .format(user, host))
                    dialog.setTextEchoMode(QLineEdit.EchoMode.Password)
                    if qt_exec(dialog) == QDialog.DialogCode.Accepted:
                        passwd = cls._passwd_cache[host, user] = dialog.textValue()
                    else:
                        return
                except Exception as err:
                    try:
                        msg = err.message
                    except AttributeError:
                        msg = str(err)
                    answer = QMessageBox.critical(None, "Connection Error",
                                                        "Could not connect to {}.\n\n{}\n\nTry again?"
                                                        .format(host, msg),
                                                        QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No)
                    if answer == QMessageBox.StandardButton.No:
                        return
                else:
                    return ssh

        def launch(self, main_window, args, defs):
            if len(self.accounts) == 0:
                QMessageBox.critical(None, "Remote Batch Job Error", "No remote account configured!")
                return
            account = self.accounts[self.accounts_combo.currentIndex()]
            user, host = account.userhost.split('@')
            port = account.port
            document = main_window.document
            loglevel = ("error_details", "warning", "important", "info", "result", "data", "detail", "debug") \
                       [self.loglevel.currentIndex()]

            ssh = self.connect(host, user, port)
            if ssh is None: return

            name = self.jobname.text()
            if not name:
                name = os.path.splitext(os.path.basename(document.filename))[0] if document.filename is not None \
                    else 'unnamed'

            params = self.get_params()

            workdir = params['workdir']
            if not workdir:
                _, stdout, _ = ssh.exec_command("pwd")
                workdir = stdout.read().decode('utf8').strip()
            elif not workdir.startswith('/'):
                _, stdout, _ = ssh.exec_command("pwd")
                workdir = '/'.join((stdout.read().decode('utf8').strip(), workdir))
            ssh.exec_command("mkdir -p {}".format(quote(workdir)))

            result, message = account.submit(ssh, document, args, defs, loglevel, name, params)

            account.save_params(document.filename)

            if message: message = "\n\n" + message
            if result:
                QMessageBox.information(None, "Job Submitted",
                                              "Job has been submitted to {}.{}".format(host, message))
            else:
                QMessageBox.critical(None, "Error Submitting Job",
                                           "Could not submit job to {}.{}".format(host, message))

        def select_workdir(self):
            if self.current_account is None:
                return

            user, host = self.accounts[self.current_account].userhost.split('@')
            port = self.accounts[self.current_account].port
            ssh = self.connect(host, user, port)
            if ssh is None: return

            workdir = self.workdir.text()
            if not workdir.startswith('/'):
                _, stdout, _ = ssh.exec_command("pwd")
                home = stdout.read().decode('utf8').strip()
                if workdir: workdir = '/'.join((home, workdir))
                else: workdir = home

            sftp = ssh.open_sftp()

            dialog = RemoteDirDialog(sftp, host, workdir)
            if qt_exec(dialog) == QDialog.DialogCode.Accepted:
                self.workdir.setText(dialog.item_path(dialog.tree.currentItem()))


    class RemoteDirDialog(QDialog):

        # class Item(QTreeWidgetItem):
        #     def __init__(self, *args):
        #         super(RemoteDirDialog.Item, self).__init__(*args)
        #         self.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
        #         self.read = False

        def __init__(self, sftp, host='/', path=None, parent=None):
            self.folder_icon = QIcon.fromTheme('folder')
            super().__init__(parent)
            self.setWindowTitle("Select Folder")
            self.sftp = sftp
            if path is None: path = ['']
            layout = QVBoxLayout()
            label = QLabel("Please choose a folder on the remote machine.")
            layout.addWidget(label)
            self.tree = QTreeWidget()
            self.tree.setHeaderHidden(True)
            layout.addWidget(self.tree)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)
            self.setLayout(layout)
            item = QTreeWidgetItem()
            item.setText(0, host)
            item.setIcon(0, QIcon.fromTheme('network-server'))
            item.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
            self.tree.addTopLevelItem(item)
            self.tree.itemExpanded.connect(self.item_expanded)
            self.resize(540, 720)
            item.setExpanded(True)
            for d in path[1:].split('/'):
                found = False
                for i in range(item.childCount()):
                    child = item.child(i)
                    if child.text(0) == d:
                        item = child
                        found = True
                        break
                if not found: break
                item.setExpanded(True)
            self.tree.scrollToItem(item)
            self.tree.setCurrentItem(item)

        @staticmethod
        def item_path(item):
            path = []
            while item:
                path.insert(0, item.text(0))
                item = item.parent()
            return '/' + '/'.join(path[1:])

        def item_expanded(self, item):
            if item.childIndicatorPolicy() == QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator:
                path = self.item_path(item)
                dirs = []
                try:
                    for f in self.sftp.listdir_attr(path):
                        if S_ISDIR(f.st_mode) and not f.filename.startswith('.'): dirs.append(f)
                except (IOError, SystemError, paramiko.SSHException):
                    pass
                else:
                    dirs.sort(key=lambda d: d.filename)
                    for d in dirs:
                        sub = QTreeWidgetItem()
                        sub.setText(0, d.filename)
                        sub.setIcon(0, self.folder_icon)
                        sub.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
                        item.addChild(sub)
                item.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.DontShowIndicatorWhenChildless)


LAUNCHERS.append(Launcher())
