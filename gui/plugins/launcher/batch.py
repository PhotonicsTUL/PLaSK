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

# coding utf:8
from __future__ import print_function

import sys
import os
from stat import S_ISDIR

from gui.qt.QtGui import *
from gui.qt.QtWidgets import *
from gui.launch import LAUNCHERS
from gui.utils.widgets import MultiLineEdit

try:
    import paramiko

except ImportError:
    import webbrowser
    import subprocess
    import platform

    class Launcher(object):
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
            pal.setColor(QPalette.Base, pal.color(QPalette.Window))
            message.setPalette(pal)
            return message

        def _open_link(self, url):
            webbrowser.open(url.toString())

        def launch(self, main_window, args, defs):
            if os.name == 'nt' and ('conda' in sys.version or 'Continuum' in sys.version):
                subprocess.Popen(['conda', 'install', 'paramiko'])
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
                    return
                subprocess.Popen([term, '-e', 'sudo {} install {}'.format(cmd, pkg)])
                QMessageBox.information(None, "Remote Batch Job Launcher",
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
        SYSTEMS[cls.NAME] = cls
        return cls


    class Account(object):
        """
        Base class for account data.
        """

        SUBMIT_OPTION = None
        NAME = None

        def __init__(self, userhost=None, port=22, program='', color=None, compress=None, bp='', mpirun=''):
            self.userhost = userhost
            self.port = port
            self.program = program
            self.color = _parse_bool(color, False)
            self.compress = _parse_bool(compress, True)
            self.bp = bp
            self.mpirun = mpirun
            self._widget = None

        def update(self, source):
            self.userhost = source.userhost
            self.port = source.port
            self.program = source.program
            self.color = source.color
            self.compress = source.compress
            self.bp = source.bp
            self.mpirun = source.mpirun

        @staticmethod
        def load(data):
            data = data.split(':')
            try:
                port = int(data[2])
            except ValueError:
                #      1          2         3        4         5       6       7        8
                # 'userhost', 'system', 'queues', 'color', 'program', 'bp', 'port', 'compress'
                return data[0], SYSTEMS[data[2]](data[1],
                                                 _parse_int(data[7], 22),
                                                 data[5],
                                                 _parse_bool(data[4], False),
                                                 _parse_bool(data[8], True),
                                                 data[6],
                                                 '',
                                                 data[3])
            else:
                return data[0], SYSTEMS[data[3]](data[1], port, *data[4:])

        def save(self, name):
            return [name, self.userhost, self.port, self.__class__.NAME, self.program,
                    int(self.color), int(self.compress), self.bp, self.mpirun]

        class EditDialog(QDialog):

            def __init__(self, account=None, name=None, parent=None):
                super(Account.EditDialog, self).__init__(parent)

                if account is None:
                    account = Account()

                if account.userhost:
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
                self.port_input.setValue(account.port)
                layout.addRow("&Port:", self.port_input)

                self.user_edit = QLineEdit()
                self.user_edit.setToolTip("Username at the execution host.")
                if user is not None:
                    self.user_edit.setText(user)
                self.user_edit.textEdited.connect(self.userhost_edited)
                layout.addRow("&User:", self.user_edit)

                self.systems_combo = QComboBox()
                self.systems_combo.setToolTip("Batch job scheduling system at the execution host.\n"
                                            "If you are not sure about the correct value, contact\n"
                                            "the host administrator.")
                systems = list(SYSTEMS.keys())
                self.systems_combo.addItems(systems)
                try:
                    systems_index = systems.index(account.NAME)
                    self.systems_combo.setCurrentIndex(systems_index)
                except ValueError:
                    systems_index = 0
                self.systems_combo.currentIndexChanged.connect(self.system_changed)
                layout.addRow("&Batch system:", self.systems_combo)

                self._system_widgets = []
                self._getters = {}
                for cls in SYSTEMS.values():
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
                self.color_checkbox.setChecked(account.color)
                layout.addRow("Co&lor Output:", self.color_checkbox)

                self._advanced_widgets = []

                self.program_edit = QLineEdit()
                self.program_edit.setToolTip("Path to PLaSK executable. If left blank 'plask' will be used.")
                self.program_edit.setPlaceholderText("plask")
                if account.program:
                    self.program_edit.setText(account.program)
                layout.addRow("&Command:", self.program_edit)
                self._advanced_widgets.append(self.program_edit)

                self.mpirun_edit = QLineEdit()
                self.mpirun_edit.setToolTip("Command to start MPI tasks. Usually 'mpirun' (this is the default value).")
                self.mpirun_edit.setPlaceholderText("mpirun")
                if account.mpirun:
                    self.mpirun_edit.setText(account.mpirun)
                layout.addRow("&MPI runner:", self.mpirun_edit)
                self._advanced_widgets.append(self.mpirun_edit)

                self.bp_edit = QLineEdit()
                self.bp_edit.setToolTip("Path to directory with batch system utilities. Normally you don't need to\n"
                                        "set this, however you may need to specify it if the submit command is\n"
                                        "located in a non-standard directory.")
                if account.bp:
                    self.bp_edit.setText(account.bp)
                layout.addRow("Batch system pat&h:", self.bp_edit)
                self._advanced_widgets.append(self.bp_edit)

                self.compress_checkbox = QCheckBox()
                self.compress_checkbox.setToolTip(
                    "Compress script on sending to the batch system. This can make the scripts\n"
                    "stored in batch system queues smaller, however it may be harder to track\n"
                    "possible errors.")
                self.compress_checkbox.setChecked(account.compress)
                layout.addRow("Compr&ess Script:", self.compress_checkbox)
                self._advanced_widgets.append(self.compress_checkbox)

                self._set_rows_visibility(self._advanced_widgets, False)

                abutton = QPushButton("&Advanced...")
                abutton.setCheckable(True)
                abutton.toggled.connect(self.show_advanced)

                buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                buttons.addButton(abutton, QDialogButtonBox.ActionRole)
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

            def accept(self):
                dynamic = ' '.join(str(g()) for g in self._getters.values())
                if any(':' in s for s in (self.name, self.host, self.user, self.bp, self.program, self.mpirun,
                                          dynamic)):
                    QMessageBox.critical(None, "Error", "Entered data contain illegal characters (:,).")
                else:
                    super(Account.EditDialog, self).accept()

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
            def mpirun(self):
                return self.mpirun_edit.text()

            @property
            def color(self):
                return self.color_checkbox.isChecked()

            @property
            def compress(self):
                return self.compress_checkbox.isChecked()

            @property
            def program(self):
                return self.program_edit.text()

            @property
            def bp(self):
                return self.bp_edit.text()

        @classmethod
        def batch(cls, name, workdir, array, path):
            return ''

        def submit(self, ssh, document, args, defs, loglevel, name, workdir, array, others):
            fname = os.path.basename(document.filename) if document.filename is not None else 'unnamed'
            bp = self.bp
            if bp:
                if not bp.endswith('/'): bp += '/'
                bp = quote(bp)
            command = self.program or 'plask'
            stdin, stdout, stderr = ssh.exec_command(self.batch(name, workdir, array, bp))
            try:
                print("#!/bin/sh", file=stdin)
                for oth in others:
                    oth = oth.strip()
                    if oth: print(self.SUBMIT_OPTION, oth, file=stdin)
                command_line = "{cmd} -{ft} -l{ll}{lc} {defs} -:{fname} {args}".format(
                    cmd=command,
                    fname=fname,
                    defs=' '.join(quote(d) for d in defs), args=' '.join(quote(a) for a in args),
                    ll=loglevel,
                    lc=' -lansi' if self.color else '',
                    ft='x' if isinstance(document, XPLDocument) else 'p')
                if self.compress:
                    print("base64 -d <<\\_EOF_ | gunzip | " + command_line, file=stdin)
                    gzipped = BytesIO()
                    with GzipFile(fileobj=gzipped, filename=name, mode='wb') as gzip:
                        gzip.write(document.get_content().encode('utf8'))
                    stdin.write(base64(gzipped.getvalue()).decode('ascii'))
                    print("_EOF_", file=stdin)
                else:
                    print(command_line + " <<\\_EOF_", file=stdin)
                    stdin.write(document.get_content())
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

        @classmethod
        def config_widgets(cls, self, dialog, parent=None):
            return []


    @batchsystem
    class Slurm(Account):

        SUBMIT_OPTION = "#SBATCH"
        NAME = 'SLURM'

        def __init__(self, userhost=None, port=22, program='', color=False, compress=True, bp='', mpirun='',
                     partitions=None, qos=None):
            super(Slurm, self).__init__(userhost, port, program, color, compress, bp, mpirun)
            if partitions:
                self.partitions = partitions if isinstance(partitions, list) else partitions.split(',')
            else:
                self.partitions = []
            if qos:
                self.qos = qos if isinstance(qos, list) else qos.split(',')
            else:
                self.qos = []

        def update(self, source):
            super(Slurm, self).update(source)
            self.partitions = source.partitions
            self.qos = source.qos
            if self.widget is not None:
                self.partition_combo.clear()
                self.partition_combo.addItems(self.partitions)
            if self.widget is not None:
                self.qos_combo.clear()
                self.qos_combo.addItems(self.qos)

        def save(self, name):
            data = super(Slurm, self).save(name)
            data.append(','.join(self.partitions))
            data.append(','.join(self.qos))
            return data

        @classmethod
        def batch(cls, name, workdir, array, path):
            if array:
                return "{path}sbatch -J {name} {array} -o {name}-%A_%a.out -D {dir}".format(
                    name=quote(name),
                    dir=quote(workdir),
                    array='-a {}-{}'.format(*array),
                    path=path)
            else:
                return "{path}sbatch -J {name} -o {name}-%j.out -D {dir}".format(
                    name=quote(name),
                    dir=quote(workdir),
                    path=path)

        def widget(self):
            if self._widget is None:
                self._widget = QWidget()
                layout = QVBoxLayout()
                # layout = QGridLayout()
                # layout.setColumnStretch(0, 1)
                # layout.setColumnStretch(1, 1000)
                layout.setContentsMargins(0, 0, 0, 0)
                label = QLabel("&Partition:", self._widget)
                layout.addWidget(label)#, 0, 0)
                self.partition_combo = QComboBox(self._widget)
                self.partition_combo.setToolTip("Select the partition to send your job to.")
                self.partition_combo.addItems(self.partitions)
                layout.addWidget(self.partition_combo)#, 0, 1)
                label.setBuddy(self.partition_combo)
                label = QLabel("&QOS:", self._widget)
                layout.addWidget(label)#, 1, 0)
                self.qos_combo = QComboBox(self._widget)
                self.qos_combo.setToolTip("Select the QOS to send your job to.")
                self.qos_combo.addItems(self.qos)
                layout.addWidget(self.qos_combo)#, 1, 1)
                label.setBuddy(self.qos_combo)
                self._widget.setLayout(layout)
            return self._widget

        @staticmethod
        def _config_list(account, dialog, attr, name, label):
            box = QVBoxLayout()
            box.setContentsMargins(0, 0, 0, 0)
            list_edit = MultiLineEdit(movable=True, placeholder='[{} name]'.format(attr))
            list_edit.setToolTip("List of available {} at the execution host.\n"
                                 "If you are not sure about the correct value, contact\n"
                                 "the host administrator.".format(name))
            list_edit.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
            box.addWidget(list_edit)
            try:
                if getattr(account, attr):
                    list_edit.set_values(getattr(account, attr))
            except AttributeError:
                pass
            get_partitions = QPushButton("&Retrieve")
            get_partitions.setToolTip("Retrieve the list of {} automatically. To use this,\n"
                                      "you must first correctly fill-in host, user, and system fields.".format(name))
            get_partitions.pressed.connect(lambda: Slurm._retrieve_list(dialog, name, list_edit))
            box.addWidget(get_partitions)
            widget = QWidget()
            widget.setLayout(box)

            return label, widget, attr, lambda: list_edit.get_values()

        @classmethod
        def config_widgets(cls, self, dialog, parent=None):
            return [Slurm._config_list(self, dialog, 'partitions', 'partitions', '&Partition'),
                    Slurm._config_list(self, dialog, 'qos', 'QOS', '&QOS')]

        _retrieve_commands = {'partitions': '{}sinfo -h -o "%R"',
                              'QOS': '{}sacctmgr -Pn list qos format=name'}

        @staticmethod
        def _retrieve_list(dialog, what, list_edit):
            ssh = Launcher.connect(dialog.host, dialog.user, dialog.port)
            if ssh is None: return

            command = Slurm._retrieve_commands[what]
            what = what[1].upper() + what[1:]

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
    class Torque(Account):

        SUBMIT_OPTION = "#PBS"
        NAME = 'PBS/Torque'

        def __init__(self, userhost=None, port=22, program='', color=False, compress=True, bp='', mpirun='',
                     queues=None):
            super(Torque, self).__init__(userhost, port, program, color, compress, bp, mpirun)
            if queues:
                self.queues = queues if isinstance(queues, list) else queues.split(',')
            else:
                self.queues = []

        def update(self, source):
            super(Torque, self).update(source)
            self.queues = source.queues
            if self.widget is not None:
                self.queue_combo.clear()
                self.queue_combo.addItems(self.queues)

        def save(self, name):
            data = super(Torque, self).save(name)
            data.append(','.join(self.queues))
            return data

        @classmethod
        def batch(cls, name, workdir, array, path):
            return "{path}qsub -N {name}{array} -d {dir}".format(
                name=quote(name),
                dir=quote(workdir),
                array='' if array is None else " -t {}-{}".format(*array),
                path=path)

        def widget(self):
            if self._widget is None:
                self._widget = QWidget()
                layout = QVBoxLayout()
                layout.setContentsMargins(0, 0, 0, 0)
                label = QLabel("Execution &queue:", self._widget)
                layout.addWidget(label)
                self.queue_combo = QComboBox(self._widget)
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
            qbox = QVBoxLayout()
            qbox.setContentsMargins(0, 0, 0, 0)
            queues_list = MultiLineEdit(movable=True, placeholder='[queue name]')
            queues_list.setToolTip("List of available queues at the execution host.\n"
                                   "If you are not sure about the correct value, contact\n"
                                   "the host administrator.")
            queues_list.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
            qbox.addWidget(queues_list)
            try:
                if self.queues:
                    queues_list.set_values(self.queues)
            except AttributeError:
                pass
            get_queues = QPushButton("&Retrieve")
            get_queues.setToolTip("Retrieve the list of available queues automatically. To use this,\n"
                                  "you must first correctly fill-in host, user, and system fields.")
            get_queues.pressed.connect(lambda: cls._get_queues(dialog, queues_list))
            qbox.addWidget(get_queues)
            qwidget = QWidget()
            qwidget.setLayout(qbox)

            return [("&Queues:", qwidget, 'queues', lambda: queues_list.get_values())]

        @staticmethod
        def _get_queues(dialog, queues_list):
            ssh = Launcher.connect(dialog.host, dialog.user, dialog.port)
            if ssh is None: return

            bp = dialog.bp
            if bp:
                if not bp.endswith('/'): bp += '/'
                bp = quote(bp)
            _, stdout, stderr = ssh.exec_command("{}qstat -Q".format(bp))
            if stdout.channel.recv_exit_status() == 0:
                queues_list.set_values(
                    sorted(line.split()[0] for line in stdout.read().decode('utf8').split("\n")[2:-1]))
            else:
                errors = stderr.read().decode('utf8').strip()
                QMessageBox.critical(None, "Error Retrieving Queues",
                                           "Queue list could not be retrieved." +
                                           ("\n\n" + errors) if errors else "")


    class Launcher(object):
        name = "Remote Batch Job"

        _passwd_cache = {}

        def __init__(self):
            self._workdirs = {}
            self._saved_account = None
            self._saved_queue = None
            self._saved_array = None
            self._saved_params = None

        def widget(self, main_window, parent):
            widget = QWidget()
            layout = QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            widget.setLayout(layout)

            self.dialog = parent

            self.filename = main_window.document.filename

            label = QLabel("&Execution account:")
            layout.addWidget(label)
            accounts_layout = QHBoxLayout()
            accounts_layout.setContentsMargins(0, 0, 0, 0)
            self._load_accounts()
            self.accounts_combo = QComboBox()
            self.accounts_combo.addItems([s for s in self.accounts])
            if self._saved_account is not None:
                self.accounts_combo.setCurrentIndex(self._saved_account)
            self.accounts_combo.currentIndexChanged.connect(self.account_changed)
            self.accounts_combo.setToolTip("Select the remote server and user to send the job to.")
            accounts_layout.addWidget(self.accounts_combo)
            account_add = QToolButton()
            account_add.setIcon(QIcon.fromTheme('list-add'))
            account_add.setToolTip("Add new remote server.")
            account_add.pressed.connect(self.account_add)
            accounts_layout.addWidget(account_add)
            account_edit = QToolButton()
            account_edit.setIcon(QIcon.fromTheme('document-edit'))
            account_edit.setToolTip("Edit the current remote server.")
            account_edit.pressed.connect(self.account_edit)
            accounts_layout.addWidget(account_edit)
            account_remove = QToolButton()
            account_remove.setIcon(QIcon.fromTheme('list-remove'))
            account_remove.setToolTip("Remove the current remote server.")
            account_remove.pressed.connect(self.account_remove)
            accounts_layout.addWidget(account_remove)
            layout.addLayout(accounts_layout)
            label.setBuddy(self.accounts_combo)

            self.accounts_layout = QVBoxLayout()
            self.accounts_layout.setContentsMargins(0, 0, 0, 0)
            accounts_widget = QWidget()
            accounts_widget.setLayout(self.accounts_layout)
            self.account_widgets = []
            for account in self.accounts.values():
                aw = account.widget()
                self.account_widgets.append(aw)
                self.accounts_layout.addWidget(aw)
                aw.setVisible(False)
            self.account_widgets[self.accounts_combo.currentIndex()].setVisible(True)
            layout.addWidget(accounts_widget)

            label = QLabel("Job &name:")
            layout.addWidget(label)
            self.jobname = QLineEdit()
            self.jobname.setToolTip("Type a job name to use in the batch system.")
            self.jobname.setPlaceholderText(os.path.basename(self.filename)
                                            if self.filename is not None else 'unnamed')
            layout.addWidget(self.jobname)
            label.setBuddy(self.jobname)

            self._load_workdirs()
            label = QLabel("&Working directory:")
            layout.addWidget(label)
            self.workdir = QLineEdit()
            self.workdir.setToolTip("Type a directory at the execution server in which the job will run.\n"
                                    "If the directory starts with / it is consider as an absolute path,\n"
                                    "otherwise it is relative to your home directory. If the directory\n"
                                    "does not exists, it is automatically created.")
            if self.filename is not None:
                self.workdir.setText(self._workdirs.get(
                    (self.filename, self.accounts_combo.currentText()), ''))
            label.setBuddy(self.workdir)
            self._auto_workdir = True
            self.workdir.textEdited.connect(self.workdir_edited)
            dirbutton = QPushButton()
            dirbutton.setIcon(QIcon.fromTheme('folder-open'))
            dirbutton.pressed.connect(self.select_workdir)
            dirlayout = QHBoxLayout()
            dirlayout.addWidget(self.workdir)
            dirlayout.addWidget(dirbutton)
            layout.addLayout(dirlayout)

            layout2 = QHBoxLayout()
            layout2.setContentsMargins(0, 0, 0, 0)

            self.array_check = QCheckBox()
            self.array_check.setText("A&rray")
            layout2.addWidget(self.array_check)

            self.array_widget = QWidget()
            array_layout = QHBoxLayout()
            array_layout.setContentsMargins(0, 0, 0, 0)

            self.array_from = QSpinBox()
            self.array_from.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.array_from.setToolTip("First job array index.")
            label = QLabel("&start:")
            label.setBuddy(self.array_from)
            array_layout.addWidget(label)
            array_layout.addWidget(self.array_from)

            self.array_to = QSpinBox()
            self.array_to.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.array_to.setToolTip("Last job array index.")
            label = QLabel(" en&d:")
            label.setBuddy(self.array_to)
            array_layout.addWidget(label)
            array_layout.addWidget(self.array_to)

            if self._saved_array is not None:
                self.array_check.setChecked(self._saved_array[0])
                self.array_from.setValue(self._saved_array[1])
                self.array_to.setValue(self._saved_array[2])
            else:
                self.array_check.setChecked(False)
                self.array_from.setValue(0)
                self.array_to.setValue(1)

            self.array_check.stateChanged.connect(self.array_changed)
            self.array_from.valueChanged.connect(self.array_changed)
            self.array_to.valueChanged.connect(self.array_changed)

            self.array_widget.setEnabled(self.array_check.isChecked())

            self.array_widget.setLayout(array_layout)
            layout2.addWidget(self.array_widget)
            layout.addLayout(layout2)

            params_layout = QHBoxLayout()
            params_layout.setContentsMargins(0, 0, 0, 0)
            params_button = QToolButton()
            params_button.setIcon(QIcon.fromTheme('menu-down'))
            params_button.setCheckable(True)
            params_button.setChecked(False)
            params_button.toggled.connect(lambda visible: self.show_others(widget, visible))
            label = QLabel("Other submit &parameters:")
            label.setBuddy(params_button)
            params_layout.addWidget(label)
            params_layout.addWidget(params_button)
            layout.addLayout(params_layout)
            self.params = QPlainTextEdit()
            self.params.setVisible(False)
            self.params.setFixedHeight(4 * self.params.fontMetrics().height())
            self.params.setToolTip("Other submit parameters. You can use them to precisely specify\n"
                                   "requested resources etc. Please refer to batch system documentation\n"
                                   "for details.")
            if self._saved_params is not None:
                self.params.setPlainText("\n".join(self._saved_params))
            layout.addWidget(self.params)

            label = QLabel("&Log level:")
            layout.addWidget(label)
            self.loglevel = QComboBox()
            loglevels = ["Error", "Warning", "Info", "Result", "Data", "Detail", "Debug"]
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

            return widget

        def _load_accounts(self):
            accounts = CONFIG['launcher_batch/accounts']
            self.accounts = OrderedDict()
            if accounts is not None:
                accounts = accounts.split('\n')
                if not isinstance(accounts, list):
                    accounts = [accounts]
                for data in accounts:
                    name, account = Account.load(data)
                    self.accounts[name] = account

        def _save_accounts(self):
            data = []
            for name, account in self.accounts.items():
                data.append(account.save(name))
            if not data:
                del CONFIG['launcher_batch/accounts']
            else:
                CONFIG['launcher_batch/accounts'] = '\n'.join(':'.join(str(i) for i in dat) for dat in data)
            CONFIG.sync()

        def _load_workdirs(self):
            workdirs = CONFIG['launcher_batch/working_dirs']
            if workdirs is not None:
                workdirs = workdirs.split('\n')
                if not isinstance(workdirs, list):
                    workdirs = [workdirs]
                for workdir in workdirs:
                    fn, ac, wd = workdir.split(';')[:3]
                    if os.path.isfile(fn) and ac in self.accounts:
                        self._workdirs[fn, ac] = wd

        def _save_workdirs(self):
            data = '\n'.join(u"{0[0]};{0[1]};{1}".format(k,v) for k,v in self._workdirs.items()
                             if os.path.isfile(k[0]) and k[1] in self.accounts)
            if not data:
                del CONFIG['launcher_batch/working_dirs']
            else:
                CONFIG['launcher_batch/working_dirs'] = data
            CONFIG.sync()

        def array_changed(self):
            self.array_widget.setEnabled(self.array_check.isChecked())
            self._saved_array = self.array_check.isChecked(), self.array_from.value(), self.array_to.value()

        def account_add(self):
            dialog = Account.EditDialog()
            if dialog.exec_() == QDialog.Accepted:
                account = dialog.name
                if account not in self.accounts:
                    self.accounts[account] = SYSTEMS[dialog.system]()
                    self.accounts[account].update(dialog)
                    self.accounts_combo.addItem(account)
                    widget = self.accounts[account].widget()
                    self.account_widgets.append(widget)
                    self.accounts_layout.addWidget(widget)
                    self.accounts_combo.setCurrentIndex(self.accounts_combo.count()-1)
                else:
                    QMessageBox.critical(None, "Add Error",
                                               "Execution account '{}' already in the list.".format(account))
                self._save_accounts()

        def account_edit(self):
            old = self.accounts_combo.currentText()
            idx = self.accounts_combo.currentIndex()
            dialog = Account.EditDialog(self.accounts[old], old)
            if dialog.exec_() == QDialog.Accepted:
                new = dialog.name
                if new != old and new in self.accounts:
                    QMessageBox.critical(None, "Edit Error",
                                               "Execution account '{}' already in the list.".format(new))
                else:
                    if dialog.system == self.accounts[old].NAME:
                        account = self.accounts[old]
                    else:
                        account = SYSTEMS[dialog.system]()
                    account.update(dialog)
                    for i in range(len(self.accounts)):
                        k, v = self.accounts.popitem(False)
                        if k == old:
                            self.accounts[new] = account
                            widget = account.widget()
                            if self.account_widgets[i] != widget:
                                self.accounts_layout.removeWidget(self.account_widgets[i])
                                self.account_widgets[i].setParent(None)  # delete the widget
                                self.account_widgets[i] = widget
                                self.accounts_layout.insertWidget(i, widget)
                        else:
                            self.accounts[k] = v
                    self.accounts_combo.setItemText(idx, new)
                    self.account_changed(new)
                    self._save_accounts()

        def account_remove(self):
            current = self.accounts_combo.currentText()
            confirm = QMessageBox.warning(None, "Remove Account?",
                                                "Do you really want to remove the account '{}'?".format(current),
                                                QMessageBox.Yes | QMessageBox.No)
            if confirm == QMessageBox.Yes:
                index = list(self.accounts.keys()).index(current)
                self.accounts_combo.removeItem(index)
                del self.account_widgets[index]
                del self.accounts[current]
                self._save_accounts()

        def account_changed(self, account):
            if isinstance(account, int):
                self._saved_account = account
            else:
                self._saved_account = self.accounts_combo.currentIndex()
            for aw in self.account_widgets:
                aw.setVisible(False)
            self.account_widgets[self._saved_account].setVisible(True)
            if self._auto_workdir and self.filename is not None:
                self.workdir.setText(self._workdirs.get(
                    (self.filename, self.accounts_combo.currentText()), ''))
            self.dialog.setFixedHeight(self.dialog.sizeHint().height())
            self.dialog.adjustSize()

        def workdir_edited(self):
            self._auto_workdir = False

        def show_others(self, widget, visible):
            dialog = widget.parent()
            self.params.setVisible(visible)
            widget.adjustSize()
            dialog.setFixedHeight(dialog.sizeHint().height())
            dialog.adjustSize()

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
                                                 QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
                if add == QMessageBox.Cancel:
                    raise Launcher.AbortException(u'Server {} not found in known_hosts'.format(hostname))
                client.get_host_keys().add(hostname, key.get_name(), key)
                if add == QMessageBox.Yes:
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
                                                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
                    if add == QMessageBox.Cancel:
                        return
                    ssh.get_host_keys().add(err.hostname, err.key.get_name(), err.key)
                    if add == QMessageBox.Yes:
                        cls._save_host_keys(ssh.get_host_keys())
                except paramiko.AuthenticationException:
                    dialog = QInputDialog()
                    dialog.setLabelText("Password required for {}@{}. Please enter valid password:"
                                        .format(user, host))
                    dialog.setTextEchoMode(QLineEdit.Password)
                    if dialog.exec_() == QDialog.Accepted:
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
                                                        QMessageBox.Yes|QMessageBox.No)
                    if answer == QMessageBox.No:
                        return
                else:
                    return ssh

        def launch(self, main_window, args, defs):
            account_name = self.accounts_combo.currentText()
            account = self.accounts[account_name]
            user, host = account.userhost.split('@')
            port = account.port
            document = main_window.document
            workdir = self.workdir.text()
            if document.filename is not None:
                self._workdirs[document.filename, account_name] = workdir
                self._save_workdirs()
            # queue = self._saved_queue = self.queue.currentText()
            name = self.jobname.text()
            if not name:
                name = os.path.basename(document.filename) if document.filename is not None else 'unnamed'
            self._saved_params = self.params.toPlainText().split("\n")
            loglevel = ("error_details", "warning", "info", "result", "data", "detail", "debug") \
                       [self.loglevel.currentIndex()]
            array = (self.array_from.value(), self.array_to.value()) if self.array_check.isChecked() else None

            ssh = self.connect(host, user, port)
            if ssh is None: return

            if not workdir:
                _, stdout, _ = ssh.exec_command("pwd")
                workdir = stdout.read().decode('utf8').strip()
            elif not workdir.startswith('/'):
                _, stdout, _ = ssh.exec_command("pwd")
                workdir = '/'.join((stdout.read().decode('utf8').strip(), workdir))
            ssh.exec_command("mkdir -p {}".format(quote(workdir)))

            result, message = account.submit(ssh, document, args, defs, loglevel,
                                             name, workdir, array, self._saved_params)

            if message: message = "\n\n" + message
            if result:
                QMessageBox.information(None, "Job Submitted",
                                              "Job has been submitted to {}.{}".format(host, message))
            else:
                QMessageBox.critical(None, "Error Submitting Job",
                                           "Could not submit job to {}.{}".format(host, message))

        def select_workdir(self):
            user, host = self.accounts[self.accounts_combo.currentText()].userhost.split('@')
            port = self.accounts[self.accounts_combo.currentText()].port
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
            if dialog.exec_() == QDialog.Accepted:
                self.workdir.setText(dialog.item_path(dialog.tree.currentItem()))
                self._auto_workdir = False


    class RemoteDirDialog(QDialog):

        # class Item(QTreeWidgetItem):
        #     def __init__(self, *args):
        #         super(RemoteDirDialog.Item, self).__init__(*args)
        #         self.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
        #         self.read = False

        def __init__(self, sftp, host='/', path=None, parent=None):
            self.folder_icon = QIcon.fromTheme('folder')
            super(RemoteDirDialog, self).__init__(parent)
            self.setWindowTitle("Select Folder")
            self.sftp = sftp
            if path is None: path = ['']
            layout = QVBoxLayout()
            label = QLabel("Please choose a folder on the remote machine.")
            layout.addWidget(label)
            self.tree = QTreeWidget()
            self.tree.setHeaderHidden(True)
            layout.addWidget(self.tree)
            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)
            self.setLayout(layout)
            item = QTreeWidgetItem()
            item.setText(0, host)
            item.setIcon(0, QIcon.fromTheme('network-server'))
            item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
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
            if item.childIndicatorPolicy() == QTreeWidgetItem.ShowIndicator:
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
                        sub.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
                        item.addChild(sub)
                item.setChildIndicatorPolicy(QTreeWidgetItem.DontShowIndicatorWhenChildless)

LAUNCHERS.append(Launcher())
