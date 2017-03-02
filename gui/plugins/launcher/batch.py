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
        except (SyntaxError, ValueError):
            return default

    def _parse_int(value, default):
        try:
            return int(value)
        except ValueError:
            return default


    ACCOUNT_DATA = 'userhost', 'system', 'queues', 'color', 'program', 'bp', 'port', 'compress'

    class Batch(object):
        """
        Base class for batch systems.
        """

        prefix = None

        @classmethod
        def batch(cls, name, queue, workdir, array, path):
            return ''

        @classmethod
        def submit(cls, ssh, account, document, args, defs, loglevel, name, queue, workdir, array, others):
            fname = os.path.basename(document.filename) if document.filename is not None else 'unnamed'
            bp = account['bp']
            if bp:
                if not bp.endswith('/'): bp += '/'
                bp = quote(bp)
            command = account['program'] or 'plask'
            stdin, stdout, stderr = ssh.exec_command(cls.batch(name, queue, workdir, array, bp))
            try:
                print("#!/bin/sh", file=stdin)
                for oth in others:
                    oth = oth.strip()
                    if oth: print(cls.prefix, oth, file=stdin)
                command_line = "{cmd} -{ft} -l{ll}{lc} {defs} -:{fname} {args}".format(
                    cmd=command,
                    fname=fname,
                    defs=' '.join(quote(d) for d in defs), args=' '.join(quote(a) for a in args),
                    ll=loglevel,
                    lc=' -lansi' if account['color'] else '',
                    ft='x' if isinstance(document, XPLDocument) else 'p')
                if account['compress']:
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


    class Slurm(Batch):

        prefix = "#SBATCH"

        @classmethod
        def batch(cls, name, queue, workdir, array, path):
            if array:
                return "{path}sbatch -J {name}{queue}{array} -o {name}-%A_%a.out -D {dir}".format(
                    name=quote(name),
                    queue=(' -p ' + quote(queue)) if queue else '',
                    dir=quote(workdir),
                    array=' -a {}-{}'.format(*array),
                    path=path)
            else:
                return "{path}sbatch -J {name}{queue} -o {name}-%j.out -D {dir}".format(
                    name=quote(name),
                    queue=(' -p ' + quote(queue)) if queue else '',
                    dir=quote(workdir),
                    path=path)

        @classmethod
        def get_queues(cls, ssh, bp=''):
            if bp:
                if not bp.endswith('/'): bp += '/'
                bp = quote(bp)
            _, stdout, stderr = ssh.exec_command('{}sinfo -h -o "%R"'.format(bp))
            if stdout.channel.recv_exit_status() == 0:
                return sorted(line.strip() for line in stdout.read().decode('utf8').split("\n")[:-1])
            else:
                errors = stderr.read().decode('utf8').strip()
                QMessageBox.critical(None, "Error Retrieving Partitions",
                                           "Partition list could not be retrieved." +
                                           ("\n\n" + errors) if errors else "")
                return []


    class Torque(Batch):

        prefix = "#PBS"

        @classmethod
        def batch(cls, name, queue, workdir, array, path):
            return "{path}qsub -N {name}{queue}{array} -d {dir}".format(
                name=quote(name),
                queue=(' -q ' + quote(queue)) if queue else '',
                dir=quote(workdir),
                array='' if array is None else " -t {}-{}".format(*array),
                path=path)

        @classmethod
        def get_queues(cls, ssh, bp=''):
            if bp:
                if not bp.endswith('/'): bp += '/'
                bp = quote(bp)
            _, stdout, stderr = ssh.exec_command("{}qstat -Q".format(bp))
            if stdout.channel.recv_exit_status() == 0:
                return sorted(line.split()[0] for line in stdout.read().decode('utf8').split("\n")[2:-1])
            else:
                errors = stderr.read().decode('utf8').strip()
                QMessageBox.critical(None, "Error Retrieving Queues",
                                           "Queue list could not be retrieved." +
                                           ("\n\n" + errors) if errors else "")
                return []


    SYSTEMS = OrderedDict([('SLURM', Slurm), ('PBS/Torque', Torque)])


    class AccountEditDialog(QDialog):

        def __init__(self, launcher, name=None, account=None, parent=None):
            super(AccountEditDialog, self).__init__(parent)
            self.launcher = launcher

            if account is None: account = {}

            if 'userhost' in account:
                user, host = account['userhost'].split('@')
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
            self.port_input.setValue(account.get('port', 22))
            layout.addRow("&Port:", self.port_input)

            self.user_edit = QLineEdit()
            self.user_edit.setToolTip("Username at the execution host.")
            if user is not None:
                self.user_edit.setText(user)
            self.user_edit.textEdited.connect(self.userhost_edited)
            layout.addRow("&User:", self.user_edit)

            self.system_edit = QComboBox()
            systems = list(SYSTEMS.keys())
            self.system_edit.setToolTip("Batch job scheduling system at the execution host.\n"
                                        "If you are not sure about the correct value, contact\n"
                                        "the host administrator.")
            self.system_edit.addItems(systems)
            try:
                self.system_edit.setCurrentIndex(systems.index(account.get('system', 'SLURM')))
            except ValueError:
                pass
            layout.addRow("&Batch system:", self.system_edit)

            qbox = QVBoxLayout()
            qbox.setContentsMargins(0, 0, 0, 0)
            self.queues_list = MultiLineEdit(movable=True, placeholder='[queue name]')
            self.queues_list.setToolTip("List of available queues at the execution host.\n"
                                        "If you are not sure about the correct value, contact\n"
                                        "the host administrator.")
            self.queues_list.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
            qbox.addWidget(self.queues_list)
            if 'queues' in account:
                self.queues_list.set_values(account['queues'])
            get_queues = QPushButton("&Retrieve")
            get_queues.setToolTip("Retrieve the list of available queues automatically. To use this,\n"
                                  "you must first correctly fill-in host, user, and system fields.")
            get_queues.pressed.connect(self.get_queues)
            qbox.addWidget(get_queues)
            qwidget = QWidget()
            qwidget.setLayout(qbox)
            layout.addRow("Execution &Queues:", qwidget)

            self.color_checkbox = QCheckBox()
            self.color_checkbox.setChecked(account.get('color', False))
            layout.addRow("Co&lor Output:", self.color_checkbox)

            self.advanced = QWidget(self)
            alayout = QFormLayout()
            alayout.setContentsMargins(0, 0, 0, 0)
            self.advanced.setLayout(alayout)

            self.program_edit = QLineEdit()
            self.program_edit.setToolTip("Path to PLaSK executable. If left blank 'plask' will be used.")
            self.program_edit.setPlaceholderText("plask")
            if 'program' in account:
                self.program_edit.setText(account['program'])
            alayout.addRow("Co&mmand:", self.program_edit)

            self.bp_edit = QLineEdit()
            self.bp_edit.setToolTip("Path to directory with batch system utilities. Normally you don't need to\n"
                                    "set this, however you may need to specify it if the submit command is\n"
                                    "located in a non-standard directory.")
            if 'bp' in account:
                self.bp_edit.setText(account['bp'])
            alayout.addRow("Batch system pat&h:", self.bp_edit)

            self.compress_checkbox = QCheckBox()
            self.compress_checkbox.setToolTip("Compress script on sending to the batch system. This can make the scripts\n"
                                           "stored in batch system queues smaller, however it may be harder to track\n"
                                           "possible errors.")
            self.compress_checkbox.setChecked(account.get('compress', True))
            alayout.addRow("Compr&ess Script:", self.compress_checkbox)

            layout.addRow(self.advanced)
            self.advanced.setVisible(False)

            abutton = QPushButton("&Advanced...")
            abutton.setCheckable(True)
            abutton.toggled.connect(self.show_advanced)

            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.addButton(abutton, QDialogButtonBox.ActionRole)
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            layout.addRow(buttons)

            self.host_edit.setFocus()

        def show_advanced(self, show):
            self.advanced.setVisible(show)
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

        def get_queues(self):
            ssh = self.launcher.connect(self.host, self.user, self.port)
            if ssh is None: return
            self.queues_list.set_values(SYSTEMS[self.system].get_queues(ssh, self.bp_edit.text()))

        def accept(self):
            queues = ' '.join(self.queues_list.get_values())
            if any(':' in s for s in (self.name, self.host, self.user, queues, self.bp)) or ',' in queues:
                QMessageBox.critical(None, "Error", "Entered data contain illegal characters (:,).")
            else:
                super(AccountEditDialog, self).accept()

        @property
        def name(self):
            return self.name_edit.text()

        @property
        def data(self):
            return OrderedDict((key, getattr(self, key)) for key in ACCOUNT_DATA)

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
            return self.system_edit.currentText()

        @property
        def queues(self):
            return self.queues_list.get_values()

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


    class Launcher(object):
        name = "Remote Batch Job"

        def __init__(self):
            self._passwd_cache = {}
            self._workdirs = {}
            self._saved_account = None
            self._saved_queue = None
            self._saved_array = None

        def widget(self, main_window):
            widget = QWidget()
            layout = QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            widget.setLayout(layout)

            self.filename = main_window.document.filename

            label = QLabel("&Execution server:")
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

            label = QLabel("Execution &Queue:")
            layout.addWidget(label)
            self.queue = QComboBox()
            # self.queue.setEditable(True)
            self.queue.setToolTip("Select the execution queue to send your job to.")
            queues = self.accounts.get(self.accounts_combo.currentText(), {'queues': []})['queues']
            self.queue.addItems(queues)
            if self._saved_queue is not None:
                try:
                    qi = queues.index(self._saved_queue)
                except (IndexError, ValueError):
                    pass
                else:
                    self.queue.setCurrentIndex(qi)
            layout.addWidget(self.queue)
            label.setBuddy(self.queue)

            label = QLabel("Job &Name:")
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

            self.array_check = QCheckBox()
            self.array_check.setText("Run as A&rray")
            layout.addWidget(self.array_check)

            self.array_widget = QWidget()
            array_layout = QHBoxLayout()
            array_layout.setContentsMargins(0, 0, 0, 0)

            self.array_from = QSpinBox()
            self.array_from.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.array_from.setToolTip("First job array index.")
            label = QLabel("Array  &start:")
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
            layout.addWidget(self.array_widget)

            others_layout = QHBoxLayout()
            others_layout.setContentsMargins(0, 0, 0, 0)
            others_button = QToolButton()
            others_button.setIcon(QIcon.fromTheme('menu-down'))
            others_button.setCheckable(True)
            others_button.setChecked(False)
            others_button.toggled.connect(lambda visible: self.show_others(widget, visible))
            label = QLabel("Other submit &parameters:")
            label.setBuddy(others_button)
            others_layout.addWidget(label)
            others_layout.addWidget(others_button)
            layout.addLayout(others_layout)
            self.others = QPlainTextEdit()
            self.others.setVisible(False)
            self.others.setFixedHeight(4*self.others.fontMetrics().height())
            self.others.setToolTip("Other submit parameters. You can use them to precisely specify\n"
                                   "requested resources, or create job arrays. Please refer to batch\n"
                                   "system documentation for details.")
            layout.addWidget(self.others)

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
                for account in accounts:
                    account = account.split(':')
                    account += [''] * (len(ACCOUNT_DATA) - len(account) + 1)
                    account_name, account = account[0], OrderedDict(zip(ACCOUNT_DATA, account[1:]))
                    account['queues'] = account['queues'].split(',') if account['queues'] else []
                    account['color'] = _parse_bool(account['color'], False)
                    account['compress'] = _parse_bool(account['compress'], True)
                    account['port'] = _parse_int(account['port'], 22)
                    self.accounts[account_name] = account

        def _save_accounts(self):
            data = []
            for name, account in self.accounts.items():
                account = account.copy()
                account['queues'] = ','.join(account['queues'])
                account['color'] = int(account['color'])
                account['compress'] = int(account['compress'])
                data.append(name + ':' + ':'.join(str(v) for v in account.values()))
            if not data:
                del CONFIG['launcher_batch/accounts']
            else:
                CONFIG['launcher_batch/accounts'] = '\n'.join(data)
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
            dialog = AccountEditDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                account = dialog.name
                if account not in self.accounts:
                    self.accounts[account] = dialog.data
                    self.accounts_combo.addItem(account)
                    self.accounts_combo.setCurrentIndex(self.accounts_combo.count()-1)
                else:
                    QMessageBox.critical(None, "Add Error",
                                               "Execution account '{}' already in the list.".format(account))
                self._save_accounts()

        def account_edit(self):
            old = self.accounts_combo.currentText()
            idx = self.accounts_combo.currentIndex()
            dialog = AccountEditDialog(self, old, self.accounts[old])
            if dialog.exec_() == QDialog.Accepted:
                new = dialog.name
                if old != new:
                    if new in self.accounts:
                        QMessageBox.critical(None, "Edit Error",
                                                   "Execution account '{}' already in the list.".format(new))
                    else:
                        newdata = dialog.data
                        for i in range(len(self.accounts)):
                            k, v = self.accounts.popitem(False)
                            if k == old:
                                self.accounts[new] = newdata
                            else:
                                self.accounts[k] = v
                        self.accounts_combo.setItemText(idx, new)
                        self.account_changed(new)
                        self._save_accounts()
                else:
                    self.accounts[old] = dialog.data
                    self.account_changed(old)
                    self._save_accounts()

        def account_remove(self):
            current = self.accounts_combo.currentText()
            confirm = QMessageBox.warning(None, "Remove Account?",
                                                "Do you really want to remove the account '{}'?".format(current),
                                                QMessageBox.Yes | QMessageBox.No)
            if confirm == QMessageBox.Yes:
                self.accounts_combo.removeItem(list(self.accounts.keys()).index(current))
                del self.accounts[current]
                self._save_accounts()

        def account_changed(self, account):
            if isinstance(account, int):
                self._saved_account = account
                account = self.accounts_combo.itemText(account)
            else:
                self._saved_account = self.accounts_combo.currentIndex()
            self.queue.clear()
            self.queue.addItems(self.accounts.get(account, {'queues': []})['queues'])
            if self._auto_workdir and self.filename is not None:
                self.workdir.setText(self._workdirs.get(
                    (self.filename, self.accounts_combo.currentText()), ''))

        def workdir_edited(self):
            self._auto_workdir = False

        def show_others(self, widget, visible):
            dialog = widget.parent()
            self.others.setVisible(visible)
            widget.adjustSize()
            dialog.setFixedHeight(dialog.sizeHint().height())
            dialog.adjustSize()

        class AbortException(Exception):
            pass

        @staticmethod
        def save_host_keys(host_keys):
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
                    Launcher.save_host_keys(client.get_host_keys())

        def connect(self, host, user, port):
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
            ssh.set_missing_host_key_policy(self.AskAddPolicy())

            passwd = self._passwd_cache.get((host, user), '')

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
                        self.save_host_keys(ssh.get_host_keys())
                except paramiko.AuthenticationException:
                    dialog = QInputDialog()
                    dialog.setLabelText("Password required for {}@{}. Please enter valid password:"
                                        .format(user, host))
                    dialog.setTextEchoMode(QLineEdit.Password)
                    if dialog.exec_() == QDialog.Accepted:
                        passwd = self._passwd_cache[host, user] = dialog.textValue()
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
            user, host = account['userhost'].split('@')
            port = account['port']
            document = main_window.document
            workdir = self.workdir.text()
            if document.filename is not None:
                self._workdirs[document.filename, account_name] = workdir
                self._save_workdirs()
            system = account['system']
            queue = self._saved_queue = self.queue.currentText()
            name = self.jobname.text()
            if not name:
                name = os.path.basename(document.filename) if document.filename is not None else 'unnamed'
            others = self.others.toPlainText().split("\n")
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

            result, message = SYSTEMS[system].submit(ssh, account, document, args, defs, loglevel,
                                                     name, queue, workdir, array, others)

            if message: message = "\n\n" + message
            if result:
                QMessageBox.information(None, "Job Submitted",
                                              "Job has been submitted to {}.{}".format(host, message))
            else:
                QMessageBox.critical(None, "Error Submitting Job",
                                           "Could not submit job to {}.{}".format(host, message))

        def select_workdir(self):
            user, host = self.accounts[self.accounts_combo.currentText()]['userhost'].split('@')
            port = self.accounts[self.accounts_combo.currentText()]['port']
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
