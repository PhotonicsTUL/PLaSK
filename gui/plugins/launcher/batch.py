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

import os.path
from collections import OrderedDict

try:
    from shlex import quote
except ImportError:
    from pipes import quote

from gui.qt import QtGui
from gui.launch import LAUNCHERS
from gui.xpldocument import XPLDocument
from gui.utils.config import CONFIG


try:
    import paramiko
except ImportError:
    paramiko = None
else:
    import paramiko.hostkeys

def hexlify(data):
    if isinstance(data, str):
        return ':'.join('{:02x}'.format(ord(d)) for d in data)
    else:
        return ':'.join('{:02x}'.format(d) for d in data)


class Torque(object):

    @staticmethod
    def get_queues(ssh):
        _, stdout, stderr = ssh.exec_command("qstat -Q")
        if stdout.channel.recv_exit_status() == 0:
            return sorted(line.split()[0] for line in stdout.read().decode('utf8').split("\n")[2:-1])
        else:
            errors = stderr.read().decode('utf8').strip()
            QtGui.QMessageBox.critical(None, "Error Retrieving Queues",
                                       "Queue list could not be retrieved." +
                                       ("\n\n" + errors) if errors else "")
            return []


    @staticmethod
    def submit(ssh, document, args, defs, name, queue, command, workdir, others):
        stdin, stdout, stderr = ssh.exec_command(
            "qsub -N {} -d {}{}".format(quote(name), quote(workdir), ' -q '+quote(queue) if queue else ''))
        try:
            print("#!/bin/sh", file=stdin)
            for oth in others:
                print("#PBS ", oth, file=stdin)
            print("{0} -{ft} {1} - {2} <<PLASK_BATCH_LAUNCHER_EOF_VAEXE4TAH7\n".format(command,
                ' '.join(quote(d) for d in defs), ' '.join(quote(a) for a in args),
                ft='x' if isinstance(document, XPLDocument) else 'p'), file=stdin)
            print(document.get_content(), file=stdin)
            print("\nPLASK_BATCH_LAUNCHER_EOF_VAEXE4TAH7", file=stdin)
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


SYSTEMS = OrderedDict([('Torque', Torque)])


class AccountEditDialog(QtGui.QDialog):

    def __init__(self, launcher, name=None, userhost=None, system='Torque', queues=None, program=None,
                 parent=None):
        super(AccountEditDialog, self).__init__(parent)
        self.launcher = launcher

        if userhost is not None:
            user, host = userhost.split('@')
        else:
            user = host = None

        self.setWindowTitle("SSH Server")

        layout = QtGui.QFormLayout()
        self.setLayout(layout)

        self.name_edit = QtGui.QLineEdit()
        self.name_edit.setToolTip("Friendly name of the account.")
        if name is not None:
            self.name_edit.setText(name)
            self.autoname = False
        else:
            self.autoname = True
        self.name_edit.textEdited.connect(self.name_edited)
        layout.addRow("&Name:", self.name_edit)

        self.host_edit = QtGui.QLineEdit()
        self.host_edit.setToolTip("Hostname to execute the batch job at.")
        if host is not None:
            self.host_edit.setText(host)
        self.host_edit.textEdited.connect(self.userhost_edited)
        layout.addRow("&Host:", self.host_edit)

        self.user_edit = QtGui.QLineEdit()
        self.user_edit.setToolTip("Username at the execution host.")
        if user is not None:
            self.user_edit.setText(user)
        self.user_edit.textEdited.connect(self.userhost_edited)
        layout.addRow("&User:", self.user_edit)

        self.system_edit = QtGui.QComboBox()
        systems = list(SYSTEMS.keys())
        self.system_edit.setToolTip("Batch job scheduling system at the execution host.\n"
                                    "If you are not sure about the correct value, contact\n"
                                    "the host administrator.")
        self.system_edit.addItems(systems)
        try:
            self.system_edit.setCurrentIndex(systems.index(system))
        except ValueError:
            pass
        layout.addRow("&System:", self.system_edit)

        qbox = QtGui.QVBoxLayout()
        qbox.setContentsMargins(0, 0, 0, 0)
        self.queues_edit = QtGui.QPlainTextEdit()
        self.queues_edit.setToolTip("List of available queues at the execution host.\n"
                                    "If you are not sure about the correct value, contact\n"
                                    "the host administrator.")
        qbox.addWidget(self.queues_edit)
        if queues is not None:
            self.queues_edit.setPlainText('\n'.join(queues).strip())
        get_queues = QtGui.QPushButton("&Retrieve")
        get_queues.setToolTip("Retrieve the list of available queues automatically. To use this,\n"
                              "you must first correctly fill-in host, user, and system fields.")
        get_queues.pressed.connect(self.get_queues)
        qbox.addWidget(get_queues)
        qwidget = QtGui.QWidget()
        qwidget.setLayout(qbox)
        layout.addRow("&Queues:", qwidget)

        self.program_edit = QtGui.QLineEdit()
        self.program_edit.setToolTip("Path to PLaSK executable.")
        self.program_edit.setPlaceholderText("plask")
        if program is not None:
            self.program_edit.setText(program)
        layout.addRow("Co&mmand:", self.program_edit)

        buttons = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.host_edit.setFocus()

    def name_edited(self):
        self.autoname = False

    def userhost_edited(self):
        if self.autoname:
            if self.user:
                self.name_edit.setText("{}@{}".format(self.user, self.host))
            else:
                self.name_edit.setText(self.host)

    def get_queues(self):
        ssh = self.launcher.connect(self.host, self.user)
        if ssh is None: return
        self.queues_edit.setPlainText('\n'.join(SYSTEMS[self.system].get_queues(ssh)).strip())

    def accept(self):
        queues = self.queues_edit.toPlainText()
        if any(':' in s for s in (self.name, self.host, self.user, queues)) or ',' in queues:
            QtGui.QMessageBox.critical(None, "Error", "Entered data contain illegal characters (:,).")
        else:
            super(AccountEditDialog, self).accept()

    @property
    def name(self):
        return self.name_edit.text()

    @property
    def host(self):
        return self.host_edit.text()

    @property
    def user(self):
        return self.user_edit.text()

    @property
    def system(self):
        return self.system_edit.currentText()

    @property
    def queues(self):
        text = self.queues_edit.toPlainText().strip()
        if not text: return []
        return [q.strip() for q in text.split('\n')]

    @property
    def program(self):
        return self.program_edit.text()


class Launcher(object):
    name = "Remote Batch Job"

    _passwd_cache = {}
    _saved_workdir = ''
    _saved_account = None
    _saved_queue = None

    def widget(self, main_window):
        widget = QtGui.QWidget()
        layout = QtGui.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)

        label = QtGui.QLabel("&Execution server:")
        layout.addWidget(label)
        accounts_layout = QtGui.QHBoxLayout()
        accounts_layout.setContentsMargins(0, 0, 0, 0)
        accounts = CONFIG['launcher_batch/accounts']
        if accounts is not None:
            accounts = accounts.split('\n')
            if not isinstance(accounts, list):
                accounts = [accounts]
            self.accounts = OrderedDict((a, (uh, s, q.split(',') if q else [], p)) for a,uh,s,q,p in
                                       (item.split(':') for item in accounts))
        else:
            self.accounts = OrderedDict()
        self.accounts_combo = QtGui.QComboBox()
        self.accounts_combo.addItems([s for s in self.accounts])
        self.accounts_combo.setEditText(self._saved_account)
        self.accounts_combo.textChanged.connect(self.account_changed)
        self.accounts_combo.setToolTip("Select the remote server and user to send the job to.")
        accounts_layout.addWidget(self.accounts_combo)
        account_add = QtGui.QToolButton()
        account_add.setIcon(QtGui.QIcon.fromTheme('list-add'))
        account_add.setToolTip("Add new remote server.")
        account_add.pressed.connect(self.account_add)
        accounts_layout.addWidget(account_add)
        account_edit = QtGui.QToolButton()
        account_edit.setIcon(QtGui.QIcon.fromTheme('document-properties'))
        account_edit.setToolTip("Edit the current remote server.")
        account_edit.pressed.connect(self.account_edit)
        accounts_layout.addWidget(account_edit)
        account_remove = QtGui.QToolButton()
        account_remove.setIcon(QtGui.QIcon.fromTheme('list-remove'))
        account_remove.setToolTip("Remove the current remote server.")
        account_remove.pressed.connect(self.account_remove)
        accounts_layout.addWidget(account_remove)
        layout.addLayout(accounts_layout)
        label.setBuddy(self.accounts_combo)

        label = QtGui.QLabel("Execution &Queue:")
        layout.addWidget(label)
        self.queue = QtGui.QComboBox()
        # self.queue.setEditable(True)
        self.queue.setToolTip("Select the execution queue to send your job to.")
        self.queue.addItems(self.accounts.get(self.accounts_combo.currentText(), ('','',[],''))[2])
        if self._saved_queue is not None:
            self.queue.setEditText(self._saved_queue)
        layout.addWidget(self.queue)
        label.setBuddy(self.queue)

        label = QtGui.QLabel("&Working directory:")
        layout.addWidget(label)
        self.workdir = QtGui.QLineEdit()
        self.workdir.setToolTip("Type a directory at the execution server in which the job will run.\n"
                                "If the directory starts with / it is consider as an absolute path,\n"
                                "otherwise it is relative to your home directory. If the directory\n"
                                "does not exists, it is automatically created.")
        self.workdir.setText(self._saved_workdir)
        layout.addWidget(self.workdir)
        label.setBuddy(self.workdir)

        others_layout = QtGui.QHBoxLayout()
        others_layout.setContentsMargins(0, 0, 0, 0)
        others_button = QtGui.QToolButton()
        others_button.setIcon(QtGui.QIcon.fromTheme('menu-down'))
        others_button.setCheckable(True)
        others_button.setChecked(False)
        others_button.toggled.connect(lambda visible: self.show_others(widget, visible))
        label = QtGui.QLabel("Other submit &parameters:")
        label.setBuddy(others_button)
        others_layout.addWidget(label)
        others_layout.addWidget(others_button)
        layout.addLayout(others_layout)
        self.others = QtGui.QPlainTextEdit()
        self.others.setVisible(False)
        self.others.setFixedHeight(4*self.others.fontMetrics().height())
        self.others.setToolTip("Other submit parameters. You can use them to precisely specify\n"
                               "requested resources, or create job arrays. Please refer to batch\n"
                               "system documentation for details.")
        layout.addWidget(self.others)

        label = QtGui.QLabel("&Log level:")
        layout.addWidget(label)
        self.loglevel = QtGui.QComboBox()
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

    def _save_accounts(self):
        data = ["{0}:{1[0]}:{1[1]}:{2}:{1[3]}".format(k, v, ','.join(v[2])) for k, v in self.accounts.items()]
        if not data:
            del CONFIG['launcher_batch/accounts']
        else:
            CONFIG['launcher_batch/accounts'] = '\n'.join(data)
        CONFIG.sync()

    def account_add(self):
        dialog = AccountEditDialog(self)
        if dialog.exec_() == QtGui.QDialog.Accepted:
            account = dialog.name
            userhost = "{}@{}".format(dialog.user, dialog.host)
            system = dialog.system
            queues = dialog.queues
            program = dialog.program
            if account not in self.accounts:
                self.accounts[account] = userhost, system, queues, program
                self.accounts_combo.addItem(account)
                self.accounts_combo.setCurrentIndex(self.accounts_combo.count()-1)
            else:
                self.accounts[account] = system, queues
            self._save_accounts()

    def account_edit(self):
        old = self.accounts_combo.currentText()
        idx = self.accounts_combo.currentIndex()
        dialog = AccountEditDialog(self, old, *self.accounts[old])
        if dialog.exec_() == QtGui.QDialog.Accepted:
            new = dialog.name
            if old != new:
                if new in self.accounts:
                    QtGui.QMessageBox.critical(None, "Edit Error",
                                               "Execution account {} already in the list.".format(new))
                else:
                    userhost = "{}@{}".format(dialog.user, dialog.host)
                    system = dialog.system
                    queues = dialog.queues
                    program = dialog.program
                    for i in range(len(self.accounts)):
                        k, v = self.accounts.popitem(False)
                        if k == old:
                            self.accounts[new] = userhost, system, queues, program
                        else:
                            self.accounts[k] = v
                    self.accounts_combo.setItemText(idx, new)
                    self.account_changed(new)
                    self._save_accounts()
            else:
                self.accounts[old] = "{}@{}".format(dialog.user, dialog.host),\
                                     dialog.system, dialog.queues, dialog.program
                self.account_changed(old)
                self._save_accounts()

    def account_remove(self):
        current = self.accounts_combo.currentText()
        self.accounts_combo.removeItem(list(self.accounts.keys()).index(current))
        del self.accounts[current]
        self._save_accounts()

    def account_changed(self, account):
        self.queue.clear()
        self.queue.addItems(self.accounts.get(account, ['','',[],''])[2])
        self._saved_account = account

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
            add = QtGui.QMessageBox.warning(None, "Unknown Host Key",
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
                                             QtGui.QMessageBox.Yes | QtGui.QMessageBox.No | QtGui.QMessageBox.Cancel)
            if add == QtGui.QMessageBox.Cancel:
                raise Launcher.AbortException(u'Server {} not found in known_hosts'.format(hostname))
            client.get_host_keys().add(hostname, key.get_name(), key)
            if add == QtGui.QMessageBox.Yes:
                Launcher.save_host_keys(client.get_host_keys())

    def connect(self, host, user):
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
                ssh.connect(host, username=user, password=passwd, timeout=15)
            except Launcher.AbortException:
                return
            except paramiko.BadHostKeyException as err:
                add = QtGui.QMessageBox.warning(None, "Bad Host Key",
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
                                                QtGui.QMessageBox.Yes | QtGui.QMessageBox.No | QtGui.QMessageBox.Cancel)
                if add == QtGui.QMessageBox.Cancel:
                    return
                ssh.get_host_keys().add(err.hostname, err.key.get_name(), err.key)
                if add == QtGui.QMessageBox.Yes:
                    self.save_host_keys(ssh.get_host_keys())
            except paramiko.AuthenticationException:
                dialog = QtGui.QInputDialog()
                dialog.setLabelText("Password required for {}@{}. Please enter valid password:"
                                    .format(user, host))
                dialog.setTextEchoMode(QtGui.QLineEdit.Password)
                if dialog.exec_() == QtGui.QDialog.Accepted:
                    passwd = self._passwd_cache[host, user] = dialog.textValue()
                else:
                    return
            except Exception as err:
                answer = QtGui.QMessageBox.critical(None, "Connection Error",
                                                    "Could not connect to {}.\n\n{}\n\nTry again?"
                                                    .format(host, err.message),
                                                    QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
                if answer == QtGui.QMessageBox.No:
                    return
            else:
                return ssh

    def launch(self, main_window, args, defs):
        account = self.accounts[self.accounts_combo.currentText()]
        user, host = account[0].split('@')
        workdir = self._saved_workdir = self.workdir.text()
        document = main_window.document
        system = account[1]
        queue = self._saved_queue = self.queue.currentText()
        command = account[3] if account[3] else 'plask'
        name = os.path.basename(document.filename) if document.filename is not None else 'unnamed'
        others = self.others.toPlainText().split("\n")

        ssh = self.connect(host, user)
        if ssh is None: return

        if not workdir:
            _, stdout, _ = ssh.exec_command("pwd")
            workdir = stdout.read().decode('utf8').strip()
        elif not workdir.startswith('/'):
            _, stdout, _ = ssh.exec_command("pwd")
            workdir = '/'.join((stdout.read().decode('utf8').strip(), workdir))
        ssh.exec_command("mkdir -p {}".format(quote(workdir)))

        result, message = SYSTEMS[system].submit(ssh, document, args, defs, name, queue, command, workdir, others)

        if message: message = "\n\n" + message
        if result:
            QtGui.QMessageBox.information(None, "Job Submitted",
                                          "Job has been submitted to {}.{}".format(host, message))
        else:
            QtGui.QMessageBox.critical(None, "Error Submitting Job",
                                       "Could not submit job to {}.{}".format(host, message))


if paramiko is not None:
    LAUNCHERS.append(Launcher())
