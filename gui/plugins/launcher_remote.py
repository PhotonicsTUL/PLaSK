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

# plugin: Remote Launcher
# description: Launcher allowing to run computations on a remote machine through SSH.<br/>Requires Paramiko.

from __future__ import print_function, unicode_literals

import sys
import os
import re
import json

from gui.qt.QtCore import *
from gui.qt.QtGui import *
from gui.qt.QtWidgets import *
from gui.launch import LAUNCHERS
from gui.launch.dock import OutputWindow
from gui import _DEBUG

try:
    import cPickle as pickle
except ImportError:
    import pickle

import platform
system = platform.uname()[0]

try:
    import paramiko

except ImportError:
    import webbrowser
    import subprocess

    class Launcher(object):
        name = "Remote Process"

        def widget(self, main_window):
            message = QTextBrowser()
            message.setText("Remote launcher cannot be used because Python module "
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
                    return
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

    import select

    from stat import S_ISDIR

    try:
        from shlex import quote
    except ImportError:
        from pipes import quote

    from gui.xpldocument import XPLDocument
    from gui.utils.config import CONFIG

    from socket import socket, AF_INET, SOCK_STREAM, \
        error as SocketError, timeout as TimeoutException

    if system != 'Windows':
        from fcntl import fcntl, F_SETFD, FD_CLOEXEC
        from socket import AF_UNIX
    else:
        fcntl = None

    display_re = re.compile(r'^(?P<host>[^:/]*)(/(?P<proto>unix|tcp))?:(?P<disp>\d+)(\.(?P<screen>\d+))?$')

    def get_x11_display():
        display = os.environ.get('DISPLAY', '')
        m = display_re.match(display)
        if not m:
            raise ValueError("Wrong DISPLAY variable: {}".format(display))
        proto, host, disp, screen = m.group('proto', 'host', 'disp', 'screen')
        if proto == 'tcp':
            if not host:
                raise ValueError("Wrong DISPLAY variable: {}".format(display))
        elif proto == 'unix':
            host = ''
        elif (host == 'unix' and not proto) or not host:
            host, proto = '', 'unix'
        disp = int(disp)
        if screen:
            screen = int(screen)
        else:
            screen = 0
        return display, host, disp, screen


    def get_x11_socket(dname, host, disp):
        if system == 'Windows' or system == 'OpenVMS':
            if host == '':
                host = 'localhost'
            sock = socket(AF_INET, SOCK_STREAM)
            sock.connect((host, 6000 + disp))
        elif system == 'Darwin' and host and host.startswith('/private/tmp/'):
            sock = socket(AF_UNIX, SOCK_STREAM)
            sock.connect(dname)
        elif host:
            sock = socket(AF_INET, SOCK_STREAM)
            sock.connect((host, 6000 + disp))
        else:
            address = '/tmp/.X11-unix/X%d' % disp
            if not os.path.exists(address):
                address = '\0' + address
            sock = socket(AF_UNIX, SOCK_STREAM)
            sock.connect(address)
        if fcntl is not None:
            fcntl(sock.fileno(), F_SETFD, FD_CLOEXEC)
        return sock


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

    X11_DEFAULT = False if system == 'Windows' else True

    class Account(object):
        """
        Base class for account data.
        """

        def __init__(self, name, userhost=None, port=22, program='', x11=None, dirs=None):
            self.name = name
            self.userhost = userhost
            self.port = _parse_int(port, 22)
            self.program = program
            self.x11 = _parse_bool(x11, X11_DEFAULT)
            self.dirs = {} if dirs is None else dirs

        def update(self, source):
            self.userhost = source.userhost
            self.port = source.port
            self.program = source.program
            self.x11 = source.x11

        @classmethod
        def load(cls, name, config):
            kwargs = dict(config)
            if 'dirs' in kwargs:
                try:
                    try:
                        kwargs['dirs'] = json.loads(CONFIG.get('dirs', 'null'))
                    except json.JSONDecodeError:
                        kwargs['dirs'] = pickle.loads(CONFIG.get('dirs', b'N.').encode('iso-8859-1'), encoding='utf8')
                except (pickle.PickleError, EOFError):
                    del kwargs['dirs']
                CONFIG.sync()

            return cls(name, **kwargs)

        def save(self):
            return dict(userhost=self.userhost, port=self.port, program=self.program, x11=int(self.x11))

        def save_dirs(self):
            key = 'launcher_remote/accounts/{}/dirs'.format(self.name)
            CONFIG[key] = json.dumps(self.dirs)

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

                self._advanced_widgets = []

                self.program_edit = QLineEdit()
                self.program_edit.setToolTip("Path to PLaSK executable. If left blank 'plask' will be used.")
                self.program_edit.setPlaceholderText("plask")
                if account is not None and account.program:
                    self.program_edit.setText(account.program)
                layout.addRow("&Command:", self.program_edit)
                self._advanced_widgets.append(self.program_edit)

                self.x11_checkbox = QCheckBox()
                self.x11_checkbox.setToolTip(
                    "Enable X11 forwarding. This allows to see graphical output from remote jobs, but requires "
                    "a working X server. If you do not know what this means, leave the box unchanged.")
                if account is not None:
                    self.x11_checkbox.setChecked(account.x11)
                else:
                    self.x11_checkbox.setChecked(True)
                layout.addRow("&X11 Forwarding:", self.x11_checkbox)
                self._advanced_widgets.append(self.x11_checkbox)

                self._set_rows_visibility(self._advanced_widgets, False)

                abutton = QPushButton("Ad&vanced...")
                abutton.setCheckable(True)
                abutton.toggled.connect(self.show_advanced)

                buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                buttons.addButton(abutton, QDialogButtonBox.ActionRole)
                buttons.accepted.connect(self.accept)
                buttons.rejected.connect(self.reject)
                layout.addRow(buttons)

                layout.setSizeConstraint(QLayout.SetFixedSize)

                self.host_edit.setFocus()

            def _set_rows_visibility(self, widgets, state, layout=None):
                if layout is None:
                    layout = self.layout()
                for widget in widgets:
                    widget.setVisible(state)
                    layout.labelForField(widget).setVisible(state)

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
            def x11(self):
                return self.x11_checkbox.isChecked()

            @property
            def program(self):
                return self.program_edit.text()


    class RemoteLaunchThread(QThread):

        def __init__(self, ssh, account, fname, workdir, dock, main_window, args, defs):
            super(RemoteLaunchThread, self).__init__()
            self.main_window = main_window

            command = account.program or 'plask'

            self.command_line = "cd {dir}; {cmd} -u -{ft} -ldebug {defs} -:{fname} {args}".format(
                dir=quote(workdir),
                cmd=command,
                fname=fname,
                defs=' '.join(quote(d) for d in defs), args=' '.join(quote(a) for a in args),
                ft='x' if isinstance(main_window.document, XPLDocument) else 'p')

            self.ssh = ssh
            self.x11 = account.x11
            self.local_x11_display = None

            fd, fb = (s.replace(' ', '&nbsp;') for s in os.path.split(fname))
            sep = os.path.sep
            if sep == '\\':
                sep = '\\\\'
                fd = fd.replace('\\', '\\\\')
            self.link = re.compile(
                '((?:{}{})?{}(?:(?:,|:)(?: XML)? line |:))(\\d+)(.*)'.format(fd, sep, fb))

            self.dock = dock

            try:
                self.terminated.connect(self.kill_process)
            except AttributeError:
                self.finished.connect(self.kill_process)
            self.main_window.closed.connect(self.kill_process)

        def __del__(self):
            self.main_window.closed.disconnect(self.kill_process)

        def x11_handler(self, remote_channel, addrs=None):
            if remote_channel is None:
                return
            try:
                self.transport.lock.acquire()
                remote_fileno = remote_channel.fileno()
                local_channel = get_x11_socket(*self.local_x11_display)
                local_fileno = local_channel.fileno()
                self.connections[remote_fileno] = local_channel
                self.connections[local_fileno] = remote_channel
                for chan in remote_channel, local_channel:
                    if chan not in self.channels:
                        self.channels.append(chan)
                # self.transport._queue_incoming_channel(remote_channel)
                self.transport.server_accept_cv.notify()
            except:
                if _DEBUG:
                    import traceback
                    traceback.print_exc()
            finally:
                self.transport.lock.release()

        def run(self):
            self.transport = self.ssh.get_transport()
            self.transport.set_keepalive(1)
            self.session = self.transport.open_session()
            self.session.set_combine_stderr(True)
            # self.session.get_pty()

            self.connections = {}
            self.channels = [self.session]
            if self.x11:
                if self.local_x11_display is None:
                    self.local_x11_display = get_x11_display()[:3]
                self.session.request_x11(handler=self.x11_handler)

            self.session.exec_command(self.command_line)

            stdin = self.session.makefile('wb')
            stdin.write(self.main_window.document.get_content())
            stdin.flush()
            self.session.shutdown_write()

            data = b''
            while not self.session.exit_status_ready():
                try:
                    self.transport.lock.acquire()
                    channels = self.channels[:]
                finally:
                    self.transport.lock.release()
                channels, _, _= select.select(channels, [], [], 1)
                for channel in channels:
                    if channel is self.session:
                        data = self.receive(data)
                    else:
                        try:
                            self.transport.lock.acquire()
                            counterpart = self.connections[channel.fileno()]
                            counterpart.sendall(channel.recv(4096))
                        except SocketError:
                            self.channels.remove(channel)
                            self.channels.remove(counterpart)
                            del self.connections[channel.fileno()]
                            del self.connections[counterpart.fileno()]
                            channel.close()
                            counterpart.close()
                        finally:
                            self.transport.lock.release()
            self.receive(data, True)

        def receive(self, data, finish=False):
            while self.session.recv_ready():
                data += self.session.recv(4096)
                if data:
                    lines = data.splitlines()
                    if data[-1] != 10:  # b'\n'
                        data = lines[-1]
                        del lines[-1]
                    else:
                        data = b''
                    for line in lines:
                        self.dock.parse_line(line, self.link)
            if finish:
                for line in data.splitlines():
                    self.dock.parse_line(line, self.link)
                data = b''
            return data

        def kill_process(self):
            try:
                self.session.close()
            except paramiko.SSHException:
                pass


    class Launcher(object):
        name = "Remote Process"

        _passwd_cache = {}

        def __init__(self):
            self.current_account = None
            self.load_accounts()

        def widget(self, main_window, parent=None):
            widget = QWidget(parent)
            layout = QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            widget.setLayout(layout)

            self.filename = main_window.document.filename

            label = QLabel("E&xecution account:")
            layout.addWidget(label)
            accounts_layout = QHBoxLayout()
            accounts_layout.setContentsMargins(0, 0, 0, 0)
            self.accounts_combo = QComboBox()
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

            layout.addWidget(QLabel("Visible Log levels:"))
            try:
                loglevel = ['error', 'warning', 'important', 'info', 'result', 'data', 'detail', 'debug'].index(
                    main_window.document.loglevel.lower())
            except AttributeError:
                loglevel = 6
            self.error = QCheckBox("&Error")
            self.error.setChecked(loglevel >= 0)
            layout.addWidget(self.error)
            self.warning = QCheckBox("&Warning")
            self.warning.setChecked(loglevel >= 1)
            layout.addWidget(self.warning)
            self.important = QCheckBox("I&mportant")
            self.important.setChecked(loglevel >= 2)
            layout.addWidget(self.important)
            self.info = QCheckBox("&Info")
            self.info.setChecked(loglevel >= 3)
            layout.addWidget(self.info)
            self.result = QCheckBox("&Result")
            self.result.setChecked(loglevel >= 4)
            layout.addWidget(self.result)
            self.data = QCheckBox("&Data")
            self.data.setChecked(loglevel >= 5)
            layout.addWidget(self.data)
            self.detail = QCheckBox("De&tail")
            self.detail.setChecked(loglevel >= 6)
            layout.addWidget(self.detail)
            self.debug = QCheckBox("De&bug")
            self.debug.setChecked(loglevel >= 7)
            layout.addWidget(self.debug)

            layout.setContentsMargins(1, 1, 1, 1)
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

            if self.accounts:
                try:
                    self.workdir.setText(self.accounts[self.current_account].dirs.get(self.filename, ''))
                except:
                    pass

            self._widget = widget
            return widget

        def exit(self, visible):
            if self.accounts and self.current_account is not None:
                self.accounts[self.current_account].dirs[self.filename] = self.workdir.text()
            for account in self.accounts:
                account.save_dirs()
            CONFIG.sync()

        def load_accounts(self):
            self.accounts = []
            with CONFIG.group('launcher_remote/accounts') as config:
                for name, account in config.groups:
                    self.accounts.append(Account.load(name, account))

        def save_accounts(self):
            del CONFIG['launcher_remote/accounts']
            CONFIG.sync()
            with CONFIG.group('launcher_remote/accounts') as config:
                for account in self.accounts:
                    with config.group(account.name) as group:
                        for k, v in account.save().items():
                            group[k] = v
            for account in self.accounts:
                account.save_dirs()
            CONFIG.sync()


        def account_add(self):
            dialog = Account.EditDialog()
            if dialog.exec_() == QDialog.Accepted:
                name = dialog.name
                if name not in self.accounts:
                    account = Account(name)
                    account.update(dialog)
                    self.accounts.append(account)
                    self.accounts_combo.addItem(name)
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
            if dialog.exec_() == QDialog.Accepted:
                new = dialog.name
                if new != old and new in (a.name for a in self.accounts):
                    QMessageBox.critical(None, "Edit Error",
                                               "Execution account '{}' already in the list.".format(new))
                else:
                    if new != old:
                        self.accounts[idx].name = new
                    self.accounts[idx].update(dialog)
                    self.accounts_combo.setItemText(idx, new)
                    self.account_changed(new)
                    self.save_accounts()

        def account_remove(self):
            confirm = QMessageBox.warning(None, "Remove Account?",
                                          "Do you really want to remove the account '{}'?"
                                          .format(self.accounts[self.current_account].name),
                                          QMessageBox.Yes | QMessageBox.No)
            if confirm == QMessageBox.Yes:
                del self.accounts[self.current_account]
                idx = self.current_account
                self.current_account = None
                self.accounts_combo.removeItem(idx)
                self.account_changed(self.accounts_combo.currentIndex())
                self.save_accounts()
                if not self.accounts:
                    self.account_edit_button.setEnabled(False)
                    self.account_remove_button.setEnabled(False)

        def account_changed(self, index):
            if self.accounts and self.current_account is not None:
                self.accounts[self.current_account].dirs[self.filename] = self.workdir.text()
            if isinstance(index, int):
                self.current_account = index
            else:
                self.current_account = self.accounts_combo.currentIndex()
            self.workdir.setText(self.accounts[self.current_account].dirs.get(self.filename, ''))

        def show_optional(self, field, visible, button=None):
            field.setVisible(visible)
            if button is not None:
                button.setChecked(visible)

        class AbortException(Exception):
            pass

        @staticmethod
        def _save_host_keys(host_keys):
            keylist = []
            for host, keys in host_keys.items():
                for keytype, key in keys.items():
                    keylist.append('{} {} {}'.format(host, keytype, key.get_base64()))
            CONFIG['launcher_remote/ssh_host_keys'] = '\n'.join(keylist)
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
                    raise Launcher.AbortException('Server {} not found in known_hosts'.format(hostname))
                client.get_host_keys().add(hostname, key.get_name(), key)
                if add == QMessageBox.Yes:
                    Launcher._save_host_keys(client.get_host_keys())

        @classmethod
        def connect(cls, host, user, port):
            ssh = paramiko.SSHClient()
            ssh.load_system_host_keys()
            host_keys = ssh.get_host_keys()
            saved_keys = CONFIG['launcher_remote/ssh_host_keys']
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
                    ssh.connect(host, username=user, password=passwd, port=port, compress=True, timeout=5)
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
            if len(self.accounts) == 0:
                QMessageBox.critical(None, "Remote Process Error", "No remote account configured!")
                return
            account = self.accounts[self.accounts_combo.currentIndex()]
            user, host = account.userhost.split('@')
            port = account.port
            ssh = self.connect(host, user, port)
            if ssh is None: return

            filename = os.path.basename(main_window.document.filename)

            workdir = self.workdir.text()

            account.dirs[self.filename] = workdir
            account.save_dirs()
            CONFIG.sync()

            if not workdir:
                _, stdout, _ = ssh.exec_command("pwd")
                workdir = stdout.read().decode('utf8').strip()
            elif not workdir.startswith('/'):
                _, stdout, _ = ssh.exec_command("pwd")
                workdir = '/'.join((stdout.read().decode('utf8').strip(), workdir))
            ssh.exec_command("mkdir -p {}".format(quote(workdir)))

            dock = OutputWindow(self, main_window, "Launch at " + account.name)
            try:
                bottom_docked = [w for w in main_window.findChildren(QDockWidget)
                                 if main_window.dockWidgetArea(w) == (Qt.BottomDockWidgetArea)][-1]
            except IndexError:
                main_window.addDockWidget(Qt.BottomDockWidgetArea, dock)
            else:
                main_window.addDockWidget(Qt.BottomDockWidgetArea, dock)
                main_window.tabifyDockWidget(bottom_docked, dock)
                dock.show()
                dock.raise_()

            dock.thread = RemoteLaunchThread(ssh, account, filename, workdir, dock, main_window, args, defs)
            dock.thread.finished.connect(dock.thread_finished)
            dock.thread.start()

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
            if dialog.exec_() == QDialog.Accepted:
                self.workdir.setText(dialog.item_path(dialog.tree.currentItem()))


    class RemoteDirDialog(QDialog):

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
