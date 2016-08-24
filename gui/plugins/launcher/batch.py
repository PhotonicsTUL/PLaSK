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
import os.path
try:
    from shlex import quote
except ImportError:
    from pipes import quote

from gui import _DEBUG
from gui.qt import QtGui
from gui.launch import LAUNCHERS
from gui.pydocument import PyDocument
from gui.xpldocument import XPLDocument
from gui.utils import config
from gui.utils.config import CONFIG


try:
    import paramiko
except ImportError:
    pass
else:

    class AbortException(Exception):
        pass

    def hexlify(data):
        return ':'.join('{:02x}'.format(d) for d in data)

    class Printer(object):
        def __init__(self, out):
            self.out = out
        def __call__(self, line, *args, **kwargs):
            self.out.write(line.format(*args, **kwargs))
            self.out.write('\n')
        def __del__(self):
            self.out.flush()
            self.out.channel.shutdown_write()

    class Launcher(object):
        name = "Remote Batch Job"

        def widget(self, main_window):
            widget = QtGui.QWidget()
            layout = QtGui.QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            widget.setLayout(layout)
            layout.addWidget(QtGui.QLabel("Log level:"))
            self.loglevel = QtGui.QComboBox()
            loglevels = ["Error", "Warning", "Info", "Result", "Data", "Detail", "Debug"]
            self.loglevel.addItems(loglevels)
            if isinstance(main_window.document, XPLDocument):
                try:
                    self.loglevel.setCurrentIndex(loglevels.index(main_window.document.loglevel.title()))
                except ValueError:
                    self.loglevel.setCurrentIndex(5)
            else:
                self.loglevel.setCurrentIndex(5)
            layout.addWidget(self.loglevel)
            layout.addWidget(QtGui.QLabel("Remote host:"))
            self.host = QtGui.QLineEdit()
            layout.addWidget(self.host)
            layout.addWidget(QtGui.QLabel("User:"))
            self.user = QtGui.QLineEdit()
            layout.addWidget(self.user)
            layout.addWidget(QtGui.QLabel("Working directory:"))
            self.workdir = QtGui.QLineEdit()
            layout.addWidget(self.workdir)
            layout.addWidget(QtGui.QLabel("Job name:"))
            self.jobname = QtGui.QLineEdit()
            layout.addWidget(self.jobname)
            layout.addWidget(QtGui.QLabel("Queue:"))
            self.queue = QtGui.QLineEdit()
            layout.addWidget(self.queue)

            if _DEBUG:
                self.host.setText('kraken.phys.p.lodz.pl')
                self.user.setText('maciek')
                self.workdir.setText('tests')

            return widget

        PASSWD_CACHE = {}

        class AskAddPolicy(paramiko.MissingHostKeyPolicy):
            def missing_host_key(self, client, hostname, key):
                add = QtGui.QMessageBox.warning(None, "Unknown Host Key",
                                                "The host key for {} is not cached\n"
                                                "in the registry. You have no guarantee that the server\n"
                                                "is the computer you think it is.\n"
                                                "The server's {} key fingerprint is:\n"
                                                "{}\n"
                                                "If you trust this host, hit Yes to add the key to\n"
                                                "the cache and carry on connecting.\n"
                                                "If you want to carry on connecting just once, without\n"
                                                "adding the key to the cache, hit No.\n"
                                                "If you do not trust this host, hic Cancel to abandon\n"
                                                "the connection."
                                                .format(hostname, key.get_name()[4:], str(hexlify(key.get_fingerprint()))),
                                                 QtGui.QMessageBox.Yes | QtGui.QMessageBox.No | QtGui.QMessageBox.Cancel)
                if add == QtGui.QMessageBox.Cancel:
                    raise AbortException(u'Server {} not found in known_hosts'.format(hostname))
                client.get_host_keys().add(hostname, key.get_name(), key)
                if add == QtGui.QMessageBox.Yes:
                    host_keys = CONFIG.get('launcher_batch/ssh_host_keys', [])
                    host_keys.append((hostname, key))
                    CONFIG['launcher_batch/ssh_host_keys'] = host_keys
                    CONFIG.sync()

        def launch(self, main_window, args, defs):
            host = self.host.text()
            user = self.user.text()
            passwd = self.PASSWD_CACHE.get((host,user))
            workdir = self.workdir.text()
            document = main_window.document
            filetype = 'p' if isinstance(document, PyDocument) else 'x'
            queue = self.queue.text()
            if self.jobname.text():
                name = self.jobname.text()
            else:
                name = os.path.basename(document.filename) if document.filename is not None else 'unnamed'

            ssh = paramiko.SSHClient()

            ssh.load_system_host_keys()
            host_keys = ssh.get_host_keys()
            for h,k in CONFIG.get('launcher_batch/ssh_host_keys', []):
                host_keys.add(h, k.get_name(), k)
            ssh.set_missing_host_key_policy(self.AskAddPolicy())

            while True:
                try:
                    ssh.connect(host, username=user, password=passwd)
                except paramiko.BadHostKeyException as err:
                    add = QtGui.QMessageBox.warning(None, "Bad Host Key",
                                                    "WARNING - POTENTIAL SECURITY BREACH!\n\n"
                                                    "The host key for {} does not\n"
                                                    "match tone one caches in the registry. This means that\n"
                                                    "either the server administrator has changed the host key,\n"
                                                    "or you have actually connected to another computer\n"
                                                    "pretending to be the server."
                                                    "The new server's {} key fingerprint is:\n"
                                                    "{}\n"
                                                    "If you trust this host, hit Yes to add the key to\n"
                                                    "the cache and carry on connecting.\n"
                                                    "If you want to carry on connecting just once, without\n"
                                                    "adding the key to the cache, hit No.\n"
                                                    "If you do not trust this host, hic Cancel to abandon\n"
                                                    "the connection."
                                                    .format(err.hostname, err.key.get_name()[4:],
                                                            str(hexlify(err.key.get_fingerprint()))),
                                                    QtGui.QMessageBox.Yes | QtGui.QMessageBox.No | QtGui.QMessageBox.Cancel)
                    if add == QtGui.QMessageBox.Cancel:
                        return
                    ssh.get_host_keys().add(err.hostname, err.key.get_name(), err.key)
                    if add == QtGui.QMessageBox.Yes:
                        host_keys = CONFIG.get('launcher_batch/ssh_host_keys', [])
                        host_keys.append((err.hostname, err.key))
                        CONFIG['launcher_batch/ssh_host_keys'] = host_keys
                        CONFIG.sync()
                except paramiko.PasswordRequiredException:
                    dialog = QtGui.QInputDialog()
                    dialog.setLabelText("Password required for {}@{}. Please enter valid password:".format(user, host))
                    dialog.setTextEchoMode(QtGui.QLineEdit.Password)
                    if dialog.exec_() == QtGui.QDialog.Accepted:
                        passwd = self.PASSWD_CACHE[host,user] = dialog.textValue()
                    else:
                        return
                except AbortException:
                     return
                else:
                    break

            if not workdir:
                _, stdout, _ = ssh.exec_command("pwd")
                workdir = stdout.read().decode('utf8').strip()
            elif not workdir.startswith('/'):
                _, stdout, _ = ssh.exec_command("pwd")
                workdir = '/'.join((stdout.read().decode('utf8').strip(), workdir))

            try:
                stdin, stdout, stderr = ssh.exec_command("qsub".format(quote(workdir)))
                s = Printer(stdin)
                s("#!/bin/sh")
                s("#PBS -N {}", name)
                s("#PBS -q {}", queue)
                s("#PBS -d {}", workdir)
                s("plask -{ft} {0} - {1} <<PLASK_BATCH_LAUNCHER_EOF_VAEXE4TAH7\n",
                  ' '.join(quote(d) for d in defs), ' '.join(quote(a) for a in args),
                  ft=filetype)
                s.out.write(document.get_content())
                s("\nPLASK_BATCH_LAUNCHER_EOF_VAEXE4TAH7")
                del s
            except OSError:
                errors = stderr.read().decode('utf8').strip()
                if errors: errors = "\n\nError message was:\n" + errors
                QtGui.QMessageBox.critical(None, "Error Submitting Job",
                                           "Could not submit job to {}.{}".format(host, errors))
            else:
                output = stdout.read().decode('utf8').strip()
                errors = stderr.read().decode('utf8').strip()
                if output: output = "\n\n" + output
                if errors: errors = "\n\n" + errors
                QtGui.QMessageBox.information(None, "Job Submited",
                                              "Job has been submited to {}.{}{}".format(host, errors, output))


    if _DEBUG:  #TODO: remove when the launcher is ready
        LAUNCHERS.append(Launcher())
