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
from gui.utils import config
from gui.utils.config import CONFIG


try:
    import paramiko
except ImportError:
    pass
else:

    PASSWD_CACHE = {}

    class Printer(object):
        def __init__(self, out):
            self.out = out
        def __call__(self, line, *args, **kwargs):
            self.out.write(line.format(*args, **kwargs))
            self.out.write('\n')


    class Launcher(object):
        name = "Remote Batch Job"

        def widget(self, main_window):
            widget = QtGui.QWidget()
            layout = QtGui.QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            widget.setLayout(layout)
            layout.addWidget(QtGui.QLabel("Remote host:"))
            self.host = QtGui.QLineEdit()
            layout.addWidget(self.host)
            layout.addWidget(QtGui.QLabel("User:"))
            self.user = QtGui.QLineEdit()
            layout.addWidget(self.user)
            layout.addWidget(QtGui.QLabel("Password:"))
            self.passwd = QtGui.QLineEdit()
            self.passwd.setEchoMode(QtGui.QLineEdit.Password)
            layout.addWidget(self.passwd)
            layout.addWidget(QtGui.QLabel("Working directory:"))
            self.workdir = QtGui.QLineEdit()
            layout.addWidget(self.workdir)

            if _DEBUG:
                self.host.setText('kraken.phys.p.lodz.pl')
                self.user.setText('maciek')
                self.workdir.setText('tests')

            return widget

        def launch(self, main_window, args, defs):
            host = self.host.text()
            user = self.user.text()
            passwd = self.passwd.text()
            if not passwd: passwd = None
            workdir = self.workdir.text()
            document = main_window.document
            filename = os.path.basename(document.filename) if document.filename else \
                'unnamed.py' if isinstance(document, PyDocument) else 'unnamed.xpl'

            ssh = paramiko.SSHClient()

            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  #TODO: ask for accepting keys

            ssh.connect(host, username=user, password=passwd)

            if not workdir:
                _, stdout, _ = ssh.exec_command("pwd")
                workdir = stdout.read().decode('utf8').strip()
            elif not workdir.startswith('/'):
                _, stdout, _ = ssh.exec_command("pwd")
                workdir = '/'.join((stdout.read().decode('utf8').strip(), workdir))
            fullname = '/'.join((workdir, filename))

            sftp = ssh.open_sftp()
            sftp.open(fullname, 'w').write(document.get_content())
            sftp.close()

            stdin, stdout, stderr = ssh.exec_command("qsub".format(quote(workdir)))
            s = Printer(stdin)
            s("#!/bin/sh")
            s("#PBS -d {}", workdir)
            s("plask {2} {0} {1}\n", quote(filename),
              ' '.join(quote(a) for a in args),
              ' '.join(quote(d) for d in defs))
            stdin.flush()
            stdin.channel.shutdown_write()
            jobid = stdout.read()

            print(str(jobid))
            import sys
            sys.stdout.flush()


    if _DEBUG:  #TODO: remo
        LAUNCHERS.append(Launcher())
