#!/usr/bin/python
# -*- coding: utf-8 -*-

version = '0.1'

copyright = "(c) 2013 Lodz University of Technology, Institute of Physics, Photonics Group"

import sys
import os
import signal
import atexit
import subprocess

from IPython.frontend.qt.kernelmanager import QtKernelManager
from IPython.frontend.qt.console.qtconsoleapp import IPythonQtConsoleApp
from IPython.frontend.qt.console.rich_ipython_widget import RichIPythonWidget
from IPython.utils.traitlets import TraitError
from PySide import QtCore, QtGui

preexec_lines = [
'from __future__ import division',
'from numpy import *',
'import plask',
'plask.config.logging.output = "stdout"',
'plask.config.logging.coloring = "ansi"',
'plask.__globals = globals()',
'from plask import *', #TODO rely on command-line option
]

import IPython.core.usage
banner_tpl = '''\
PLaSK %s --- Photonic Laser Simulation Kit
(c) 2013 Lodz University of Technology, Institute of Physics, Photonics Group

Package 'plask' is already imported into global namespace.
'''

#def show_widget():
    #app = QtGui.QApplication([])

    #executable = os.path.join(os.path.dirname(sys.argv[0]), "plask")

    #kernel = QtKernelManager()
    #kernel.start_kernel(executable=executable, extra_arguments=['--pylab=qt', '--profile=plask'])
    #kernel.start_channels()

    #try: # Ipython v0.13
        #widget = RichIPythonWidget(gui_completion='droplist')
    #except TraitError:  # IPython v0.12
        #widget = RichIPythonWidget(gui_completion=True)
    #widget.kernel_manager = kernel

    #for line in preexec_lines:
        #widget.execute(line, hidden=True)

    #widget.setWindowTitle("PLaSK")
    #widget.show()
    #app.exec_()

class PlaskQtConsoleApp(IPythonQtConsoleApp):

    def init_kernel_manager(self):
        # Don't let Qt or ZMQ swallow KeyboardInterupts.
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        # Create a KernelManager and start a kernel.
        self.kernel_manager = self.kernel_manager_class(
                                ip=self.ip,
                                shell_port=self.shell_port,
                                iopub_port=self.iopub_port,
                                stdin_port=self.stdin_port,
                                hb_port=self.hb_port,
                                connection_file=self.connection_file,
                                config=self.config,
        )
        # start the kernel
        if not self.existing:
            self.kernel_manager.start_kernel(executable=self.executable, extra_arguments=self.kernel_argv)
        elif self.sshserver:
            # ssh, write new connection file
            self.kernel_manager.write_connection_file()
        atexit.register(self.kernel_manager.cleanup_connection_file)
        self.kernel_manager.start_channels()


    def new_frontend_master(self):
        """ Create and return new frontend attached to new kernel, launched on localhost.
        """
        kernel_manager = self.kernel_manager_class(
                                ip=self.ip,
                                connection_file=self._new_connection_file(),
                                config=self.config,
        )

        # start the kernel
        kernel_manager.start_kernel(executable=self.executable, extra_arguments=self.kernel_argv)
        kernel_manager.start_channels()

        widget = self.widget_factory(config=self.config, local_kernel=True)
        self.init_colors(widget)
        widget.kernel_manager = kernel_manager
        widget._existing = False
        widget._may_close = True
        widget._confirm_exit = self.confirm_exit

        self.init_widget(widget)

        return widget

    def init_widget(self, widget):
        for line in preexec_lines:
            widget.execute(line, hidden=True)

def get_version(executable):
    pipe = subprocess.Popen([executable, "-version"], stdout=subprocess.PIPE).stdout
    return pipe.read().strip()

def main():
    app = PlaskQtConsoleApp()
    app.executable = os.path.join(os.path.dirname(sys.argv[0]), "plask")
    IPython.core.usage.default_gui_banner = banner_tpl % get_version(app.executable)
    app.initialize(['--pylab=qt', '--profile=plask'])
    app.init_widget(app.widget)
    app.window.setWindowTitle("PLaSK")
    app.start()

# Application entry point.
if __name__ == '__main__':
    main()
