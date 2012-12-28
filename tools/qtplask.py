#!/usr/bin/python

from IPython.frontend.qt.kernelmanager import QtKernelManager
from IPython.frontend.qt.console.rich_ipython_widget import RichIPythonWidget
from IPython.utils.traitlets import TraitError
from PySide import QtCore, QtGui

preexec_lines = [
'from __future__ import division',
'from numpy import *',
'import plask',
'from plask import *',
'plask.__globals = globals()'
]


def main():
    app = QtGui.QApplication([])

    kernel = QtKernelManager()
    kernel.start_kernel(executable="plask", extra_arguments=['--pylab=qt', '--profile=plask'])
    kernel.start_channels()

    try: # Ipython v0.13
        widget = RichIPythonWidget(gui_completion='droplist')
    except TraitError:  # IPython v0.12
        widget = RichIPythonWidget(gui_completion=True)
    widget.kernel_manager = kernel

    for line in preexec_lines:
        widget.execute(line, hidden=True)

    widget.setWindowTitle("PLaSK")
    widget.show()
    app.exec_()

# Application entry point.
if __name__ == '__main__':
    main()