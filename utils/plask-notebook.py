#!/usr/bin/python
import sys
sys.executable = "/usr/local/bin/plask"
from IPython.frontend.terminal import ipapp
app = ipapp.TerminalIPythonApp()
app.profile = "plask"
app.display_banner = False
app.initialize(['notebook', '--pylab=inline'])
app.start()
