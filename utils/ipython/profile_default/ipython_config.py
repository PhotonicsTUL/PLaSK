import plask

c = get_config()
c.TerminalInteractiveShell.confirm_exit = False
c.IPKernelApp.pylab = 'inline'
c.IPythonWidget.gui_completion = 'droplist'
c.IPythonWidget.banner = """\
PLaSK {} --- Photonic Laser Simulation Kit
(c) 2014 Lodz University of Technology, Photonics Group

You are entering interactive mode of PLaSK.
Package 'plask' is already imported into global namespace.\
""".format(plask.version)
c.InteractiveShellApp.exec_lines = ['from __future__ import division']

