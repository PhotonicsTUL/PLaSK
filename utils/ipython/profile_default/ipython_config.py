c = get_config()
c.TerminalInteractiveShell.confirm_exit = False
c.IPKernelApp.pylab = 'inline'
c.InteractiveShellApp.exec_lines = ['from __future__ import division']
