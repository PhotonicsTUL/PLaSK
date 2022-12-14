# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

"""
Module launching interactive session.

Based on the sympy session.py:

Copyright (c) 2006, 2007, 2008 SymPy developers

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of the SymPy nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""

import plask

BANNER = '''\

You are entering interactive mode of PLaSK.
Package 'plask' is already imported into global namespace.
'''

preexec_lines = [
'from numpy import *',
'import plask',
]

no_ipython = """\
Couldn't locate IPython. Having IPython installed is greatly recommended.
See http://ipython.scipy.org for more details. If you use Debian/Ubuntu,
just install the 'ipython' package and start plask again."""

def _init_ipython_session(argv=[]):
    """Construct new IPython session. """
    import IPython
    if IPython.__version__ >= '1.0.0':
        from IPython.terminal import ipapp
    else:
        from IPython.frontend.terminal import ipapp
    app = ipapp.TerminalIPythonApp()
    app.profile = "plask"
    app.display_banner = False
    app.exec_lines = preexec_lines
    app.initialize(argv)
    return app

def _init_python_session(argv=[]):
    """Construct new Python session. """
    from code import InteractiveConsole

    class PythonConsole(InteractiveConsole):
        """An interactive console with readline support. """

        def __init__(self):
            InteractiveConsole.__init__(self)

            try:
                import readline
            except ImportError:
                pass
            else:
                import os
                import atexit

                readline.parse_and_bind('tab: complete')

                if hasattr(readline, 'read_history_file'):
                    history = os.path.expanduser('~/.plask_history')

                    try:
                        readline.read_history_file(history)
                    except IOError:
                        pass

                    atexit.register(readline.write_history_file, history)

    console = PythonConsole()

    for line in preexec_lines:
        console.runsource(line)

    console.runsource("from sys import exit")

    import sys
    sys.argv = argv
    if argv and argv != ['']:
        console.runsource(open(argv[0]).read(), argv[0], 'exec')

    return console

def interact(ipython=None, argv=[]):
    """Initialize an embedded IPython or Python session. """
    import sys
    sys.argv = argv
    global preexec_lines, BANNER

    preexec_lines.append('from plask import *')
    preexec_lines.append('try: ion()\nexcept NameError: pass')

    in_ipython = False

    if ipython is False:
        ip = _init_python_session(argv)
        mainloop = lambda: ip.interact(BANNER)
    else:
        try:
            import IPython
            if IPython.__version__ < '0.11': raise ImportError
        except ImportError:
            if ipython is not True:
                print (no_ipython)
                ip = _init_python_session(argv)
                mainloop = lambda: ip.interact(BANNER)
            else:
                raise RuntimeError("IPython 0.11 or newer is not available on this system")
        else:
            ipython = True
            try:
                ip = get_ipython()
            except NameError:
                ip = None

            if ip is not None:
                in_ipython = True
            else:
                ip = _init_ipython_session(argv)
                if not ip:
                    raise RuntimeError("Cannot initialize IPython shell")

            try:
                if ip.shell is None: raise AttributeError
            except AttributeError:
                mainloop = ip.start
            else:
                def mainloop():
                    ip.shell.show_banner(BANNER)
                    ip.shell.mainloop()

    if not in_ipython:
        try:
            mainloop()
        except SystemExit:
            sys.stdout.write('PLaSK exiting...\n')
            return
    else:
        ip.write(BANNER)
        ip.set_hook('shutdown_hook', lambda ip: ip.write("PLaSK exiting...\n"))

if __name__ == "__main__":
    import sys
    interact(argv=sys.argv)
