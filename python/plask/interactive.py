# -*- coding: utf-8 -*-
"""
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

BANNER = '''\
You are entering interactive mode of PLaSK.
Modules 'plask' is already imported'''

preexec_source = '''\
from __future__ import division
import plask
from numpy import *
'''

no_ipython = """\
Couldn't locate IPython. Having IPython installed is greatly recommended.
See http://ipython.scipy.org for more details. If you use Debian/Ubuntu,
just install the 'ipython' package and start plask again.
"""

_import_all = False

def _init_ipython_session(argv=[]):
    """Construct new IPython session. """
    import IPython
    if IPython.__version__ >= '0.11':
        from IPython.frontend.terminal import ipapp
        app = ipapp.TerminalIPythonApp()
        app.display_banner = False
        app.initialize(argv)
        return app.shell
    else:
        from IPython.Shell import make_IPython
        return make_IPython(argv)

def _init_python_session():
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
                    history = os.path.expanduser('~/.plask-history')

                    try:
                        readline.read_history_file(history)
                    except IOError:
                        pass

                    atexit.register(readline.write_history_file, history)

    return PythonConsole()

def interact(ipython=None, argv=[]):
    """Initialize an embedded IPython or Python session. """
    import sys
    sys.argv = argv
    global preexec_source

    if _import_all_:
        preexec_source += "from plask import *\n"
        banner = BANNER + " into global namespace.\n"
    else:
        banner = BANNER + ".\n";

    in_ipython = False

    if ipython is False:
        ip = _init_python_session()
        mainloop = ip.interact
    else:
        try:
            import IPython
        except ImportError:
            if ipython is not True:
                print no_ipython
                ip = _init_python_session()
                mainloop = ip.interact
            else:
                raise RuntimeError("IPython is not available on this system")
        else:
            ipython = True
            if IPython.__version__ >= '0.11':
                try:
                    ip = get_ipython()
                except NameError:
                    ip = None
            else:
                ip = IPython.ipapi.get()
                if ip:
                    ip = ip.IP

            if ip is not None:
                in_ipython = True
            else:
                ip = _init_ipython_session(argv)

            if IPython.__version__ >= '0.11':
                # runsource is gone, use run_cell instead, which doesn't
                # take a symbol arg.  The second arg is `store_history`,
                # and False means don't add the line to IPython's history.
                ip.runsource = lambda src, symbol='exec': ip.run_cell(src, False)
                mainloop = ip.mainloop
            else:
                mainloop = ip.interact

    _preexec_source = preexec_source

    ip.runsource(_preexec_source, symbol='exec')

    if not in_ipython:
        mainloop(banner)
        sys.exit('Exiting ...')
    else:
        ip.write(banner)
        ip.set_hook('shutdown_hook', lambda ip: ip.write("Exiting ...\n"))

if __name__ == "__main__":
    import sys
    interact(argv=sys.argv)