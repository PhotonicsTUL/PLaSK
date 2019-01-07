.. _sec-introduction:

************
Introduction
************

.. _sec-Instalation:

Instalation
-----------

.. _sec-Instalation-Windows:

Windows
^^^^^^^
In order to install PLaSK on MS Windows, perform two steps:

1. Install Anaconda Python Distribution **Python 3.7 version** (you must use 64-bit installer). While installing select “Register Anaconda as default Python version of the system” option. Do not install Anaconda in the folder with any spaces in the name (e.g. ``C:\Program Files\``)! This step needs to be performed only once.

2. Download and install latest build of the PLaSK Windows binary. This should be done each time the new version is issued (PLaSK graphical launcher will inform you about it automatically).

.. note::

   PLaSK for Windows is 64-bit now. You must use 64 Windows (most probably you are) and install 64-bit Python to be able to use it!

.. _sec-Instalation-Linux:

Linux Systems
^^^^^^^^^^^^^

DEB-based Distros (Debian, Ubuntu, Mint, etc.)
""""""""""""""""""""""""""""""""""""""""""""""
First download the package You should choose the correct one for your architecture (amd64 or i386) and Linux version.

Then issue the following commands:

.. code-block:: bash

   $ sudo apt-get install libexpat1 libboost-python-dev \
                          libboost-system-dev libboost-filesystem-dev \
                          libopenblas-base liblapack3 python3-numpy \
                          ipython3 python3-matplotlib python3-h5py \
                          python3-pyqt5 python3-lxml python3-yaml

   $ sudo dpkg -i your_downloaded_file.deb

where your_downloaded_file.deb is the file name of the downloaded package.

If there is no error, you can run PLaSK.


.. _sec-Running-PLaSK:

Running PLaSK from the Command Line
-----------------------------------

In general PLaSK is standalone command line program. It can be run from the command line by typing ``plask`` in the shell [#shell-windows]_. If run without any parameter, it will open a simple interactive :term:`shell` where you can type Python commands for immediate execution. In order to run prepared script (saved either in *.xpl* or *.py* file) simply add its name as the first parameter. Further parameters are optional and will be available to the :term:`Python` script as ``sys.argv``. In the following example :file:`vcsel.xpl` is the input file the number ``10`` a some parameter:

In Linux :term:`shell` or MACOS terminal:

.. code-block:: bash

   joe@cray:~/laser$ plask vcsel.xpl 10

In Windows from the Command Prompt:

.. code-block:: bat

   C:\Users\joe\laser> "C:\Program Files\PLaSK\bin\plask.exe" vcsel.xpl 10

Such command-line invocation can be used to run PLaSK locally, but is also useful in HPC clusters with any kind of batch queue. In the latter case make sure that you provide the command plask :file:`your_file.xpl` to the queue, adding absolute or relative paths to files where necessary. PLaSK looks for all its system files in location relative to the main binary, so there is no need to install it in the default system location (although it is recommended). The following example shows how to add the command to the queue in a one of popular batch systems like Torque or SGE (it is assumed that PLaSK is installed in user home directory in the subdirectory :file:`plask`):

.. code-block:: bash

   joe@cray:~$ echo $HOME/plask/bin/plask laser/vcsel.xpl 10 | qsub -N vcsel

If you run PLaSK without any parameters, you enter interactive mode. Do so and once you see the prompt type (type only commands, not the prompt [#IPython-prompt]_):

.. code-block:: python

    In[1]: help(plask)

You will see a Python documentation for PLaSK. Press ``'q'`` to return back to the interactive shell. In the interactive mode you can run the script in any XPL file by typing:

.. code-block:: python

   In[2]: runxpl('your_file.xpl')

To exit the shell type:

.. code-block:: python

   In[3]: exit()

On Windows you can find the "PLaSK Console" in the Start Menu which will start the interactive mode. Furthermore there is a simple graphical launcher provided, simply named "PLaSK". Just run it and press F5 to run your XPL or Python file in PLaSK. In the future this launcher will become a part of a more complex GUI.

Program ``plask`` may be run with several command-line parameters tha--if present--must be specified before the script name. The most useful parameters are:

.. option:: -l loglevel

      specify the logging level used during this program run. Log levels set up in the XPL file or Python script are ignored. Possible values are: ``error``, ``error_detail``, ``warning``, ``important``, ``info``, ``result``, ``data``, ``detail``, or ``debug``. Mind that it is generally a bad idea to set the logging level to anything less than warning.

.. option:: -c command

      run a single command instead of a script.

.. option:: -D key=val

      define `def` to the value `val`; this can be used only when running XPL file (the value defined in the file is ignored).

.. option:: -i

      always enter the interactive console, even if there is a script name specified. All the parameters are redirected to the console.

.. option:: -V

      print PLaSK version and exit.

.. option:: -s

      print system id for license and exit.

.. option:: variable=value

      set the value of a variable defined in XPL section <defines>. This overrides the value from the file.

Running PLaSK GUI
-----------------

*PLaSK GUI* is a separate program that helps to edit PLaSK input files. It can be started from the command line by typing ``plaskgui`` or it is available in the applications menu as other programs are. Hence, you can launch it as any other application.

.. rubric:: Footnotes
.. [#shell-windows] To be able to do so on Windows, you must add the folder bin from the PLaSK install folder (e.g. :file:`c:\\Program Files\\PLaSK\\bin`) to the environment variable Path.
.. [#IPython-prompt] If you do not have IPython installed, you will see a different prompt. It will look like this: ``'>>>'``
