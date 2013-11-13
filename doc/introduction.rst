.. _sec-Introduction:
************
Introduction
************

.. _sec-Instalation:

Instalation
-----------

.. _sec-Instalation-Windows:

Windows
^^^^^^^
Install complete Python distribution. We recommend Anaconda from Continuum Analytics (https://store.continuum.io/cshop/anaconda/). While installing select "Register Anaconda as default Python version of the system" option. Then run the PLaSK installer provided. We recommend selecting "Add PLaSK to the system PATH for all users" and "Create PLaSK Desktop Icon" options.

.. _sec-Instalation-Linux:

Linux systems
^^^^^^^^^^^^^

DEB based distros (Debian, Ubuntu, Mint, etc.)
""""""""""""""""""""""""""""""""""""""""""""""
Run the following commands::

	$ sudo apt-get install g++ libboost-all-dev libexpat1-dev subversion \
	  cmake cmake-qt-gui doxygen libeigen3-dev libopenblas-dev liblapack-dev \
	  liblapacke-dev libfftw3-dev python python-numpy python-scipy \
	  python-matplotlib python-h5py python-pyside ipython libqt4-dev
	$ cd Your/choosen/directory/for/project
	$ svn checkout https://phys.p.lodz.pl/svn/plask/trunk .
	$ mkdir build
	$ cd build
	$ cmake ..
	$ make -j4
	$ sudo make install

And so far that's all.

RPM based distros
"""""""""""""""""
TODO

.. _sec-Running-PLaSK:

Running PLaSK
-------------
In general PLaSK is standalone command line program. It can be run from the command line by typing ``plask`` in the shell [#shell-windows]_. If run without any parameter, it will open a simple interactive :term:`shell` where you can type Python commands for immediate execution. In order to run prepared script (saved either in *.xpl* or *.py* file) simply add its name as the first parameter. Further parameters are optional and will be available to the :term:`Python` script as ``sys.argv``. In the following example :file:`vcsel.xpl` is the input file the number ``10`` a some parameter:

In Linux :term:`shell` or MACOS terminal:

.. code-block:: bash

	joe@cray:~/laser$ plask vcsel.xpl 10

In Windows from the Command Prompt:

.. code-block:: bash

	C:\Users\joe\laser> "C:\Program Files\PLaSK\bin\plask.exe" vcsel.xpl 10

Such command-line invocation can be used to run PLaSK locally, but is also useful in HPC clusters with any kind of batch queue. In the latter case make sure that you provide the command plask :file:`your_file.xpl` to the queue, adding absolute or relative paths to files where necessary. PLaSK looks for all its system files in location relative to the main binary, so there is no need to install it in the default system location (although it is recommended). The following example shows how to add the command to the queue in a one of popular batch systems like Torque or SGE (it is assumed that PLaSK is installed in user home directory in the subdirectory :file:`plask`):

.. code-block:: bash

	joe@cray:~$ echo $HOME/plask/bin/plask laser/vcsel.xpl 10 | qsub -N vcsel

If you run PLaSK without any parameters, you enter interactive mode. Do so and once you see the prompt type (type only commands, not the prompt [#IPython-prompt]_):

	In[1]: help(plask)

You will see a Python documentation for PLaSK. Press ``'q'`` to return back to the interactive shell. In the interactive mode you can run the script in any XPL file by typing::

	In[2]: runxpl("your_file.xpl")

To exit the shell type::

	In[3]: exit()

On Windows you can find the "PLaSK Console" in the Start Menu which will start the interactive mode. Furthermore there is a simple graphical launcher provided, simply named "PLaSK". Just run it and press F5 to run your XPL or Python file in PLaSK. In the future this launcher will become a part of a more complex GUI.

Program ``plask`` may be run with several command-line parameters that -- if present -- must be specified before the script name. The most useful parameters are:

.. program:: plask

.. option:: -l loglevel

	specify the logging level used during this program run. Log levels set up in the XPL file or Python script are ignored. Possible values are: error, error_detail, warning, info, result, data, detail, or debug. Mind that it is generally bad idea to set the logging level to anything less than warning.

.. option:: -c command

	run a single command instead of the script.

.. option:: -i

	always enter the interactive console, even if there is a script name specified. All the parameters are redirected to the console.

.. option:: -version

	print PLaSK version and exit.

.. option:: variable=value

	set the value of a variable defined in XPL section <defines>. This overrides the value from the file.

.. rubric:: Footnotes
.. [#shell-windows] To be able to do so on Windows, you must add the folder bin from the PLaSK install folder (e.g. :file:`c:\\Program Files\\PLaSK\\bin`) to the environment variable Path.
.. [#IPython-prompt] If you do not have IPython installed, you will see a different prompt. It will look like this: ``'>>>'``
