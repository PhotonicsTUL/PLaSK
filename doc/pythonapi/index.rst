.. PLaSK documentation master file, created by
   sphinx-quickstart on Tue Oct  8 15:58:59 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*********************************
Welcome to PLaSK's documentation!
*********************************

.. toctree::
   :maxdepth: 2

.. highlight:: python

****************
PLaSK User Guide
****************

.. _sec-Preface:

Preface
=======
**P**\ hotonic **L**\ aser **S**\ imulation **K**\ it is a comprehensive tool for numerical analysis of broad range of physical phenomena in photonic devices. It has been designed for simulating mainly semiconductor lasers, however the range of the covered devices is much larger and includes e.g. transistors, light emitting diodes, photodetectors, etc. Due to its modular nature it is possible to perform computations of virtually any physical phenomenon in micro-scale structures.

PLaSK has been originally developed in a Photonics Group of Lodz University of Technology, which has many-year experience in numerical analysis of semiconductor lasers. Such structures, due to their complex structure, often fail to be correctly simulated by popular general-purpose software. However, in designing PLaSK we have taken special care to consider all the special cases of semiconductor lasers and to choose (or invent where necessary) appropriate algorithms.

This manual presents the software and gives instructions on using it. It is structured as follows: First there is a quick start guide, which gives brief overview on the whole software, instructs how to install it, and introduces its main features in a form of short tutorials and examples. The second part describes in detail all the elements of the input data file and presents complete manual of all the general features available in the software. The third part covers usage of graphical user interface and, finally, the last part lists all the default computational solvers and gives the complete reference to their usage.

.. _sec-Introduction:

Introduction
============

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

.. _sec-Tutorials:

Tutorials and Examples
======================

.. _sec-Thermo-electrical-modeling-of-simple-ee-laser:

Thermo-electrical modeling of simple edge-emitting diode
--------------------------------------------------------

This tutorial presents basic usage of PLaSK. It shows how to define the basic geometry and how to perform computations using solvers. During the tutorial creation of all parts of the input file is described step-by-step, together with the discussion of it meaning. Before starting following the instructions described here, please make sure that PLaSK is installed correctly and create a separate working directory. We assume that you are able to launch the main binary as described in section :ref:`sec-Running-PLaSK`. In order to begin, you need to open your favorite text editor and create a file named :file:`tutorial1.xpl`. Save it (an empty file at the moment) in the directory you have just created.

In general, PLaSK uses :term:`XML` as a format of its input files. By default they have standard extension ``.xpl`` (which you may consider as a shortcut from "**X**\ ML in **PL**\ aSK"). They contain definition of custom materials, analyzed structure geometry, configuration of the computational solvers and a Python script defining the step-by-step operations to perform during computations. However, it is a common situation that these operations remain the same for a large number of different analyzed devices. Hence, it is possible to save them in a separate file with an extension ``.py`` that is just a Python script and use the :term:`XPL` only for definition of the structure [#run-py-file]_. We will demonstrate such option in the next tutorial in section :ref:`sec-Optical-analysis-of-VCSEL`.

After :term:`reading a very brief introduction to XML syntax <XML>`, you start writing your first PLaSK input file. First you have to define the main tag ``<plask>``. Everything you write will be the content of this tag:

.. code-block:: xml

    <plask>
    
        <!-- Here you will put all the content of your file. -->
    
    </plask>

The strange characters ``<!--`` and ``-->`` in the above example mark the beginning and the end of the comment, respectively. So you can use them in your file to write your notes, or to temporarily disable any part of the file.

Inside the main tag you may define the following sections: ``materials``, ``geometry``, ``grids``, ``solvers``, ``connects``, and ``script``. All these sections are optional, however, if present, they must follow in the order specified above. You will learn the roles of these sections in this and the following tutorials. In details, they are described further in this manual.

    
Geometry
^^^^^^^^

.. _fig-tutorial1-geometry:
.. figure:: tutorial1-geometry.svg
   :align: center

   Simple edge-emitting gallium-arsenide diode modeled in :file:`tutorial1.xpl` file.


In this tutorial we start with the geometry section. It describes the geometry of the structure. We want to define a simple gallium arsenide edge-emitting diode, presented in Figure :ref:`Simple edge-emitting gallium-arsenide diode <fig-tutorial1-geometry>`. It can be modeled as a two-dimensional Cartesian structure, hence, we begin by specifying the geometry section and Cartesian two-dimensional geometry type:

.. code-block:: xml

	<plask>

	<geometry>
	  <cartesian2d axes="xy" left="mirror" length="1000" name="main">
	    <!-- Here we put the geometry definition. -->
	  </cartesian2d>
	</geometry>

	</plask>

The ``axes`` attribute of means that we will use the *xy* axes in our geometry definition i.e. *x* is the name of the horizontal axis and *y* of the vertical one. As the structure has mirror symmetry in the *x*-direction, it is sufficient to model only half of it and the left edge of the computational domain is the mirror, which is indicated by the ``left="mirror"`` attribute. The length of the chip in the third *z* direction is set to 1 mm (``length="1000"``, where the value is specified in microns as all geometrical dimensions in PLaSK). The last attribute ``name`` simply gives the geometry name (``"main"``) for later reference.

Due to the nature of the structure, it is the most natural to describe it as a stack of layers. Each layer is shifted to the left side of the stack (i.e. to the symmetry plane), which is a default. Hence, the structure definition will be (from now on we will skip ``<plask>…</plask>`` main tag from listings, although you must remember to keep them in your file):

.. code-block:: xml

	<geometry>
	  <cartesian2d axes="xy" left="mirror" length="1000" name="main">
	    <stack>
	      <block dx="1.5" dy="1.350" material="Al(0.3)GaAs:C=1e20 name="top-layer""/>
	      <block dx="150" dy="0.150" material="Al(0.3)GaAs:C=1e20"/>
	      <block dx="150" dy="0.150" material="GaAs"/>
	      <block dx="150" dy="0.007" material="In(0.2)GaAs" role="active" name="junction"/>
	      <block dx="150" dy="0.150" material="GaAs"/>
	      <block dx="150" dy="1.500" material="Al(0.3)GaAs:Si=5e19"/>
	      <block dx="150" dy="300" material="GaAs:Si=5e19" name="substrate"/>
	      <zero/>
	      <block dx="1000" dy="1000" material="Cu"/>
	    </stack>
	  </cartesian2d>
	</geometry>

In the above listing, two new tags appeared. One is ``<stack>`` and means that its whole content should be organized in the vertical :ref:`stack <geometry-object-stack>, starting from top to bottom. By default, the stack coordinate system is set in a such way that *y=0* is at the bottom of the stack. However, we want to have *y=0*, at the top of the heatsink, so indicate this by the tag ``<zero/>`` between substrate and heatsink blocks.

Another new tag is ``<block>``, which means a :ref:`rectangular block <geometry-object-block>`. As this tag has no further content, is is finished with ``/>``. ``dx`` and ``dy`` attributes give dimensions of the blocks. Their positions is determined automatically to form a compact left-aligned stack. As different layers have different widths, the empty space will be automatically filled with air to form a rectangular computational domain [#rect-mesh-skip-empty]_. At this point it is important to say that PLaSK uses fixed units for all physical quantities and they are summarized in Appendix :ref:`sec-Units-in-PLaSK`. For example all spatial dimensions must be given in micrometers, as this matches the typical dimensions of most photonic devices. Look back at the any `<block>` tag. Its attribute ``material`` gives information about the material of each block. As there is no materials section in our input file, the material parameters will be taken from default database (more on this in chapter :ref:`sec-Materials`). The value of this attribute contains the name of the material, composition of tertiary compounds and doping information. For example ``Al(0.3)GaAs:C=1e20`` means :math:`Al_{0.3}Ga_{0.7}As` (missing amount of gallium is computed automatically) doped with carbon and dopant concentration :math:`1\!\times\!10^{20}\,\mathrm{cm}^{-3}` (doping concentration is always given in :math:`\mathrm{cm}^{-3}`).

Three of the blocks are given names ``"top-layer"``, ``"substrate"``, and ``"junction"`` for the future reference. `Top-layer` and `substrate` will be used to specify boundary conditions for the electrical solver at the edges of these blocks, while we will need junction to make plots of the computed current a little easier.

You might have also noticed another attribute ``role="active"`` in one of the blocks. This is an information for the phenomenological electrical solver, which we are going to use for this structure, that the marked object is the active layer and the voltage drop on this layer should be computed using the diode equation instead on the Ohm's law. In general ``role`` attributes can be used to provide additional information about the roles of some objects (or groups -- the role could be given to the whole stack if desired) and are interpreted by solvers. You should refer to the particular solver documentation for the details of what roles should be given to what objects.


    
Mesh definition
^^^^^^^^^^^^^^^

Having the analyzed structure geometry defined, we must put the ``<grids>`` section. Its content strongly depends on the solvers we are going to use. As we plan to perform thermo-electrical calculations, we choose basic thermal solver ``thermal.Static2D`` and phenomenological electrical solver ``electrical.Shockley2D``. Details of this solvers are presented in chapter :ref:`sec-Solvers`. For now it is important that both of them require user-provided rectilinear mesh, so we define a generator for such mesh in the grids section.

In PLaSK user-defined meshes can be specified in two ways. First of all, you can simple give the points of the mesh yourself, although such approach is cumbersome and usually does not result in optimal computational mesh. However, specifying own grid can be useful for creating plots of the computed fields, as it will be shown later in this tutorial. On the other hand, PLaSK offers *mesh generators*, which can create a mesh matching certain criteria and based on the structure geometry. They are the most convenient way to make a computational mesh.

For our purpose we will use simple, but surprisingly powerful generator of two-dimensional rectilinear mesh called *DivideGenerator*. It divides the structure along the edges of all geometrical objects and then subdivides such crude cells into smaller ones according to the user wish and additionally taking care that two adjacent mesh elements do not differ more than twice in a size along each dimension.

The generator definition in XPL file is done using ``<generator>`` tag and looks as follows (put the ``<grids>`` section between ``</geometry>`` and ``</plask>``):

.. code-block:: xml

	<grids>
	  <generator type="rectilinear2d" method="divide" name="default">
	    <postdiv by="2"/>
	  </generator>
	</grids>

Here we have defined the generator for a mesh of type ``"rectilinear2d"``, using ``"divide"`` method (i.e. *DivideGenerator*). We will refer to this generator later on using the specified name "default". As in solver configuration the meshes and generators are indistinguishable by the type, each of them must have unique name.

The ``<postdiv>`` tag is the generator configuration (for more details see chapter :ref:`sec-Meshes`) and says that, after ensuring that two adjacent cells do not differ more than twice in size, each mesh element should be divided by 2 along each axis (i. e. into four quarters). The fact that our structure has both very thick and very thin layers and that we have used DivideGenerator makes the manual final mesh division by two sufficient. Later on we may plot the resulted mesh and fine-tune the ``postdiv by`` value in the XPL file, add more configuration parameters (we will do this in the next tutorial), or even automatically tune the generator from the Python script.


Computational solvers
^^^^^^^^^^^^^^^^^^^^^

As the structure geometry and the mesh generator is defined, it is time to create computational solvers. As mentioned earlier, we use ``thermal.Static2D`` for thermal modeling (i.e. computing the temperature distribution) and ``electrical.Shockley2D`` to determine the current flow. As, on the one hand, the temperature affects the material parameters (electrical conductivity in particular) and, on the other hand, the current flow is the source of the Joules heat, we will need to run both solvers in self-consistent loop, ensuring the mutual exchange of data. By now however, let us create the solvers. It is done in ``<solvers>`` section, which should immediately follow ``</grids>`` [#blank-lines-in-XML]_. Let us start with thermal solver:

.. code-block:: xml

	<solvers>

	  <thermal solver="Static2D" name="therm">
	    <geometry ref="main"/>
	    <mesh ref="default"/>
	    <temperature>
	      <condition value="300.0" place="bottom"/>
	    </temperature>
	  </thermal>

The tag name in the solvers section specified the type of the solver and the ``solver`` attribute its particular type. So in the example above we have created the ``thermal.Static2D`` solver and named it *therm*. This solver will be visible as a variable in the Python script and its name will be exactly the name specified here in the attribute ``name``. Hence, the value of this attribute must be a proper identifier i.e. must begin with a letter and contain only letters (lower or capital), digits and '_' character.

The content of the ``<thermal>`` tag specifies the configuration of the ``thermal.Static2D`` solver. Tags ``<geometry>`` and ``<mesh>`` specify the geometry and mesh or mesh generator used for computations. The values ``ref`` attributes must match the names given particular geometry and mesh in the earlier sections. The role of the ``<temperature>`` tag is to provide constant-temperature boundary conditionsboundary conditions. In this case, we have set 300K at the bottom of the whole analyzed structure (i.e. at the bottom of the copper heatsink). This location is indicated by the attribute ``place``, which is assigned a default value ``bottom``, meaning the bottom of the whole structure.

.. rubric:: Footnotes
.. [#run-py-file] Actually it is possible to run the computations without creating :term:`XPL` file at all, as everything can be defined using Python, however, this option will be presented later.
.. [#rect-mesh-skip-empty] Actually this is true only for rectangular meshes. There are special mesh types in PLaSK, which can skip empty areas from computations.
.. [#blank-lines-in-XML] You can naturally put some blank lines and comments between each section, to make your file easier to read.

****************
PLaSK Python API
****************

geometry
========
.. automodule:: plask.geometry
   :members:

mesh
====
.. automodule:: plask.mesh
   :members:

pylab
=====
.. automodule:: plask.pylab
   :members: plot_field, plot_vectors, plot_stream, plot_geometry, plot_mesh, plot_boundary, plot_material_param

hdf5
====
.. automodule:: plask.hdf5
   :members:

********
Glossary
********
.. glossary::
   
    Python
        	Python is a remarkably powerful dynamic programming language that is used in a wide variety of application domains. See: http://python.org/

    shell
        See: http://en.wikipedia.org/wiki/Command-line_interface
	
    XPL
        Standard extension of files that are used by PLaSK, a shortcut from "**X**\ ML in **PL**\ aSK"
    
    XML
        Extensible Markup Language (XML) is a markup language that defines a set of rules for encoding documents in a format that is both human-readable and machine-readable.
           
        In XML every element is described by *tags*, which are denoted by ``<…>`` brackets. Tags always have some name and may optionally contain some attributes and content. Attributes are always put together with the tag name inside the ``<…>`` brackets, i.e. ``<tag attr1="value1" attr2="value2">``. On the other hand, the tag content is always put after the brackets and must end with the ``</tag>`` element. Inside the tag content you may put text or some other tags, depending on the kind of input you want (and more importantly may) to enter. If a tag does not have any content, this must be indicated by putting the ``/`` character before the closing bracket (i.e. ``<tag/>``). Tag attributes are still allowed in such a case.
    
        See: http://en.wikipedia.org/wiki/XML
.. :ref:`short description of XML in Tutorial <desc-XML>` and  

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

