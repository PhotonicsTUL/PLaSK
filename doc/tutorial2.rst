.. _sec-Optical-analysis-of-VCSEL:

Optical analysis of a step-profile VCSEL
----------------------------------------
   
Analyzed structure
^^^^^^^^^^^^^^^^^^

In this section we will perform a simple purely optical analysis of an arsenide oxide confined Vertical-Cavity Surface-Emitting Laser (VCSEL) with an arbitrary step-profile gain. We will look for its resonant wavelength and the threshold gain using the popular effective frequency method. The schematic diagram of the VCSEL is shown in Figure #. It consists of...

.. _fig-tutorial2-geometry:
.. figure:: tutorial2-geometry.*
   :align: center

   Geometry of VCSEL modeled in section :ref:`sec-Optical-analysis-of-VCSEL`.

Because of the axial symmetry of the device, the natural coordinate system used in the simulations is the two-dimensional cylindrical one. Hence, we start the tutorial2.xpl file with our geometry specification:

.. code-block:: xml

    <plask>
    
    <materials>
    </materials>
    
    <geometry>
      <cylindrical2d axes="rz" name="main" top="air" bottom="AlAs" outer="extend">
      
The empty :xml:tag:`<materials>` section will be discussed and expanded later. Geometry of the type :xml:tag:`cylindrical2d` means a set of axi-symmetrical disk created by rotating all two-dimensional objects around the vertical axis (*z* in this case). Its attributes top and bottom specify materials directly below and above the defined structure. ``outer="extend"`` tells PLaSK that all the outermost objects in the defined cylinder should be extended to infinity. This way we are able to simulate infinite lateral layers with only an oxide aperture located at the origin having some finite radius of 8µm. The objects outside of this aperture need to have some dimension defined, but it will be ignored as long as the outer radius of each layer is equal (we set it to 10µm).

Again, the most convenient way of defining the geometry is creating the stack and specifying the consecutive layers starting from the top. First we need to define 24 pairs of identical quarter-wavelength layers of DBR. As doing it by hand would be a tedious task, we may create another stack (within the original one) and tell PLaSK to repeat its contents 24 times:

.. code-block:: xml

    <stack>
          <stack name="top-DBR" repeat="24">
            <block dr="10" dz="###" material="GaAs"/>
            <block dr="10" dz="###" material="Al(0.##)GaAs/>
          </stack>
          
Next, according to Figure :ref:`fig-tutorial2-geometry` we complete the definition of the geometry by specifying the cavity with the oxide and the active region, followed by the bottom DBR.

.. code-block:: xml

          <shelf>
            <block dr="4" dz="###" material="Al(0.##)GaAs"/>
            <block dr="6" dz="###" material="AlxOy"/>
          </shelf>
          <block dr="10" dz="###" material="Al(0.##)GaAs"/>
          <block dr="10" dz="###" material="GaAs"/>
          <shelf>
            <block dr="4" dz="###" material="active" name="gain-region"/>
            <block dr="6" dz="###" material="inactive"/>
          </shelf>
          <block dr="10" dz="###" material="GaAs"/>
          <stack name="bottom-DBR" repeat="29">
            <block dr="10" dz="###" material="Al(0.##)GaAs"/>
            <block dr="10" dz="###" material="GaAs"/>
          </stack>
      </cylindrical2d>
    </geometry>

Note that there are no materials named *active* and *inactive* in the materials database. We may define these materials ourselves and set its refractive index and absorption to some arbitrary value. This way PLaSK offers big flexibility in analysis of new systems, where, for example, some unknown materials parameters need to be fitted to the experimental data. This is what the mysterious :xml:tag:`<materials>` section is used for. Please move back to this section and fill it with the following content:

.. code-block:: xml
    
    <materials>
      <material name="active" kind="semiconductor">
        <nr>3.6</nr>
        <absp>0.</absp>
      </material>
      <material name="inactive" base="active">
        <absp>1000.</absp>
      </material>
    </materials>

This defines two materials with names given in name attribute of the material tags. These tags must also have the second attribute which is either ``kind`` or ``base``. The former one is used for creating completely new materials and specifies to which group it belongs. Allowed values are *"semiconductor"*, *"dielectric"*, *"oxide"*, *"metal"*, or *"liquid crystal"*. The in the following internal tags you have to specify all the material properties used in the simulation (see chapter :ref:`sec-Materials` for the complete list). In our case, we will perform only optical simulations, so the refractive index and the absorption are sufficient.

The second defined material has the ``base`` attribute instead of ``kind``. This tells PLaSK that we want to define the modification of the existing material. The ``base`` value must be a proper material specification, as used e.g. in the geometry section. In this case every undefined property, will be looked up in the base material. Hence, in this example, the *inactive* material will have exactly the same refractive index as the *active* one.

The whole XPL file with VCSEL geometry specification is presented in :ref:`Listing of tutorial2.xpl <lis-Listing-of-tutorial2.xpl>`.

.. topic:: Listing of :file:`tutorial2.xpl`.

    .. _lis-Listing-of-tutorial2.xpl:
    .. code-block:: xml

        <plask>
        
        <materials>
          <material name="active" kind="semiconductor">
            <nr>3.6</nr>
            <absp>0.</absp>
          </material>
          <material name="inactive" base="active">
            <absp>1000.</absp>
          </material>
        </materials>
        
        <geometry>
          <cylindrical2d axes="rz" name="main" top="air" bottom="AlAs" outer="extend">
          </cylindrical2d>
        </geometry>
        </plask>

Organization of the computations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the section :ref:`sec-Thermo-electrical-modeling-of-simple-ee-laser` you have learned how to create an XPL file defining a basic structure and two solvers exchanging data with each other. Also you have written a simple Python script controlling the computations flow in the last XPL section. However, in many cases it is common to perform exactly the same computations for different structures. In such a case it would be convenient to be able to write the definition of the solvers and the computation script separately from the geometry definition.

In order to facilitate such use-case, plask program can run with an arbitrary Python script, which will read the XPL file with the geometry definition in the next stage. By general, Python scripts have a default extension .py, so create the file tutorial2.py with the following content::

    import sys
    filename = sys.argv[1]
    loadxpl(filename)

The first line of this file is a Python command telling it to import the standard module ``sys`` [#module-sys]_. In the next line we read the first command-line argument provided while running the program and assign it to the variable filename. Finally, we read the XPL file with the given filename. As we extract this name from the command line parameters, which we will provide on the program invocation, we will be able to use the same script for many different geometries.

``loadxpl`` does not execute the script present in the ``<script>`` section of the XPL file. Instead, we should put all the commands in the rest of the :file:`tutorial2.py` file. On the other hand, this function reads and creates all solvers specified in the XPL. However, in this tutorial, we will create the solver in the Python script. In order to do this, continue writing :file:`tutorial2.py`::

    efm = optical.EffectiveFrequencyCyl("efm")
    efm.geometry = GEO.main
    
This two commands are equivalent to the following definition in the :xml:tag:`<solvers>` section of the XPL file:

.. code-block:: xml

    <optical solver="EffectiveFrequencyCyl" name="efm">
      <geometry ref="main"/>
    </optical>
    
Mind that, while defining a solver in the Python script, we should put its name (``"efm"`` in this case) as an argument of the solver constructor. It does not need to match the variable name, but it is a good idea to keep them consistent. Otherwise any logs and error messages might be hard to read.

The next line assigns the geometry named ``"main"`` present in the XPL file to the solver. This time we refer to it trough attribute access of the global dictionary ``GEO``, which is simply a shorter form of ``GEO["main"]`` [#hyphens-in-py]_. Naturally, we have assumed here that the XPL file has the geometry *"main"* defined. Luckily this is the case with our file :file:`tutorial2.xpl`.

Effective frequency solver does not need to have a mesh defined, as it will come out with a sensible default. So, in the next step, we must specify a step-profile gain as an input to the ``inGain`` receiver of the *efm* solver. To do this, we will use a ``StepProfile`` Python class, conveniently provided by PLaSK and create a custom gain::

    profile = StepProfile(GEO.main)
    profile[GEO.gain_region] = 500.
    
    efm.inGain = ProviderForGain(profile)
    
The first line of the above snippet creates the ``profile`` object. ``StepProfile`` class takes a geometry in which the profile is defined as an argument. In the next line, we specify that there is a step gain of :math:`500 cm^{-1}` (default units for the gain in PLaSK) at the object named gain-region in the XPL file (``-`` in names is replaced with ``_`` when using the attribute access to geometry objects). Finally we create the temporary custom gain provider and connect it to the ``efm.inGain`` receiver.

Now we can perform the computations. First we set the reference wavelength to 980nm (i.e. the effective frequency will be expanded around this wavelength) and then we look for the mode with the wavelength closest to 981nm, storing the result in the variable lam and then writing it to the log.

::

    efm.lam0 = 980.
    lam = efm.compute(981.)
    print_log(LOG_INFO, "Found resonant wavelength " + str(lam))

The complete Python script from this tutorial is presented in :ref:`Listing of the file tutorial2.py <lis-Listing-of-tutorial2.py>`. We may run it from the system shell (Command Prompt in Windows) by typing:

.. code-block:: bash

    plask tutorial2.py tutorial2.xpl

In this case the string ``tutorial2.xpl`` is the program argument that will be read with ``sys.argv[1]`` and which, as you remember, specifies the name od the XPL file to read. When run, the program will compute the resonant wavelength of the fundamental mode of the VCSEL and print it to the screen. It will be a complex value with negative imaginary part, which means that the mode is still below threshold. We will see below, how to find the proper threshold gain value. By now, you may try to extend this script with the plot of the light intensity, which can be obtained using the ``efm.outLightIntensity`` provider. Consider this as a homework exercise.

.. topic:: Content of the file :file:`tutorial2.py`.

    .. _lis-Listing-of-tutorial2.py:
    .. code-block:: python

        import sys
        filename = sys.argv[1]
        loadxpl(filename)
        
        efm = optical.EffectiveFrequencyCyl("efm")
        efm.geometry = GEO.main
        
        profile = StepProfile(GEO.main)
        profile[GEO.gain_region] = 500.
        
        efm.inGain = ProviderForGain(profile)
        
        efm.lam0 = 980.
        lam = efm.compute(981.)
        print_log(LOG_INFO, "Found resonant wavelength " + str(lam))

.. rubric:: Footnotes
.. [#module-sys] In Python modules are some external libraries that extend its functionality. The ``sys`` module give access to many system function and objects.
.. [#hyphens-in-py] Both these forms can be used simultaneously. However, with the attribute access you must replace any hyphens in the name (``-``) with the underscore (``_``). So ``GEO["the-geometry"]`` is equivalent to ``GEO.the_geometry``.

Searching for the threshold gain using Scipy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

using scipy.optimize for fine-control of the loops
