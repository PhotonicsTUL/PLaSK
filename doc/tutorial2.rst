.. _sec-Optical-analysis-of-VCSEL:

Optical analysis of a step-profile VCSEL
----------------------------------------

Analysed structure
^^^^^^^^^^^^^^^^^^

In this section we will perform a simple purely optical analysis of an arsenide oxide confined Vertical-Cavity Surface-Emitting Laser (VCSEL) with an arbitrary step-profile gain. We will look for its resonant wavelength and the threshold gain using the popular effective frequency method. The schematic diagram of the VCSEL is shown in Figure :ref:`Geometry of the VCSEL <fig-tutorial2-geometry>`. It consists of 24.5 pair of the top GaAs/AlGaAs Distributed Bragg Reflectors (DBRs), a resonant cavity with a single qantum well, and 29.5 pais of the bottom GaAs/AlGaAs DBRs located on a GaAs substrate. As this the optical mode size is much slower than typical VCSEL mesa etching, we do consider the laser geometry as an infinite multi-stack of semiconductor layers. The only laterally limited fatures are the gain region and the oxide aperture necessary to confine the light.

.. _fig-tutorial2-geometry:
.. figure:: tutorial2-geometry.*
   :align: center

   Geometry of the VCSEL modeled in section :ref:`sec-Optical-analysis-of-VCSEL`.

Because of the axial symmetry of the device, the natural coordinate system used in the simulations is the two-dimensional cylindrical one. Hence, we start the tutorial2.xpl file with our geometry specification:

.. code-block:: xml

   <plask>

   <materials>
   </materials>

   <geometry>
     <cylindrical2d axes="rz" name="main" top="air" bottom="AlAs" outer="extend">

The empty :xml:tag:`<materials>` section will be discussed and expanded later. Geometry of the type :xml:tag:`cylindrical2d` means a set of axi-symmetrical disk created by rotating all two-dimensional objects around the vertical axis (*z* in this case). Its attributes ``top`` and ``bottom`` specify materials directly below and above the defined structure. ``outer="extend"`` tells PLaSK that all the outermost objects in the defined cylinder should be extended to infinity. This way we are able to simulate infinite lateral layers with only an oxide aperture located at the origin having some finite radius of 8µm. The objects outside of this aperture need to have some dimension defined, but it will be ignored as long as the outer radius of each layer is equal (we set it to 10µm).

Again, the most convenient way of defining the geometry is creating the stack and specifying the consecutive layers starting from the top. First we need to define 24 pairs of identical quarter-wavelength layers of DBR. As doing it by hand would be a tedious task, we may create another stack (within the original one) and tell PLaSK to repeat its contents 24 times:

.. code-block:: xml

   <stack>
     <stack name="top-DBR" repeat="24">
       <block dr="10" dz="0.07" material="GaAs"/>
       <block dr="10" dz="0.07945" material="Al(0.73)GaAs"/>
     </stack>

Next, according to Figure :ref:`fig-tutorial2-geometry` we finish the top DBR with one and a half more pairs, including an oxide aperture in the middle of a low refractive index *n_L* layer.

.. code-block:: xml

   <block dr="10" dz="0.07003" material="GaAs"/>
   <block dr="10" dz="0.03178" material="Al(0.73)GaAs"/>
   <shelf>
     <block dr="4" dz="0.01603" material="AlAs"/>
     <block dr="6" dz="0.01603" material="AlxOy"/>
   </shelf>
   <block dr="10" dz="0.03178" material="Al(0.73)GaAs"/>
   <block dr="10" dz="0.13756" material="GaAs"/>

we complete the definition of the geometry by specifying the cavity with the quantum well, followed by the bottom DBR.

.. code-block:: xml

       <shelf>
         <block dr="4" dz="0.005" role="gain" material="active"
                name="gain-region"/>
         <block dr="6" dz="0.005" material="inactive"/>
       </shelf>
       <block dr="10" dz="0.13756" material="GaAs"/>
       <stack name="bottom-DBR" repeat="29">
         <block dr="10" dz="0.07945" material="Al(0.73)GaAs"/>
         <block dr="10" dz="0.07003" material="GaAs"/>
       </stack>
       <block dr="10" dz="0.07945" material="Al(0.73)GaAs"/>
       </stack>
     </cylindrical2d>
   </geometry>

   </plask>

The `gain-region` block has the role *gain*, which tells the optical solver that there is a gain located in it. Note that this layer's material *active* and the adjacent *incative* are absent from the materials database. We may define these materials ourselves and set their refractive indices and absorptions to some arbitrary values. This way PLaSK offers big flexibility in analysis of new systems, where, for example, some unknown materials parameters need to be fitted to the experimental data. This is what the mysterious :xml:tag:`<materials>` section is used for. Please move back to this section and fill it with the following content:

.. code-block:: xml

   <materials>
      <material name="active" base="semiconductor">
        <nr>3.53</nr>
        <absp>0.</absp>
      </material>
      <material name="inactive" base="active">
        <absp>1000.</absp>
      </material>
   </materials>

This defines two materials with names given in name attribute of the material tags. These tags must also have the second attribute ``base``. It is either used for creating completely new materials and specifies to which group it belongs, or for providing the base material that we want to modify. In the first case, the allowed values are *"semiconductor"*, *"dielectric"*, *"oxide"*, *"metal"*, or *"liquid crystal"*. In the second case ``base`` must be a proper material specification, as used e.g. in the geometry section. In this case every undefined property of the newly defined material will be looked up in the base material. Hence, in this example, the *inactive* material will have exactly the same refractive index as the *active* one.

In the internal tags of each ``<material>`` you have to specify all the material properties used in the simulation (see chapter :ref:`sec-Materials` for the complete list). In our case, we will perform only optical simulations, so the refractive index and the absorption are sufficient.

The whole XPL file with VCSEL geometry specification is presented in :ref:`Listing of tutorial2.xpl <lis-Listing-of-tutorial2.xpl>`.


.. topic:: Listing of :file:`tutorial2.xpl`

   .. _lis-Listing-of-tutorial2.xpl:

   .. code-block:: xml

      <plask>

      <materials>
      <material name="active" base="semiconductor">
         <nr>3.53</nr>
         <absp>0.</absp>
      </material>
      <material name="inactive" base="active">
         <absp>1000.</absp>
      </material>
      </materials>

      <geometry>
      <cylindrical2d axes="rz" name="main" top="air" bottom="AlAs" outer="extend">
         <stack>
            <stack name="top-DBR" repeat="24">
            <block dr="10" dz="0.07" material="GaAs"/>
            <block dr="10" dz="0.07945" material="Al(0.73)GaAs"/>
            </stack>
            <block dr="10" dz="0.07003" material="GaAs"/>
            <block dr="10" dz="0.03178" material="Al(0.73)GaAs"/>
            <shelf>
            <block dr="4" dz="0.01603" material="AlAs"/>
            <block dr="6" dz="0.01603" material="AlxOy"/>
            </shelf>
            <block dr="10" dz="0.03178" material="Al(0.73)GaAs"/>
            <block dr="10" dz="0.13756" material="GaAs"/>
            <shelf>
            <block dr="4" dz="0.005" role="gain" material="active"
                   name="gain-region"/>
            <block dr="6" dz="0.005" material="inactive"/>
            </shelf>
            <block dr="10" dz="0.13756" material="GaAs"/>
            <stack name="bottom-DBR" repeat="29">
            <block dr="10" dz="0.07945" material="Al(0.73)GaAs"/>
            <block dr="10" dz="0.07003" material="GaAs"/>
            </stack>
            <block dr="10" dz="0.07945" material="Al(0.73)GaAs"/>
         </stack>
      </cylindrical2d>
      </geometry>

      </plask>

Organization of the computations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the section :ref:`sec-Thermo-electrical-modeling-of-simple-ee-laser` you have learned how to create an XPL file defining a basic structure and two solvers exchanging data with each other. You have also written a simple Python script controlling the computations flow in the last XPL section. However, in many cases it is common to perform exactly the same computations for different structures. In such a case it would be convenient to be able to write the definition of the solvers and the computation script separately from the geometry definition.

In order to facilitate such use-case, plask program can run with an arbitrary Python script, which will read the XPL file with the geometry definition in the next stage. By general, Python scripts have a default extension .py, so create the file tutorial2.py with the following content::

   import sys
   filename = sys.argv[1]
   loadxpl(filename)

The first line of this file is a Python command telling it to import the standard module ``sys`` [#module-sys]_. In the next line we read the first command-line argument provided while running the program and assign it to the variable filename. Finally, we read the XPL file with the given filename. As we extract this name from the command line parameters---which we will provide on the program invocation---we will be able to use the same script for many different geometries.

``loadxpl`` does not execute the script present in the ``<script>`` section of the XPL file. Instead, we should put all the commands in the rest of the :file:`tutorial2.py` file. On the other hand, this function reads and creates all solvers specified in the XPL. However, in this tutorial, we will create the solver in the Python script. In order to do this, continue writing :file:`tutorial2.py`::

   import optical

   efm = optical.EffectiveFrequencyCyl("efm")
   efm.geometry = GEO.main

This three commands are equivalent to the following definition in the :xml:tag:`<solvers>` section of the XPL file:

.. code-block:: xml

   <optical solver="EffectiveFrequencyCyl" name="efm">
     <geometry ref="main"/>
   </optical>

Mind that, while defining a solver in the Python script, we must explicitly import the solver category package (``optical`` in this case) to Python. Also, we should put its name (``"efm"`` ) as an argument of the solver constructor. It does not need to match the variable name, but it is a good idea to keep them consistent. Otherwise any logs and error messages might be hard to read.

The next line assigns the geometry named ``"main"`` present in the XPL file to the solver. This time we refer to it trough attribute access of the global dictionary ``GEO``, which is simply a shorter form of ``GEO["main"]`` [#hyphens-in-py]_. Naturally, we have assumed here that the XPL file has the geometry *"main"* defined. Luckily this is the case with our file :file:`tutorial2.xpl`.

Effective frequency solver does not need to have a mesh defined, as it will come out with a sensible default. So, in the next step, we must specify a step-profile gain as an input to the ``inGain`` receiver of the *efm* solver. To do this, we will use a ``StepProfile`` Python class, conveniently provided by PLaSK and create a custom gain::

   profile = plask.StepProfile(GEO.main, default=0.)
   profile[GEO.gain_region] = 500.

   efm.inGain = profile.outGain

The first line of the above snippet creates the ``profile`` object. ``StepProfile`` class takes a geometry in which the profile is defined as an argument. It is also possible to set the default value for every object in the geometry by providing a value to the ``default`` parameter. In the next line, we specify that there is a step gain of 500 cm\ :sup:`-1` (default units for the gain in PLaSK) at the object named gain-region in the XPL file (``-`` in names is replaced with ``_`` when using the attribute access to geometry objects). Finally, we connect the gain receiver of the ``efm`` solver with the profile's gain provider. This way, all future changes to the ``profile`` be visible from the connected solver.

Now we can perform the computations. First we set the reference wavelength to 980nm (i.e. the effective frequency will be expanded around this wavelength) and then we look for the mode with the wavelength closest to 980nm. The solver can be used more than once (e.g. to find resonant wavelengths of other modes) and it stores every solution in its attribute ``efm.modes``, which is a read-only list. The mode searching function ``efm.find_mode``, we use, returns an index of the found mode in the ``efm.modes`` list. In the code below we assign this number to the variable ``mode_number``. We can then use this number to obtain the mode's resonant wavelength and its modal losses [cm\ :sup:`-1`] either by accessing the relevant ``efm.modes`` element, or by using providers ``efm.outWavelength`` and ``efm.outLoss``, respectively. These two providers are multi-value providers, so you call them without any mesh, but with the requested mode number as their argument. The relevant part of the scipt looks as follows::

   efm.lam0 = 980.
   mode_number = efm.find_mode(980.)
   mode_wavelength = efm.outWavelength(mode_number)
   mode_loss = efm.outLoss(mode_number)
   print_log(LOG_INFO,
             "Found resonant wavelength @ %s nm, with modal loss %s /cm" %
             (mode_wavelength, mode_loss)
            )

Having written the script, we may run it from the system shell (Command Prompt in Windows) by typing:

.. code-block:: bash

    plask tutorial2.py tutorial2.xpl

In this case the string ``tutorial2.xpl`` is the program argument that is read with ``sys.argv[1]`` and which, as you remember, specifies the name of the XPL file to read. When run, the program will compute the resonant wavelength of the fundamental mode of the VCSEL, together with the losses for that mode, and print them to the screen. The modal losses will have a positive value, which means that the mode is still below threshold. We will see below, how to find the proper threshold gain value. By now, you may try to extend this script with the plot of the light intensity, which can be obtained using the ``efm.outLightIntensity`` provider. Consider this as a homework exercise, keeping in mind, that the first argument for this provider has to be the solution number (``mode_number`` in our case) and the second one, the target mesh (see :ref:`the first tutorial <sec-Thermo-electrical-modeling-of-simple-ee-laser>` for details).

Searching for the threshold gain using Scipy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We are now going to find the threshold gain of the simulated structure, which we define as the gain value in the provided ``StepProfile`` for which the material losses reach 0. This could be done by manually changing the gain value in the previous section until obtaining satisfyingly low losses, or writing an automated algorithm. But, naturally, there is better, simpler and faster solution — we may utilize the Brent root-finding algorithm from the ``scipy.optimize`` package. The function we want to use from this package is named ``brentq`` and it finds a root of a provided *f*\ (*x*) function on a provided *x* interval. You can read the function description in the *scipy* documentation at http://docs.scipy.org/doc/scipy-0.7.x/reference/generated/scipy.optimize.brentq.html.

In order to perform the root search, we have to import the ``scipy.optimize`` package and define a function (*f*\ (*x*)) that takes the gain value in the active region as it's argument (*x*) and returns the modal loss of the resonant mode (which must be 0 at the threshold i.e. *f*\ (*threshold gain*) = 0)::

   import scipy.optimize

   def loss_on_gain(gain):
       profile[GEO.gain_region] = gain
       mode_number = efm.find_mode(980.)
       return efm.outLoss(mode_number)

You notice that first, we modify the gain profile in the *gain-region* geometry object and then recompute the resonant mode. Because of the gain modification, all the modes computed ealier are lost as they become obsolete with the new gain. However, the ``mode_number`` variable in the above function will always be set to the current, recently computed, mode number we are interested in. We use this information to retrieve the computed modal loss and return it as the result of the function.

Now we can provide ``loss_on_gain`` to the ``brentq`` function, together with the gain interval, in which we expect to find the threshold (we make it 0/cm – 2500/cm). The function has to be continuous on this interval and may contain exactly one root, otherwise an error might occur. Hence, we set the reference wavelength (which is always the necessary step) and run the root search as follows::

   efm.lam0 = 980.

   threshold_gain = scipy.optimize.brentq(loss_on_gain, 0., 2500., xtol=0.1)

The ``xtol`` argument allows us to set the desired solution's tolerance.

When the ``brentq`` function completes, the ``threshold_gain`` variable contains the value we were looking for. Now we just have to set the found threshold gain and run the optical calculations for the last time and print the final result to the log::

   profile[GEO.gain_region] = threshold_gain
   mode_number = efm.find_mode(980.)
   mode_wavelength = efm.outWavelength(mode_number)
   print_log(LOG_INFO,
             "Threshold material gain is %s /cm with resonant wavelength %s nm" %
             (threshold_gain, mode_wavelength)
            )

The complete Python script (with some clean-ups) for this tutorial is presented in :ref:`Listing of tutorial2.py <lis-Listing-of-tutorial2.py>`. Feel free to expand it with the presentation of the light intensity for the found mode at threshold.

.. topic:: Content of the file :file:`tutorial2.py`.

   .. _lis-Listing-of-tutorial2.py:
   .. code-block:: python

      import sys
      import scipy.optimize

      import optical

      filename = sys.argv[1]
      loadxpl(filename)

      efm = optical.EffectiveFrequencyCyl("efm")
      efm.geometry = GEO.main

      profile = plask.StepProfile(GEO.main, default=0.)
      profile[GEO.gain_region] = 500.

      efm.inGain = profile.outGain

      def loss_on_gain(gain):
          profile[GEO.gain_region] = gain
          mode_number = efm.find_mode(980.)
          return efm.outLoss(mode_number)

      efm.lam0 = 980.

      threshold_gain = scipy.optimize.brentq(loss_on_gain, 0., 2500., xtol=0.1)

      profile[GEO.gain_region] = threshold_gain
      mode_number = efm.find_mode(980.)
      mode_wavelength = efm.outWavelength(mode_number)
      print_log(LOG_INFO,
               "Threshold material gain is %s /cm with resonant wavelength %s nm" %
               (threshold_gain, mode_wavelength)
               )

.. rubric:: Footnotes
.. [#module-sys] In Python modules are some external libraries that extend its functionality. The ``sys`` module give access to many system function and objects.
.. [#hyphens-in-py] Both these forms can be used simultaneously. However, with the attribute access you must replace any hyphens in the name (``-``) with the underscore (``_``). So ``GEO["the-geometry"]`` is equivalent to ``GEO.the_geometry``.


