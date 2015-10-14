.. _sec-Optical-analysis-of-VCSEL:

Optical Analysis of a Step-Profile VCSEL
----------------------------------------

Analyzed structure
^^^^^^^^^^^^^^^^^^

In this section we will perform a simple purely optical analysis of an arsenide oxide confined Vertical-Cavity Surface-Emitting Laser (VCSEL) with an arbitrary step-profile gain. We will look for its resonant wavelength and the threshold gain using the popular effective frequency method. The schematic diagram of the VCSEL is shown in Figure :ref:`Geometry of the VCSEL <fig-tutorial2-geometry>`. It consists of 24.5 pair of the top GaAs/AlGaAs Distributed Bragg Reflectors (DBRs), a resonant cavity with a single qantum well, and 29.5 pais of the bottom GaAs/AlGaAs DBRs located on a GaAs substrate. As this the optical mode size is much lower than typical VCSEL mesa etching, we do consider the laser geometry as an infinite multi-stack of semiconductor layers. The only laterally limited features are the gain region and the oxide aperture necessary to confine the light.

.. _fig-tutorial2-geometry:
.. figure:: tutorial2-geometry.*
   :align: center

   Geometry of the VCSEL modeled in section :ref:`sec-Optical-analysis-of-VCSEL`.
   Its detailed structure is shown in the table below.

   +--------+--------------+----------------+------------------------------------------------+
   | Layer                 | Thickness [nm] | Material                                       |
   +========+==============+================+================================================+
   |        | Top DBR      | 70.0           | GaAs                                           |
   |        |              |                |                                                |
   |        | 24.5 pairs   | 79.5           | Al\ :sub:`0.73`\ GaAs                          |
   +--------+--------------+----------------+------------------------------------------------+
   |        | Oxide        | 16.0           | AlAs (*r* ≤ 4 µm) / AlOx (*r* > 4 µm)          |
   +        +--------------+----------------+------------------------------------------------+
   |        |              | 63.5           | Al\ :sub:`0.73`\ GaAs                          |
   +        +--------------+----------------+------------------------------------------------+
   | Cavity |              | 137.6          | GaAs                                           |
   +        +--------------+----------------+------------------------------------------------+
   |        | Gain Region  | 5.0            | active (*r* ≤ 4 µm) / inactive (*r* > 4 µm)    |
   +        +--------------+----------------+------------------------------------------------+
   |        |              | 137.6          | GaAs                                           |
   +--------+--------------+----------------+------------------------------------------------+
   |        | Bottom DBR   | 79.5           | Al\ :sub:`0.73`\ GaAs                          |
   |        |              |                |                                                |
   |        | 29.5 pairs   | 70.0           | GaAs                                           |
   +--------+--------------+----------------+------------------------------------------------+
   |        | Substrate    | infinite       | GaAs                                           |
   +--------+--------------+----------------+------------------------------------------------+

   The *active* material is an InGaAs with refractive index 3.53 and no absorption, and the *inactive*
   material is a similar one, however, we assume it has a constant absorption of 1000 cm\ :sup:`-1`.

Custom Materials
~~~~~~~~~~~~~~~~

Before we begin defining the geometry of our VCSEL structure, take a look a the specification of the gain region in the table above. It contains two custom materials: *active* and *inactive*. All we know about them is that they are some kind of an InGaAs with refractive index of 3.53 and specified absorption. There are no such materials in the database, so we need to define them ourselves. For this, switch to the *Materials* tab, where you should see two tables (both now empty). To the left there is a list of custom materials and to the right, the definition of its parameters. To create a new material, press ``Ctrl``\ +\ ``Shift``\ +\ ``=``. In the materials table a new material should have appeared. By default it has the name ‘new’ and the base ‘semiconductor’. Double-click the material name (or press ``N`` and change it to ``active``. Next, click |list-add| icon above the right table. This adds a new property definition of our material. Double-click the empty property name in a new row and select ``nr`` in the list and press ``Enter``. Now in the *Help* column you can see that you have selected a refractive index *n*\ :sub:`R`. In the *Value* column you can type its value, which should be 3.53 in this case. Next, add another material property ``absp``, which stands for absorption. Set its value to 0.

You do not have to define all the properties of your custom materials. The ones you skip, will be looked-up in another material, specified in the *Base* column of the material table. This way you can create variations of existing materials (either in the default database or the ones you have previously defined) with only some parameters altered. Mind however, that the default base material ‘semiconductor’ has most of its parameters unspecified, which will result in an error if you try to use them in the computations [#material-parameters]_.

To create *inactive* material, click |list-add| icon above the materials list and set the name of the new material to ``inactive`` and its base to ``active``. In the properties list add only one parameter — absorption — and set its value to 1000. The *inactive* refractive index will be taken from the previously defined *active* material.

After pressing ``F4``, verify that the XML of materials section looks as follows:

.. code-block:: xml

   <material name="active" base="semiconductor">
     <nr>3.53</nr>
     <absp>0.</absp>
   </material>
   <material name="inactive" base="active">
     <absp>1000.</absp>
   </material>

If the materials are defined properly, switch back to graphical editor and change to *Geometry* tab.

Geometry Definition
~~~~~~~~~~~~~~~~~~~

Because of the axial symmetry of the device, the natural coordinate system used in the simulations is the two-dimensional cylindrical one. Hence, after opening GUI, press ``Ctrl``\ +\ ``Shift``\ +\ ``=`` in the *Geometry* tab and create a ``Cylindrical`` geometry. Such geometry of the type :xml:tag:`cylindrical2d` means a set of axi-symmetrical disks created by rotating all two-dimensional objects around the vertical axis (we will name it *z*). In the *Basic settings* set the axes to ``r,z`` and the name to ``main``. In the *Border settings** set the bottom border (left box next to *Top/Bottom* label) to ``GaAs`` and the outer one (right box in *Inner/Outer*) to ``extend``. This way you tell PLaSK that below the your defined structure there is infinite layer of gallium arsenide and to laterally all the layers are infinitely extended in the lateral direction.  This way we are able to simulate infinite lateral layers with only an oxide aperture located at the origin having some finite diameter of 8 µm. The objects outside of this aperture need to have some dimension defined, but it will be ignored as long as the outer radius of each layer is equal (we set it to 10 µm).

Again, the most convenient way of defining the geometry is creating the stack and specifying the consecutive layers starting from the top. First we need to define 30 pairs of identical quarter-wavelength layers of DBR. As doing it by hand would be a tedious task, we may create another stack (within the original one) and tell PLaSK to repeat its contents 30 times. To do this create one stack for the whole VCSEL structure and another one within it (see :ref:`previous tutorial <sec-Thermo-electrical-modeling-of-simple-ee-laser>` if you do not remember how). Set the name of the inner stack to ``bottom-DBR`` and in the *Stack Settings* set the *Repeat* value to 24. This will make this stack to repeat its contents, creating 24 pairs of DBRs. You only need to add two rectangles defining a single DBR layer to it. So create two rectangles, both 10 µm wide. The top one should be 79.5 nm (0.0795 µm) high consisting of material ``Al(0.73)GaAs`` (meaning Al\ :sub:`0.73`\ GaAs) and the second one 70.0 nm high made of ``GaAs`` . After doing so, your geometry XML (displayed after pressing ``F4``) should look like this:

.. code-block:: xml

   <cylindrical2d name="main" axes="r,z" outer="extend" bottom="GaAs">
     <stack>
       <stack name="top-DBR" repeat="30">
         <rectangle material="Al(0.73)GaAs" dr="10" dz="0.0795"/>
         <rectangle material="GaAs" dr="10" dz="0.0700"/>
       </stack>
     </stack>
   </cylindrical2d>

Next, according to the table below Figure :ref:`fig-tutorial2-geometry` we need to add a cavity. Hence, select the outer stack (the one without the name) and add a 137.6 nm GaAs layer. Next, we need to specify the gain region. In consists of two objects located at the same level and organized from left to right. This requires a new container to be added to the stack, called ``Shelf``. It holds its items like books on  book shelf — one next to another. Hence, add such container to the stack, select it and add two rectangles to it: both should be 5 nm thick and the first one should have its first dimension (width) equal to 4 µm and the second one should be 6 µm wide. According to the table, they require materials ``active`` and ``inactive``, respectively.

As we will need to put some gain into the rectangle with the *active* material, we need to give it a name (``gain-region``) for further reference and set its role (parameter *Roles* in the *Basic Settings*) to ``gain``, which will tell the optical solver that this is where the gain is present.

After the above edits, your geometry XML should look as follows:

.. code-block:: xml
aperture
   <cylindrical2d name="main" axes="r,z" outer="extend" bottom="GaAs">
       <shelf>
         <rectangle material="active" dr="4" dz="0.0050"
                    name="gain-region" role="gain"/>
         <rectangle material="inactive" dr="6" dz="0.0050"/>
       </shelf>
       <rectangle material="GaAs" dr="10" dz="0.1376"/>
       <stack name="bottom-DBR" repeat="30">
         <rectangle material="Al(0.73)GaAs" dr="10" dz="0.0795"/>
         <rectangle material="GaAs" dr="10" dz="0.0700"/>
       </stack>
     </stack>
   </cylindrical2d>

Now, you can complete the other VCSEL layers according to Fig. :ref:`fig-tutorial2-geometry`. You will need another shelf for the oxide and another repeated stack for the top DBR. In the end, the XML of the geometry section should be similar to the one below:

.. code-block:: xml

   <cylindrical2d name="main" axes="r,z" outer="extend" bottom="GaAs">
     <stack>
       <rectangle material="GaAs" dr="10" dz="0.0700"/>
       <stack name="top-DBR" repeat="24">
         <rectangle material="Al(0.73)GaAs" dr="10" dz="0.0795"/>
         <rectangle material="GaAs" dr="10" dz="0.0700"/>
       </stack>
       <rectangle material="Al(0.73)GaAs" dr="10" dz="0.0318"/>
       <shelf>
         <rectangle material="AlAs" dr="4" dz="0.0160"/>
         <rectangle material="AlOx" dr="6" dz="0.0160"/>
       </shelf>
       <rectangle material="Al(0.73)GaAs" dr="10" dz="0.0635"/>
       <rectangle material="GaAs" dr="10" dz="0.1376"/>
       <shelf>
         <rectangle material="active" dr="4" dz="0.0050"
                    name="gain-region" role="gain"/>
         <rectangle material="inactive" dr="6" dz="0.0050"/>
       </shelf>
       <rectangle material="GaAs" dr="10" dz="0.1376"/>
       <stack name="bottom-DBR" repeat="30">
         <rectangle material="Al(0.73)GaAs" dr="10" dz="0.0795"/>
         <rectangle material="GaAs" dr="10" dz="0.0700"/>
       </stack>
     </stack>
   </cylindrical2d>

Organization of the computations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the section :ref:`sec-Thermo-electrical-modeling-of-simple-ee-laser` you have learned how to create an XPL file defining a basic structure and two solvers exchanging data with each other. You have also written a simple Python script controlling the computations flow in the last XPL section. Here, we will do the same, however, first we need to create an optical solver that will perform the computations. So, switch to the *Solvers* tab, and click the |list-add| button. In the dialog select *Category*: ``Optical``, *Solver*: ``EffectiveFrequencyCyl``, and *Name*: ``efm``. After confirming, choose the ``main`` geometry in the solver configuration. Next, in the *Mode Properties* section set *Approximate wavelength* to ``980``\  nm. This will make the XML of this section look as follows:

.. code-block:: xml

   <optical solver="EffectiveFrequencyCyl" name="efm">
     <geometry ref="main"/>
     <mode lam0="980."/>
   </optical>

Effective frequency solver does not need to have a mesh defined, as it will come out with a sensible default. However, we want to specify a step-profile gain in the gain region. This can be done only in the Python script. So, switch to the *Script* section and type::

   profile = plask.StepProfile(GEO.main, default=0.)
   profile[GEO.gain_region] = 500.

   efm.inGain = profile.outGain

The first line of the above snippet creates the ``profile`` object. The ``StepProfile`` class — conveniently provided by PLaSK — takes a geometry in which the profile is defined as an argument. It is also possible to set the default value for every object in the geometry by providing a value to the ``default`` parameter. In the next line, we specify that there is a step gain of 500 cm\ :sup:`-1` (default units for the gain in PLaSK) at the object named ‘gain-region’ in the XPL file (``-`` in names is replaced with ``_`` when using the attribute access to geometry objects) [#object-names]_. Finally, we connect the gain receiver of the ``efm`` solver with the profile's gain provider. This way, all future changes to the ``profile`` be visible from the connected solver.

Now we can perform the computations. We have already set the reference wavelength to 980nm (i.e. the effective frequency will be expanded around this wavelength) in the solver configuration. Then we look for the mode with the wavelength closest to 981.nm (we expect that the fundamental mode is at higher wavelengths). The solver can be used more than once (e.g. to find resonant wavelengths of other modes) and it stores every solution in its attribute ``efm.modes``, which is a read-only list. The mode searching function is called ``efm.find_mode``. It takes a starting wavelength approximation as its argument (we set it to 981. nm to make sure it will converge to the funcamental mode) and returns an index of a found mode in the ``efm.modes`` list. In the code below we assign this number to the variable ``mode_number``. We can then use it to obtain the mode's resonant wavelength and its modal losses [cm\ :sup:`-1`] either by accessing the relevant ``efm.modes`` element, or by using providers ``efm.outWavelength`` and ``efm.outLoss``, respectively. These two providers are multi-value providers, so you call them without any mesh, but with the requested mode number as their argument. The relevant part of the scipt looks as follows::

   efm.lam0 = 980.
   mode_number = efm.find_mode(981.)
   mode_wavelength = efm.outWavelength(mode_number)
   mode_loss = efm.outLoss(mode_number)
   print_log(LOG_INFO,
       "Threshold material gain is {:.0f}/cm with resonant wavelength {:.2f}nm"
       .format(threshold_gain, mode_wavelength))

Having written the script, we may run it by pressing ``F5`` in the GUI. The program will compute the resonant wavelength of the fundamental mode of the VCSEL, together with the losses for that mode, and print them to the screen. The modal losses will have a positive value, which means that the mode is still below threshold. We will see below, how to find the proper threshold gain value. By now, you may try to extend this script with the plot of the light intensity, which can be obtained using the ``efm.outLightMagnitude`` provider. Consider this as a homework exercise, keeping in mind, that the first argument for this provider has to be the solution number (``mode_number`` in our case) and the second one, the target mesh (see :ref:`the first tutorial <sec-Thermo-electrical-modeling-of-simple-ee-laser>` for details).

Searching for the threshold gain using Scipy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We are now going to find the threshold gain of the simulated structure, which we define as the gain value in the provided ``StepProfile`` for which the material losses reach 0. This could be done by manually changing the gain value in the previous section until obtaining satisfyingly low losses, or writing an automated algorithm. But, naturally, there is better, simpler and faster solution — we may utilize the Brent root-finding algorithm from the ``scipy.optimize`` module [#module-scipy-optimize]_. The function we want to use from this module is named ``fsolve`` and it finds a root of a provided *f*\ (*x*) function starting from a given *x* value. You can read the function description in the *scipy* documentation at http://docs.scipy.org/doc/scipy-0.7.x/reference/generated/scipy.optimize.fsolve.html.

In order to perform the root search, we have to import the ``scipy.optimize`` module, using Python built-in command ``import``, and define a function (*f*\ (*x*)) that takes the gain value in the active region as it's argument (*x*) and returns the modal loss of the resonant mode (which must be 0 at the threshold i.e. *f*\ (*threshold gain*) = 0)::

   import scipy.optimize

   def loss_on_gain(gain):
       profile[GEO.gain_region] = gain
       mode_number = efm.find_mode(981.)
       return efm.outLoss(mode_number)

You notice that first, we modify the gain profile in the *gain-region* geometry object and then recompute the resonant mode. Because of the gain modification, all the modes computed earlier are lost as they become obsolete with the new gain. However, the ``mode_number`` variable in the above function will always be set to the current, recently computed, mode number we are interested in. We use this information to retrieve the computed modal loss and return it as the result of the function.

Now we can provide ``loss_on_gain`` to the ``fsolve`` function, together with the gain value, which we expect to be near the threshold (2000/cm). The function has to be continuous on this interval and may contain exactly one root, otherwise an error might occur. Hence, we set the reference wavelength (which is always the necessary step) and run the root search as follows::

   efm.lam0 = 980.

   threshold_gain = scipy.optimize.fsolve(loss_on_gain, 2000., xtol=0.1)[0]

The ``xtol`` argument allows us to set the desired solution's tolerance.

When the ``fsolve`` function completes it returns a Python list with the found solutions (which in this case hase only one element), so the ``threshold_gain`` variable contains the value we were looking for. Now we just have to set the found threshold gain and run the optical calculations for the last time and print the final result to the log::

   profile[GEO.gain_region] = threshold_gain
   mode_number = efm.find_mode(981.)
   mode_wavelength = efm.outWavelength(mode_number)
   print_log(LOG_INFO,
       "Threshold material gain is {:.0f}/cm with resonant wavelength {:.2f}nm"
       .format(threshold_gain, mode_wavelength))

The complete Python script (with some clean-ups) for this tutorial is presented in the :ref:`listin <lis-listing-of-tutorial2>`. Feel free to expand it with the presentation of the light intensity for the found mode at the threshold.

.. topic:: Python script in file :file:`tutorial2.xpl`.

   .. _lis-listing-of-tutorial2:
   .. code-block:: python

      import scipy.optimize

      profile = plask.StepProfile(GEO.main, default=0.)
      profile[GEO.gain_region] = 500.

      def loss_on_gain(gain):
          profile[GEO.gain_region] = gain
          mode_number = efm.find_mode(981.)
          return efm.outLoss(mode_number)

      threshold_gain = scipy.optimize.fsolve(loss_on_gain, 2000., xtol=0.1)[0]

      profile[GEO.gain_region] = threshold_gain
      mode_number = efm.find_mode(981.)
      mode_wavelength = efm.outWavelength(mode_number)
      print_log(LOG_INFO,
          "Threshold material gain is {:.0f}/cm with resonant wavelength {:.2f}nm"
          .format(threshold_gain, mode_wavelength))

.. rubric:: Example files

You can download the complete file from this tutorial: :download:`tutorial2.xpl <tutorial2.xpl>`.

.. rubric:: Footnotes
.. [#material-parameters] In this tutorial, we perform only optical analysis, so the refractive index and absorption is all we need. Other parameters can be unspecified as they are never requested by optical solvers.
.. [#module-scipy-optimize] In Python modules are some external libraries that extend its functionality. The ``sys`` module give access to many system function and objects.
.. [#object-names] ``GEO.gain_region`` is an alternative way to access named geometry objects. It is equivalent to ``GEO["gain-region"]``. Such an attribute access is often shorter, however when you use it, you must replace any hyphens in the name (``-``) with the underscore (``_``).


.. |list-add| image:: list-add.png
   :align: middle
   :alt: +

