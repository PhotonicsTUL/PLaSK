.. _sec-full-threshold-analysis-of-VCSEL:

Finding the threshold current of a gallium-arsenide VCSEL
---------------------------------------------------------

In this tutorial we will perform a full thermo-electro-optical modeling of a VCSEL structure to find it's threshold current.

Analyzed structure
^^^^^^^^^^^^^^^^^^

We will run our simulations on a structure similar to the one introduced in the previous tutorial. We will have to expand it's geometry by adding a heat sink with a temperature boundary condition, contacts with voltage boundary conditions and specify doping levels in the structure. For simplicity, the VCSEL will remain a single mesa structure, however, we will specify its radius as a variable, so we can easily change its value later. To to this, switch to the *Defines* tab. You see a simple table. Similarly to other sections, you can add, edit, and remove its rows with clicking the appropriate icons. To define a new variable, add a new row to this table, and put ``mesa`` in the *Name* column and ``10.`` in the *Value*. Next, add another row defining a variable named ``aperture`` with a value ``{mesa-6.}`` (in curly braces). This way, you have defined two variables that can be used layer in the file. The values defined in the *Defines* section have to be a value of a basic type (i.e. a float ``10.`` or a string: ``'Al(0.9)GaAs'``) or an expression that evaluates to such a value. Then, any parameter in any text box in the rest of the XPL file can be an expression, in which case it has to be enclosed in ``{}`` (assigning a single predefined value is treated as an expression, therefore it has to include ``{}`` as well). This way we can make the oxide aperture (``aperture`` parameter) dependent on the ``mesa`` value (however, we must define these parameters in the proper order).

Having ``mesa`` and ``aperture`` parameters defined, we can move on to the geometry definition. Switch to the *geometry* tab and press ``F4`` to change to the XML edit mode. We start with a 4 µm golden ring contact at the top of the structure. its inner radius is 1 µm smaller than the mesa edge:

.. code-block:: xml

   <cylindrical2d name="GeoTE" axes="r,z">
     <stack>
       <shelf>
         <gap total="{mesa-1}"/>
         <block name="n-contact" material="Au" dr="4" dz ="0.0500"/>
       </shelf>
     </stack>
   </cylindrical2d>

As you see, we use the ``shelf`` geometry container. In this container, it is possible to make a horizontal air gap by using the the ``<gap>`` tag inside a ``<shelf>``. It has to contain either ``size``, or ``total`` attribute. The difference between these attributes is that the ``size`` will place a gap of the size specified as the attribute value, while the ``total`` will automatically calculate the gap size, in such way that the whole shelf have the total size specified as the attribute value. Since we know the contact's distance from the mesa edge, it is more convenient to use the ``total`` attribute, as ``size={mesa-5}`` would stop being valid after changing the contact's width. You can switch to the graphical editor, to see how this parameters are defined in the GUI.

Next, in the XML view, we can define the whole VCSEL structure:

.. code-block:: xml

   <cylindrical2d name="GeoTE" axes="r,z">
     <stack>
       <shelf>
         <gap total="{mesa-1}"/>
         <rectangle name="n-contact" material="Au" dr="4" dz="0.0500"/>
       </shelf>
       <stack name="VCSEL">
         <rectangle material="GaAs:Si=2e+18" dr="{mesa}" dz="0.0700"/>
         <stack name="top-DBR" repeat="24">
           <rectangle material="Al(0.73)GaAs:Si=2e+18"
                                dr="{mesa}" dz="0.0795"/>
           <rectangle material="GaAs:Si=2e+18" dr="{mesa}" dz="0.0700"/>
         </stack>
         <shelf>
           <rectangle name="aperture" material="AlAs:Si=2e+18"
                      dr="{aperture}" dz="0.0160"/>
           <rectangle name="oxide" material="AlOx"
                      dr="{mesa-aperture}" dz="0.0160"/>
         </shelf>
         <rectangle material="Al(0.73)GaAs:Si=2e+18"
                    dr="{mesa}" dz="0.0635"/>
         <rectangle material="GaAs:Si=5e+17" dr="{mesa}" dz="0.1160"/>
         <stack name="junction" role="active">
           <stack repeat="4">
             <rectangle name="QW" role="QW" material="InGaAsQW"
                        dr="{mesa}" dz="0.0050"/>
             <rectangle material="GaAs" dr="{mesa}" dz="0.0050"/>
           </stack>
           <again ref="QW"/>
         </stack>
         <rectangle material="GaAs:C=5e+17" dr="{mesa}" dz="0.1160"/>
         <stack name="bottom-DBR" repeat="30">
           <rectangle material="Al(0.73)GaAs:C=2e+18"
                      dr="{mesa}" dz="0.0795"/>
           <rectangle material="GaAs:C=2e+18" dr="{mesa}" dz="0.0700"/>
         </stack>
       </stack>
       <zero/>
       <rectangle material="GaAs:C=2e+18" dr="200." dz="150."/>
       <rectangle name="p-contact" material="Cu" dr="2500." dz="5000."/>
     </stack>
   </cylindrical2d>

In the above definition we have several named geometry objects (mind that the name *‘aperture’* has nothing to do with the ``{aperture}`` parameter). The whole laser is contained it in a stack named *‘VCSEL’*, so we can reuse it when creating a geometry for optical calculations. You probably have noticed that the active region is different than the one defined in the previous tutorial: it has been replaced with a more realistic multiple-quantum-well structure. Also the spacer thicknesses were adjusted to maintain the proper resonator length. Mind that it is possible to assign a ``role`` to a whole container, like we just did with the *junction* stack. The ``role="active"`` tells the :class:`~plask.electrical.fem.ShockleyCyl` electrical solver to consider the current flow through it using the Shockley equation. The ``role="QW"`` is used to mark quantum-well layers and it is important for the diffusion and gain solvers.

We have also added a substrate with the copper heatsink on the bottom side. The heatsink will be also used as the bottom contact. Above the substrate we have included the ``<zero/>`` tag, to keep the thermo-electrical ``GeoTE`` geometry coordinates compatible with an optical geometry ``GeoO``, which we are going to add now: Move the cursor the the end of the last tag (``</cylindrical2d>``) and add the following three lines

.. code-block:: xml

   <cylindrical2d axes="rz" name="GeoO" top="air"
                  bottom="GaAs" outer="extend">
     <again ref="VCSEL"/>
   </cylindrical2d>

This way you have created a new geometry, which will be used by the optical solver. The new tag ``<again>`` tells PLaSK to simply repeat the whole stack named ``VCSEL``. This geometry has it's own coordinates with the origin located at it's bottom left corner, therefore adding ``<zero/>`` at a corresponding location in the ``GeoTE`` makes these geometries compatible. As you can see this geometry does not include neither substrate nor the heatsink. This is a proper approach for optical calculations as the optical mode is located far from them and such simpler geometry makes it much easier to find the mode (In fact we do include the substrate into this geometry by specifying ``bottom="GaAs"``).

In our VCSEL definition quantum wells are made of In\ :sub:`0.22`\ GaAs. Since we want to define our own recombination parameters required by the gain solver for it (that might have been e.g. estimated from an experiment), we have to define a custom material for quantum wells in the *Materials* section:

.. code-block:: xml

   <material name="InGaAsQW" base="In(0.22)GaAs">
     <nr>3.621</nr>
     <absp>0</absp>
     <A>110000000</A>
     <B>7e-011-1.08e-12*(T-300)</B>
     <C>1e-029+1.4764e-33*(T-300)</C>
     <D>10+0.01667*(T-300)</D>
   </material>

Having the geometry completed, we need to define meshes for all the solvers we are going to use. Switch to the *Meshing* section and type:

.. code-block:: xml

   <generator type="rectangular2d" method="divide" name="default">
     <postdiv by0="3" by1="2"/>
   </generator>

   <mesh type="regular" name="diffusion">
     <axis start="0" stop="{mesa}" num="2000"/>
   </mesh>

   <generator type="rectangular2d" method="divide" name="optical">
     <prediv by0="10" by1="3"/>
   </generator>

   <generator type="rectangular2d" method="divide" name="plots">
     <postdiv by="30"/>
   </generator>

The first ``default`` mesh will be used by the thermal and electrical solvers. The ``diffusion`` mesh of type ``regular`` is an one-dimensional mesh of evenly spaced ``num``-ber of points between the values provided in ``start`` and ``stop`` attributes. Such a mesh is required by the diffusion solver, which can automatically automatically detect the vertical positions of quantum wells (marked with ``role="QW"`` in our case). The last mesh generated by the generator ``optical`` will be used by the optical solver. Effective frequency algorithm is relatively fast and not used as frequently as thermal and electrical solvers, so we can make it horizontally dense for higher accuracy.

It's now time to define the solvers. Switch to the *Solvers* tab and type:

.. code-block:: xml

   <thermal solver="StaticCyl" name="THERMAL">
     <geometry ref="GeoTE"/>
     <mesh ref="default"/>
     <temperature>
       <condition value="300." place="bottom"/>
     </temperature>
   </thermal>

   <electrical solver="ShockleyCyl" name="ELECTRICAL">
     <geometry ref="GeoTE"/>
     <mesh ref="default"/>
     <junction js="1" beta="11"/>
     <voltage>
       <condition value="2.0">
         <place object="p-contact" side="bottom"/>
       </condition>
       <condition value="0.0">
         <place object="n-contact" side="top"/>
       </condition>
     </voltage>
   </electrical>

   <electrical solver="DiffusionCyl" name="DIFFUSION">
     <geometry ref="GeoO"/>
     <mesh ref="diffusion"/>
     <config fem-method="parabolic" accuracy="0.005"/>
   </electrical>

   <gain solver="FreeCarrierCyl" name="GAIN">
     <geometry ref="GeoO"/>
     <config lifetime="0.5" matrix-elem="10"/>
   </gain>

   <optical solver="EffectiveFrequencyCyl" name="OPTICAL">
     <geometry ref="GeoO"/>
     <mesh ref="optical"/>
     <mode lam0="980."/>
   </optical>

``THERMAL`` and ``ELECTRICAL`` solvers are analogous to these used in :ref:`the first tutorial <sec-Thermo-electrical-modeling-of-simple-ee-laser>`, but designed for cylindrical symmetries (i.e. ``StaticCyl`` instead of ``Static2D``). The ``OPTICAL`` solver is similar to that from :ref:`the previous tutorial <sec-Optical-analysis-of-VCSEL>`, but here we also specify a mesh for it, so it does not perform calculations on it's default simplified mesh. It is important to note, that this solver is assigned to a different geometry than ``THERMAL`` and ``ELECTRICAL`` solvers. This is the geometry that we adjusted for optical simulations. ``DIFFUSION`` and ``GAIN`` could be assigned to either full, or optical geometry, but in the second case we limit the calculations range to the mesa radius (instead of calculating in the air outside the mesa for the range of the full geometry, which is the heatsink radius equal to 2500 microns), therefore saving some time and memory.

.. TODO: diffusion and gain description...

Having our solvers defined, we must define their mutual connections properly in the *Connects* section:

.. code-block:: xml

   <connect in="ELECTRICAL.inTemperature" out="THERMAL.outTemperature"/>
   <connect in="THERMAL.inHeat" out="ELECTRICAL.outHeat"/>

   <connect in="DIFFUSION.inTemperature" out="THERMAL.outTemperature"/>
   <connect in="DIFFUSION.inCurrentDensity"
            out="ELECTRICAL.outCurrentDensity"/>

   <connect in="GAIN.inTemperature" out="THERMAL.outTemperature"/>
   <connect in="GAIN.inCarriersConcentration"
            out="DIFFUSION.outCarriersConcentration"/>

   <connect in="OPTICAL.inTemperature" out="THERMAL.outTemperature"/>
   <connect in="OPTICAL.inGain" out="GAIN.outGain"/>

These are all the connects we need in our case. The first two are for achieving self-consistency in the thermo-electrical part. The final temperature distribution calculated by ``THERMAL`` solver will be then used by all other solvers. Additionally the ``DIFFUSION`` solver has to be provided with the current density distribution from ``ELECTRICAL`` solver, ``GAIN`` requires carriers concentration obtained in ``DIFFUSION`` to calculate the gain, which then has to be eventually connected to the ``OPTICAL`` solver.

---------------------------------------------------------------------------

All the examples here have been presented as fragments of the XML code. Every element in the XPL file can be defined this way. However, the only reason we have presented it as a XML code in this tutorial is much easier copy-pasting the whole contents of each section than retyping it manually in the text boxes. While creating your own geometries, feel free to use graphical tools provided by the GUI.

Manual refinements of the divide mesh generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We could now run our calculations. However, it is a good habit, to check the geometries for any design flaws and the grids for proper density. To do this, we write a simple script in the *Script* section that will just draw the ``GeoTE`` geometry and the ``default`` mesh with the boundary conditions:

.. code-block:: python

   plot_geometry(GEO.GeoTE, margin=0.01)
   defmesh = MSG.default(GEO.GeoTE.item)
   plot_mesh(defmesh, color="0.75")
   plot_boundary(ELECTRICAL.voltage_boundary, defmesh,
                 ELECTRICAL.geometry, color="b", marker="D")
   plot_boundary(THERMAL.temperature_boundary, defmesh,
                 THERMAL.geometry, color="r")
   gcf().canvas.set_window_title("Default mesh")

   show()

Now, save the XPL and execute it. You can now see, that the default mesh is rather sparse. It could be improved by increasing the post-refining divisions of every element in the geometry (the values in the ``<postdiv by0="3" by1="2"/>`` line), but this would either end up with a mesh that is still too sparse at important locations or overally too dense and numerical ineffective. PLaSK allows for a better approach: manual addition of refinements at a desired location in a desired dimension. Let's modify our ``default`` mesh generator by adding a vertical refinement at the very bottom of the heatsink, where the temperature boundary condition is located. We should also add three horizontal refinements: two at the inner part of the oxidation, where strong current crowding is expected, and one near the optical axis of the laser. This can be done using the *Refinements* table in the generator configuration or modifying the XML code for this mesh to look like below (note that only the code for the default mesh generator is shown — do not remove other meshes nor generators):

.. code-block:: xml

   <generator type="rectilinear2d" method="divide" name="default">
     <postdiv by0="3" by1="2"/>
     <refinements>
       <axis1 object="p-contact" at="50"/>
       <axis0 object="oxide" at="-0.1"/>
       <axis0 object="oxide" at="-0.05"/>
       <axis0 object="aperture" at="0.1"/>
     </refinements>
     <warnings outside="no"/>
   </generator>

The refinements have to be included within the ``<refinements>`` element and are described with the ``axis#`` tag, where ``#`` means the axis number (0 for horizontal and 1 for vertical; in our case *r* and *z*, respectively). The ``at`` attribute places a single refinement line at the location provided in the ``at`` attribute along the requested direction in the local coordinates of an object specified in the ``object`` attribute. So the first refinement will add a single refinement line 50 microns in the *z* direction above the bottom of the *‘p-contact’* (heatsink), while the next two will place two horizontal refinements to the left of the *‘oxide’* object's left edge. The last two refinements are defined outside the object they are referred to, which would result in a warning-message when executing the file if it were not suppressed with ``<warnings outside="no"/>`` directive. We defined these refinements this way on purpose, because this notation is simpler than referring to the *aperture* object and using expressions with predefined values (``<axis0 object="aperture" at="{aperture-0.1}"/>``) and we are sure that these refinements are still within our geometry. Therefore we can ignore corresponding warnings, however it is always important to check the warning-messages, as they may point to a serious flaw in our code, especially when lots of predefined variables and/or real-time geometry changes are involved.

Instead of the ``at`` attribute, it is also possible to use either ``by``, or ``every`` attribute. ``by`` results in dividing the specified objects into provided number of elements, while ``every`` places refinement lines spaced equally with a distance specified as this attribute value. We must remember that adding a single refinement line does not actually result in just one additional line in the final mesh, as the generator automatically ensures that the distance between adjacent grid lines never change too rapidly.

You can see the new mesh by executing the file again.

Threshold current calculations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With having the geometries and meshes prepared, we can move on to scripting the real calculations. Similarly to the :ref:`previous tutorial <sec-Optical-analysis-of-VCSEL>`, we could manually write a function that computes the modal loss for specified voltage and use Python ``fsolve`` subroutine to find the threshold (at threshold the modal loss is zero). However, much simpler approach would be to use PLaSK high-level :ref:`algorithm <sec-algorithms>` to do this. So, before the last ``show()`` command in your script, type:

.. code-block:: python

   task = algorithm.ThresholdSearch(THERMAL, ELECTRICAL, DIFFUSION,
                                    GAIN, OPTICAL, 0, 1.5, 980.5)
   threshold_voltage = task.run()
   threshold_current = task.threshold_current
   print("Vth = {:.3f}V, Ith = {:.3f}mA"
      .format(threshold_voltage, threshold_current))

   figure()
   task.plot_optical_field()
   axvline(x=GEO.aperture.dr, color='0.75', ls=":", linewidth=1, alpha=0.35)

:py:class:`~plask.algorithm.ThresholdSearch` is a PLaSK algorithm class that will do the threshold search loop for you. It needs to be initialized with configured solvers (thermal, electrical, diffusion, gain, optical).
The other constructor parameters are: ``0``\ : the index of the voltage boundary condition to vary while changing the voltage, ``1.5``\ : the initial voltage to start from, ``980.5``\ : the initial wavelength to start looking for the optical mode.

After initializing the algorithm you simply run it using the method :py:meth:`~plask.algorithm.ThresholdSearch.run()`. If succeeded, it will return the threshold voltage. Furthermore, it saves the results of thermal and electrical calculations to disk in a HDF5 file, which can be later loaded and reused (this is, however, out of scope of this tutorial). The following lines in the above script, print the threshold voltage and current and and plot the found optical mode at the threshold. In the last line, we add a vertical bar to the plot to mark the oprical aperture.


Real-time structure modifications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It might be often important to perform an analysis of structure geometry parameters (like electrical and oxide apertures, resonator length etc.) influence on the output characteristics. For this we don't need to create several ``xpl`` files, or change the geometry description in a single ``xpl`` file every time, as we can operate on objects named in the geometry section from within the script. With this approach we could write an algorithm, that finds the oxide aperture radius, for which the threshold current has the minimum value. However, we will just show the idea by modifying the aperture once and running the threshold calculations again, as the full analysis would be unnecessarily complicated and calculations too time consuming for this tutorial purposes.

To do this, at the end of the file we have to add lines modifying the size of the ``oxide`` and ``aperture`` blocks defined in the ``xpl`` file. We want to change their lengths in the *r* direction (widths). This can be done by changing the ``dr``, or ``width`` parameter of these blocks:

.. code-block:: python

   new_aperture = 3.
   GEO.aperture.dr = new_aperture
   GEO.oxide.dr = mesa - new_aperture

Note that we have used the variable *mesa* that we have defined in the *Defines* section. Every value specified there is available as Python variable in the script.

Now we just have to repeat the calculations with the drawing part and to move the ``show()`` line to the end of the file:

.. code-block:: python

   threshold_voltage = task.run()
   threshold_current = task.threshold_current
   print("New aperture: Vth = {:.3f}V, Ith = {:.3f}mA"
       .format(threshold_voltage, threshold_current))

   figure()
   task.plot_optical_field()
   axvline(x=GEO.aperture.dr, color='0.75', ls=":",linewidth=1)
   gcf().canvas.set_window_title("Light Intensity (new aperture)")


   show()

.. rubric:: Example files

You can download the complete file from this tutorial: :download:`tutorial3.xpl <tutorial3.xpl>`.
