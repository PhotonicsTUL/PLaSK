.. _sec-full-threshold-analysis-of-VCSEL:

Finding the threshold current of a gallium-arsenide VCSEL
---------------------------------------------------------

In this tutorial we will perform a full thermo-electro-optical modeling of a VCSEL structure to find it's threshold current.

Analyzed structure
^^^^^^^^^^^^^^^^^^

We will run our simulations on a structure similar to the one introduced in the previous tutorial. We will have to expand it's geometry by adding a heat sink with a temperature boundary condition, contacts with voltage boundary conditions and specify doping levels in the structure. For simplicity, the VCSEL will remain a single mesa structure, however, we will specify its radius as a variable, so we can easily change its value later. To to this, switch to the *Defines* tab. You see a simple table. Similarly to other sections, you can add, edit, and remove its rows with clicking the appropriate icons. To define a new variable, add a new row to this table, and put ``mesa`` in the *Name* column and ``40.`` in the *Value*. Next, add another row defining a variable named ``aperture`` with a value ``8.``. This way, you have defined two variables that can be used layer in the file. The values defined in the *Defines* section have to be a value of a basic type (i.e. a float ``10.`` or a string: ``'Al(0.9)GaAs'``) or an expression that evaluates to such a value. Then, any parameter in any text box in the rest of the XPL file can be an expression, in which case it has to be enclosed in ``{}`` (assigning a single predefined value is treated as an expression, therefore it has to include ``{}`` as well). This way we can make the oxide aperture (``mesa`` parameter) dependent on the ``aperture`` value (however, we must define these parameters in the proper order).

Having ``mesa`` and ``aperture`` parameters defined, we can move on to the geometry definition. Switch to the *geometry* tab and press ``F4`` to change to the XML edit mode. We start with a 4 µm golden ring contact at the top of the structure. its inner radius is 1 µm smaller than the mesa edge:

.. code-block:: xml

   <cylindrical2d name="GeoE" axes="r,z">
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

   <cylindrical2d name="GeoE" axes="r,z">
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
       <rectangle name="p-contact" material="GaAs:C=2e+18" dr="{mesa/2}" dz="5."/>
     </stack>
   </cylindrical2d>

In the above definition we have several named geometry objects (mind that the name *‘aperture’* has nothing to do with the ``{aperture}`` parameter). The whole laser is contained it in a stack named *‘VCSEL’*, so we can reuse it when creating a geometry for optical calculations. You probably have noticed that the active region is different than the one defined in the previous tutorial: it has been replaced with a more realistic multiple-quantum-well structure. Also the spacer thicknesses were adjusted to maintain the proper resonator length. Mind that it is possible to assign a ``role`` to a whole container, like we just did with the *junction* stack. The ``role="active"`` tells the :class:`~plask.electrical.fem.ShockleyCyl` electrical solver to consider the current flow through it using the Shockley equation. The ``role="QW"`` is used to mark quantum-well layers and it is important for the diffusion and gain solvers.

We have also added a substrate on the bottom side. It will be also used as the bottom contact. Above the substrate we have included the ``<zero/>`` tag, to keep the thermo-electrical ``GeoE`` geometry coordinates compatible with a thermal geometry ``GeoT``, which we are going to add now: Move the cursor the the end of the last tag (``</cylindrical2d>``) and add the following lines:

.. code-block:: xml

   <cylindrical2d name="GeoT" axes="r,z">
     <stack>
       <item right="{mesa/2-1}">
         <rectangle material="Au" dr="4" dz="0.0500"/>
       </item>
       <again ref="VCSEL"/>
       <zero/>
       <rectangle material="GaAs:C=2e+18" dr="2500." dz="150."/>
       <rectangle material="Cu" dr="2500." dz="5000."/>
     </stack>
   </cylindrical2d>

In this geometry we have a thicker substrate and and a copper heatsink. The new tag ``<again>`` tells PLaSK to simply repeat the whole stack named ``VCSEL``. This geometry has it's own coordinates with the origin located at it's bottom left corner, therefore adding ``<zero/>`` at a corresponding location in the ``GeoE`` makes these geometries compatible. Mind also that we have added a top contact once again to match it with the electrical geometry. However, istead of using the shelf with a gap, we have simply specified horizontal alignment ot the object individually (by putting it in the ``<item>`` and specifying the right edge position).

We need to define another geometry for optical calculations:

.. code-block:: xml

   <cylindrical2d axes="rz" name="GeoO" top="air"
                  bottom="GaAs" outer="extend">
     <again ref="VCSEL"/>
   </cylindrical2d>

As you can see this geometry does not include neither substrate nor the heatsink. This is a proper approach for optical calculations as the optical mode is located far from them and such simpler geometry makes it much easier to find the mode (In fact we do include the substrate into this geometry by specifying ``bottom="GaAs"``).

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
     <prediv by0="10"/>
   </generator>

The first ``default`` mesh will be used by the thermal and electrical solvers. The ``diffusion`` mesh of type ``regular`` is an one-dimensional mesh of evenly spaced ``num``-ber of points between the values provided in ``start`` and ``stop`` attributes. Such a mesh is required by the diffusion solver, which can automatically automatically detect the vertical positions of quantum wells (marked with ``role="QW"`` in our case). The last mesh generated by the generator ``optical`` will be used by the optical solver. Effective frequency algorithm is relatively fast and not used as frequently as thermal and electrical solvers, so we can make it horizontally dense for higher accuracy.

It's now time to define the solvers. PLaSK offers so called meta-solvers, which automatically combine some other solvers and their connections. Switch to the *Solvers* tab and type:

.. code-block:: xml

   <meta name="SOLVER" solver="ThresholdSearchCyl" lib="shockley">
     <geometry electrical="GeoE" thermal="GeoT" optical="GeoO"/>
     <mesh electrical="default" thermal="default" diffusion="diffusion" optical="optical"/>
     <optical lam0="980.5"/>
     <voltage>
       <condition value="1.4">
         <place side="bottom" object="p-contact"/>
       </condition>
       <condition value="0.0">
         <place side="top" object="n-contact"/>
       </condition>
     </voltage>
     <temperature>
       <condition place="bottom" value="300."/>
     </temperature>
     <root bcond="0"/>
     <junction beta0="11" js0="1"/>
     <diffusion accuracy="0.005" fem-method="parabolic"/>
     <gain lifetime="0.5" matrix-elem="10"/>
   </meta>

The important parameters here are ``lam0`` in the ``<optical>`` tag, which tells that the optical computations will be looking for a mode near 980.5 nm, and ``bcond`` in the ``<root>`` tag informing the solver that we want to modify the value applied at the first voltage boundary condition (currently set to 1.4 V) during the threshold search.

---------------------------------------------------------------------------

All the examples here have been presented as fragments of the XML code. Every element in the XPL file can be defined this way. However, the only reason we have presented it as a XML code in this tutorial is much easier copy-pasting the whole contents of each section than retyping it manually in the text boxes. While creating your own geometries, feel free to use graphical tools provided by the GUI.

Manual refinements of the divide mesh generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We could now run our calculations. However, it is a good habit, to check the geometries for any design flaws and the grids for proper density. To do this, we write a simple script in the *Script* section that will just draw the ``GeoE`` geometry and the ``default`` mesh with the boundary conditions:

.. code-block:: python

   plot_geometry(GEO.GeoE, margin=0.01)
   defmesh = MSG.default(GEO.GeoE.item)
   plot_mesh(defmesh, color="0.75")
   plot_boundary(ELECTRICAL.voltage_boundary, defmesh,
                 ELECTRICAL.geometry, color="b", marker="D")
   window_title("Default mesh")

   show()

Now, save the XPL and execute it. You can now see, that the default mesh is rather sparse. It could be improved by increasing the post-refining divisions of every element in the geometry (the values in the ``<postdiv by0="3" by1="2"/>`` line), but this would either end up with a mesh that is still too sparse at important locations or overally too dense and numerical ineffective. PLaSK allows for a better approach: manual addition of refinements at a desired location in a desired dimension. Let's modify our ``default`` mesh generator by adding a vertical refinement at the very bottom of the heatsink, where the temperature boundary condition is located. We should also add three horizontal refinements: two at the inner part of the oxidation, where strong current crowding is expected, and one near the optical axis of the laser. This can be done using the *Refinements* table in the generator configuration or modifying the XML code for this mesh to look like below (note that only the code for the default mesh generator is shown — do not remove other meshes nor generators):

.. code-block:: xml

   <generator type="rectilinear2d" method="divide" name="default">
     <postdiv by0="3" by1="2"/>
     <refinements>
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

With having the geometries and meshes prepared, we can move on to scripting the real calculations. Meta-solvers provide all the logic automatically, so all you need to launch the computations is to type before the last ``show()`` command:

.. code-block:: python

   volts = 1.4, 1.6

   threshold_voltage = SOLVER.compute(volts)

The variable ``volts`` passed as a method argument defines the voltage range, in which we expect to find the threshold. This voltage will be applied on the boundary condition indicated in the solver configuration.

We need to add a little more code to see the actual results:

.. code-block:: python

   threshold_voltage = SOLVER.compute(volts)
   threshold_current = SOLVER.threshold_current
   print("Vth = {:.3f} V,  Ith = {:.3f} 4mA"
       .format(threshold_voltage, threshold_current))

   figure()
   SOLVER.plot_optical_field()
   axvline(GEO.aperture.dr, color='0.75', ls=":", linewidth=1)
   window_title("Light Intensity")

The solver method :py:meth:`~meta.shockley.ThresholdSearchCyl.compute()` will return the threshold voltage. Furthermore, it saves the results of thermal and electrical calculations to disk in a HDF5 file, which can be later loaded and reused (this is, however, out of scope of this tutorial). The following lines in the above script, print the threshold voltage and current and and plot the found optical mode at the threshold. In the last line, we add a vertical bar to the plot to mark the oprical aperture.


Real-time structure modifications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It might be often important to perform an analysis of structure geometry parameters (like electrical and oxide apertures, resonator length etc.) influence on the output characteristics. For this we don't need to create several ``xpl`` files, or change the geometry description in a single ``xpl`` file every time, as we can operate on objects named in the geometry section from within the script. With this approach we could write an algorithm, that finds the oxide aperture radius, for which the threshold current has the minimum value. However, we will just show the idea by modifying the aperture once and running the threshold calculations again, as the full analysis would be unnecessarily complicated and calculations too time consuming for this tutorial purposes.

To do this, at the end of the file we have to add lines modifying the size of the ``oxide`` and ``aperture`` blocks defined in the ``xpl`` file. We want to change their lengths in the *r* direction (widths). This can be done by changing the ``dr``, or ``width`` parameter of these blocks:

.. code-block:: python

   new_aperture = 6.
   GEO.aperture.dr = new_aperture / 2.
   GEO.oxide.dr = (mesa - new_aperture) / 2.

Note that we have used the variable *mesa* that we have defined in the *Defines* section. Every value specified there is available as Python variable in the script.

Now we just have to repeat the calculations with the drawing part and to move the ``show()`` line to the end of the file:

.. code-block:: python

   threshold_voltage = SOLVER.compute(volts)
   threshold_current = SOLVER.threshold_current
   print("New aperture:  Vth = {:.3f} V,  Ith = {:.3f} mA"
       .format(threshold_voltage, threshold_current))

   figure()
   SOLVER.plot_optical_field()
   axvline(GEO.aperture.dr, color='0.75', ls=":", linewidth=1)
   window_title("Light Intensity (new aperture)")

   show()

.. rubric:: Example files

You can download the complete file from this tutorial: :download:`tutorial3.xpl <tutorial3.xpl>`.
