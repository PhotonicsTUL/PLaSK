.. _sec-full-threshold-analysis-of-VCSEL:

Finding the threshold current of a gallium-arsenide VCSEL
---------------------------------------------------------

In this tutorial we will perform a full thermo-electro-optical modeling of a VCSEL structure to find it's threshold current.

Analyzed structure
^^^^^^^^^^^^^^^^^^

We will run our simulations on a structure similar to the one introduced in previous tutorial. We will have to expand it's geometry by adding a heatsink with a temperature boundary condition, contacts with voltage boundary conditions and specify doping levels in the structure. For simplicity, the VCSEL will remain a single mesa structure, however, we will specify its radius in the ``<defines>`` section of the XPL file under a name ``mesa`` to refer to it later:

.. code-block:: xml

   <plask>

   <defines>
     <define name="mesa" value="10."/>
     <define name="aperture" value="{mesa-6.}"/>
   </defines>

   <materials>
   </materials>

The values defined in the ``<defines>`` section have to be a value of a basic type (f.e. a float ``value="10."`` or a string: ``value="'Al(0.9)GaAs'"``), or an expression that returns a value. Then, any value assignment in the XML part of the XPL file can be an expression, in such case it has to be enclosed in ``{}`` (assigning a single predefined value is treated as an expression, therefore it has to include ``{}`` as well). This way we can make the oxide aperture (``aperture`` parameter) dependent on the ``mesa`` value (however, in such case, we must define these parameters in the proper order).

Having ``mesa`` and ``aperture`` parameters defined, we can move on to the geometry definition (for now we can skip the ``<materials>`` section). We start with a 4 µm golden ring contact at the top of the structure. its inner radius is 1 µm smaller than the mesa edge:

.. code-block:: xml

   <geometry>
     <cylindrical2d axes="rz" name="GeoTE">
       <stack>
         <shelf>
           <gap total="{mesa-1}"/>
           <block dr="4" dz ="0.0500" material="Au" name="n-contact"/>
         </shelf>

As you see, we use the ``shelf`` geometry container. In this container, it is possible to make a horizontal air gap by using the the ``<gap>`` tag inside a ``<shelf>``. It has to contain either ``size``, or ``total`` attribute. The difference between these attributes is that the ``size`` will place a gap of the size specified as the attribute value, while the ``total`` will automatically calculate the gap size, in such way that the whole shelf have the total size specified as the attribute value. Since we know the contact's distance from the mesa edge, it is more convenient to use the ``total`` attribute, as ``size={mesa-5}`` would stop being valid after changing the contact's width.

Next we can define a VCSEL structure:

.. code-block:: xml

   <stack name="VCSEL">
     <stack name="top-DBR" repeat="24">
       <block dr="{mesa}" dz="0.07003" material="GaAs:Si=2e+18"/>
       <block dr="{mesa}" dz="0.07945" material="Al(0.73)GaAs:Si=2e+18"/>
     </stack>
     <block dr="{mesa}" dz="0.07003" material="GaAs:Si=2e+18"/>
     <block dr="{mesa}" dz="0.03178" material="Al(0.73)GaAs:Si=2e+18"/>
     <shelf>
       <block dr="{aperture}" dz="0.01603" material="AlAs:Si=2e+18"
                  name="aperture"/>
       <block dr="{mesa-aperture}" dz="0.01603" material="AlxOy"
                 name="oxide"/>
     </shelf>
     <block dr="{mesa}" dz="0.03178" material="Al(0.73)GaAs:Si=2e+18"/>
     <block dr="{mesa}" dz="0.13756" material="GaAs:Si=5e+17"/>
     <block dr="{mesa}" dz="0.11756" material="GaAs:Si=5e+17"/>
     <stack role="active" name="junction">
       <block dr="{mesa}" dz="0.005" material="InGaAsQW" role="QW"/>
       <stack repeat="4">
         <block dr="{mesa}" dz="0.005" material="GaAs"/>
         <block dr="{mesa}" dz="0.005" material="InGaAsQW" role="QW"/>
       </stack>
     </stack>
     <block dr="{mesa}" dz="0.11756" material="GaAs:C=5e+17"/>
     <stack name="bottom-DBR" repeat="29">
       <block dr="{mesa}" dz="0.07945" material="Al(0.73)GaAs:C=2e+18"/>
       <block dr="{mesa}" dz="0.07003" material="GaAs:C=2e+18"/>
     </stack>
     <block dr="{mesa}" dz="0.07945" material="Al(0.73)GaAs:C=2e+18"/>
   </stack>

In the above definition we have several named geometry objects (mind that the name *‘aperture’* has nothing to do with the ``{aperture}`` parameter). The whole laser is contained it in a stack named *‘VCSEL’*, so we can reuse it when creating a geometry for optical calculations. You probably have noticed that the active region is different than the one defined in the previous tutorial: it has been replaced with a more realistic multiple-quantum-well structure. Also the spacer thicknesses were adjusted to maintain the proper resonator length. Mind that it is possible to assign a ``role`` to a whole container, like we just did with the *junction* stack. The ``role="active"`` tells the :class:`~plask.electrical.fem.ShockleyCyl` electrical solver to consider the current flow through it using the Shockley equation. The ``role="QW"`` is used to mark a quantum-well layer and it is important for the diffusion and gain solvers.

Since the material database for InGaAs does not contain recombination parameters required by the gain solver, we have to define a custom material for quantum wells in the ``<materials>`` section:

.. code-block:: xml

   <material name="InGaAsQW" base="In(0.22)GaAs">
     <nr>3.621</nr>
     <absp>0</absp>
     <A>110000000</A>
     <B>7e-011-1.08e-12*(T-300)</B>
     <C>1e-029+1.4764e-33*(T-300)</C>
     <D>10+0.01667*(T-300)</D>
   </material>

Finally, we have to add a substrate with the copper heatsink on the bottom side. The heatsink will be also used as the bottom contact:

.. code-block:: xml

         <zero/>
         <block dr="200." dz="150." material="GaAs:C=2e+18"/>
         <block dr="2500." dz="5000." material="Cu" name="p-contact"/>
       </stack>
     </cylindrical2d>

We included the ``<zero/>`` tag, to keep the thermo-electrical ``GeoTE`` geometry coordinates compatible with an optical geometry ``GeoO``, which we are going to add now:

.. code-block:: xml

     <cylindrical2d axes="rz" name="GeoO" top="air" bottom="GaAs" outer="extend">
       <again ref="VCSEL"/>
     </cylindrical2d>

   </geometry>

The new tag ``<again>`` tells PLaSK to simply repeat the whole stack named ``VCSEL``. This geometry has it's own coordinates with the origin located at it's bottom left corner, therefore adding ``<zero/>`` at a corresponding location in the ``GeoTE`` makes these geometries compatible. As you can see this geometry does not include neither substrate nor the heatsink. This is a proper approach for optical calculations as the optical mode is located far from them and such simpler geometry makes it much easier to find the mode (In fact we do include the substrate into this geometry by specifying ``bottom="GaAs"``).

Having the geometry completed, we need to define meshes for all the solvers we are going to use:

.. code-block:: xml

   <grids>

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
   </grids>

The first ``default`` mesh will be used by the thermal and electrical solvers. The ``diffusion`` mesh of type ``regular`` is an one-dimensional mesh of evenly spaced ``num``-ber of points between the values provided in ``start`` and ``stop`` attributes. Such a mesh is required by the diffusion solver, which can automatically automatically detect the vertical positions of quantum wells (marked with ``role="QW"`` in our case). The last mesh generated by the generator ``optical`` will be used by the optical solver. Effective frequency algorithm is relatively fast and not used as frequently as thermal and electrical solvers, so we can make it horizontally dense for higher accuracy.

It's now time to define the solvers:

.. code-block:: xml

   <solvers>

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

     <gain solver="FermiCyl" name="GAIN">
       <geometry ref="GeoO"/>
       <config lifetime="0.5" matrix-elem="8"/>
     </gain>

     <optical solver="EffectiveFrequencyCyl" name="OPTICAL">
       <geometry ref="GeoO"/>
       <mesh ref="optical"/>
     </optical>

   </solvers>

``THERMAL`` and ``ELECTRICAL`` solvers are analogous to these used in :ref:`the first tutorial <sec-Thermo-electrical-modeling-of-simple-ee-laser>`, but designed for cylindrical symmetries (i.e. ``StaticCyl`` instead of ``Static2D``). The ``OPTICAL`` solver is similar to that from :ref:`the previous tutorial <sec-Optical-analysis-of-VCSEL>`, but here we also specify a mesh for it, so it does not perform calculations on it's default simplified mesh. It is important to note, that this solver is assigned to a different geometry than ``THERMAL`` and ``ELECTRICAL`` solvers. This is the geometry that we adjusted for optical simulations. ``DIFFUSION`` and ``GAIN`` could be assigned to either full, or optical geometry, but in the second case we limit the calculations range to the mesa radius (instead of calculating in the air outside the mesa for the range of the full geometry, which is the heatsink radius equal to 2500 microns), therefore saving some time and memory.

==>  TODO: diffusion and gain description...

Having our solvers defined, we must define their mutual connections properly:

.. code-block:: xml

   <connects>
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
   </connects>

These are all the connects we need in our case. The first two are for achieving self-consistency in the thermo-electrical part. The final temperature distribution calculated by ``THERMAL`` solver will be then used by all other solvers. Additionally the ``DIFFUSION`` solver has to be provided with the current density distribution from ``ELECTRICAL`` solver, ``GAIN`` requires carriers concentration obtained in ``DIFFUSION`` to calculate the gain, which then has to be eventually connected to the ``OPTICAL`` solver.

Manual refinements of the divide mesh generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We could now run our calculations. However, it is a good habit, to check the geometries for any design flaws and the grids for proper density. To do this, we write a simple script (remember to include it within ``<script><![CDATA[`` and ``]]></script>`` tags), that will just draw the ``GeoTE`` geometry and the ``default`` mesh with the boundary conditions:

.. code-block:: python

   figure()
   plot_geometry(GEO.GeoTE, set_limits=True)
   gcf().canvas.set_window_title("GEO TE")

   figure()
   plot_geometry(GEO.GeoTE, set_limits=True)
   defmesh = MSG.default(GEO.GeoTE.item)
   plot_mesh(defmesh, color="0.75")
   plot_boundary(ELECTRICAL.voltage_boundary, defmesh,
                 ELECTRICAL.geometry, color="b", marker="D")
   plot_boundary(THERMAL.temperature_boundary, defmesh,
                 THERMAL.geometry, color="r")
   gcf().canvas.set_window_title("Default mesh")

   show()

Now, close the XPL file with the ``</plask>`` tag and execute it. You can now see, that the lattice is rather sparse. It could be improved by increasing the values in the ``<postdiv by0="3" by1="2"/>`` line (that corresponds to horizontal and vertical divisions of every element in the geometry), but this would either end up with a mesh that is still too sparse at important locations or overally too dense and numerical ineffective. PLaSK allows for a better approach: manual addition of refinements at a desired location in a desired dimension. Let's modify our ``default`` mesh generator by adding a vertical refinement at the very bottom of the heatsink, where the temperature boundary condition is located. We should also add three horizontal refinements: two at the inner part of the oxidation, where strong current crowding is expected, and one near the optical axis of the laser:

.. code-block:: xml

  <grids>

    <generator type="rectilinear2d" method="divide" name="default">
      <postdiv by0="3" by1="2"/>
      <refinements>
        <axis1 object="p-contact" at="50"/>
        <axis0 object="oxide" at="-0.1"/>
        <axis0 object="oxide" at="-0.05"/>
        <axis0 object="aperture" at="0.1"/>
      </refinements>
    </generator>

The refinements have to be included within the ``<refinements>`` element and are described with the ``axis#`` tag, where ``#`` means the axis number (0 for horizontal and 1 for vertical; in our case *r* and *z*, respectively). The ``at`` attribute places a single refinement line at the location provided in the ``at`` attribute along the requested direction in the local coordinates of an object specified in the ``object`` attribute. So the first refinement will add a single refinement line 50 microns in the *z* direction above the bottom of the *‘p-contact’* (heatsink), while the next two will place two horizontal refinements to the left of the *‘oxide’* object's left edge. The last two refinements are defined outside the object they are referred to, which will result in a warning-message when executing the file. We defined these refinements this way on purpose, because this notation is simpler than referring to the *aperture* object and using expressions with predefined values (``<axis0 object="aperture" at="{aperture-0.1}"/>``) and we are sure that these refinements are still within our geometry. Therefore we can ignore corresponding warnings, however it is always important to check the warning-messages, as they may point to a serious flaw in our code, especially when lots of predefined variables and/or real-time geometry changes are involved [#disable-warning]_.

Instead of the ``at`` attribute, it is also possible to use either ``by``, or ``every`` attribute. ``by`` results in dividing the specified objects into provided number of elements, while ``every`` places refinement lines spaced equally with a distance specified as this attribute value. We must remember that adding a single refinement line does not actually result in just one additional line in the final mesh, as the generator automatically ensures that the distance between adjacent grid lines never change too rapidly.

You can see the new mesh by executing the file again.

Threshold current calculations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With having the geometries and meshes prepared, we can move on to the script part. Like in the previous tutorial, we are going to create a separate ``tutorial3.py`` file for the scripting purpose and begin it with:

.. code-block:: python

   import sys
   import scipy.optimize

   filename = sys.argv[1]
   loadxpl(filename)

Then we can move directly to defining a function for the Brent root-finding algorithm, but this time, it will take the voltage applied to the structure as it's argument and return the modal loss:

.. code-block:: python

   def loss_on_voltage(voltage):
       ELECTRICAL.invalidate()
       ELECTRICAL.voltage_boundary[0].value = voltage
       verr = ELECTRICAL.compute(1)
       terr = THERMAL.compute(1)
       iters=0
       while (terr > THERMAL.maxerr or verr > ELECTRICAL.maxerr) and iters<15:
          verr = ELECTRICAL.compute(8)
          terr = THERMAL.compute(1)
          iters+=1
       DIFFUSION.compute_threshold()
       det_lams = linspace(OPTICAL.lam0-2, OPTICAL.lam0+2, 401)+0.2j*(voltage-0.5)/1.5
       det_vals = abs(OPTICAL.get_determinant(det_lams, m=0))
       det_mins = np.r_[False, det_vals[1:] < det_vals[:-1]] & \
                  np.r_[det_vals[:-1] < det_vals[1:], False] & \
                  np.r_[det_vals[:] < 1]
       mode_number = OPTICAL.find_mode(max(det_lams[det_mins]))
       mode_loss = OPTICAL.outLoss(mode_number)
       print_log(LOG_RESULT,
	    'V = {:.3f}V, I = {:.3f}mA, lam = {:.2f}nm, loss = {}/cm'
	    .format(voltage, ELECTRICAL.get_total_current(),
                    OPTICAL.outWavelength(mode_number), mode_loss))
       return mode_loss

In the first line we invalidate the ``ELECTRICAL`` solver, as after every calculations it stores the resulting junction conductivity and uses it as initial value for future calculations. This approach saves time when consequent calculations are being made for small variations of voltage. However, for cases where broad range of applied voltages is analysed, especially including values producing conductivities close to zero, these stored values can be rather a problem than the merit. Hence, it is better to reset the electrical solver by using it's ``invalidate()``.

After achieving the steady-state thermo-electrical solution, we calculate the carriers concentration in the quantum wells via ``DIFFUSION.compute_threshold()`` function.

Finally, we have to introduce a reliable algorithm that will find the fundamental mode for any given voltage. The optical solver has to solve a two dimensional (complex numbers) problem, which is sensitive to the structure's condition (voltage, temperature, gain) and to the starting wavelength value. A good approach is to probe the determinant function used by the optical model on a wavelength range close to the expected value (in our case 401 points over a ±2 nm range from the ``OPTICAL.lam0`` value):

.. code-block:: python

   det_lams = linspace(OPTICAL.lam0-2, OPTICAL.lam0+2, 401)+0.2j*(voltage-0.5)/1.5
   det_vals = abs(OPTICAL.get_determinant(det_lams, m=0))

in order to find it's minima, we do some numpy* array manipulations:

.. code-block:: python

   det_mins = np.r_[False, det_vals[1:] < det_vals[:-1]] & \
              np.r_[det_vals[:-1] < det_vals[1:], False] & \
              np.r_[det_vals[:] < 1]

to finally use the minimum with highest wavelength as the starting point for the calculations:

.. code-block:: python

   mode_number = OPTICAL.find_mode(max(det_lams[det_mins]))

This approach may seem a little complicated at first. However, we must rememeber that we are altering current flowing through the structure, which in turn alters its termerature distribution and, thus, the refractive index. Hence, the simple approach ``mode_number = OPTICAL.find_mode(980.)`` may fail.

Generally, it is highly recommended to perform an analysis of the determinant function for every simulated structure and various driving conditions. As an example, we can add a determinant plotting block in our function as shown below (although it is advised to omit it in the final algorithm):

.. code-block:: python

   plot(det_lams, det_vals)
   gca().set_yscale('log')
   show()

Outside the function we set the reference wavelength ``lam0`` and tell the optical solver to look for the vertical solution at the optical axis of the laser by setting it's ``vat`` parameter to 0.
Finally we run our function to find the threshold voltage between 0.5V and 2.0V and print the result to the log:

.. code-block:: python

   OPTICAL.lam0 = 981.5
   OPTICAL.vat = 0

   threshold_voltage = scipy.optimize.brentq(loss_on_voltage, 0.5, 2., xtol=0.05)

   loss_on_voltage(threshold_voltage)
   threshold_current = abs(ELECTRICAL.get_total_current())
   print_log(LOG_WARNING, "Vth = {:.3f}V    Ith = {:.3f}mA"
                          .format(threshold_voltage, threshold_current))

We might want to visualise the found mode at the threshold. For this we have to define a mesh for the output field on the optical geometry, get the intensity field for the last calculated mode from the optical solver and plot it:

.. code-block:: python

   geometry_width = GEO.GeoO.bbox.upper[0]
   geometry_height = GEO.GeoO.bbox.upper[1]
   RR = linspace(-geometry_width, geometry_width, 200)
   ZZ = linspace(0, geometry_height, 500)
   intensity_mesh = mesh.Rectangular2D(RR, ZZ)

   IntensityField = OPTICAL.outLightMagnitude(len(OPTICAL.outWavelength)-1,
                                              intensity_mesh)
   figure()
   plot_field(IntensityField, 100)
   plot_geometry(GEO.GeoO, mirror=True, color="w")
   gcf().canvas.set_window_title('Light Intensity Field ({0} micron \
                                 aperture)'.format(GEO.aperture.dr))
   axvline(x=GEO.aperture.dr, color='w', ls=":", linewidth=1)
   axvline(x=-GEO.aperture.dr, color='w', ls=":", linewidth=1)
   xticks(append(xticks()[0], [-GEO.aperture.dr, GEO.aperture.dr]))
   xlabel(u"r [\xb5m]")
   ylabel(u"z [\xb5m]")
   show()


Real-time structure modifications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It might be often important to perform an analysis of structure geometry parameters (like electrical and oxide apertures, resonator length etc.) influence on the output characteristics. For this we don't need to create several ``xpl`` files, or change the geometry description in a single ``xpl`` file every time, as we can operate on objects named in the ``<geometry>`` section from within the script. With this approach we could write an algorithm, that finds the oxide aperture radius, for which the threshold current has the minimum value. However, we will just show the idea by modifying the aperture once and running the threshold calculations again, as the full analysis would be unnecessarily complicated and calculations too time consuming for this tutorial purposes.

To do this, at the end of the file we have to add lines modifying the size of the ``oxide`` and ``aperture`` blocks defined in the ``xpl`` file. We want to change their lengths in the *r* direction (widths). This can be done by changing the ``dr``, or ``width`` parameter of these blocks:

.. code-block:: python

   new_aperture = 3.
   GEO.aperture.dr = new_aperture
   GEO.oxide.dr = DEF["mesa"] - new_aperture

Here it is important to mention, that the axes configuration specified in the ``xpl`` file (``axes="rz"``) does not automatically apply to the ``py`` script file, where the default ``xy`` setting is use by default. To change it, we have to add a line ``config.axes = 'rz'``, preferably just below the ``loadxpl(filename)`` line in the beginning of the file.

.. code-block:: python

   loadxpl(filename)
   config.axes = 'rz'

Now we just have to repeat the calculations with the drawing part and to move the ``show()`` line to the end of the file:

.. code-block:: python

   OPTICAL.lam0 = 982.
   threshold_voltage = scipy.optimize.brentq(loss_on_voltage, 0.5, 2., xtol=0.05)
   loss_on_voltage(threshold_voltage)
   threshold_current = abs(ELECTRICAL.get_total_current())
   print_log(LOG_WARNING, "Vth = {:.3f}V    Ith = {:.3f}mA"
                          .format(threshold_voltage, threshold_current))

   IntensityField = OPTICAL.outLightMagnitude(len(OPTICAL.outWavelength)-1,
                                              intensity_mesh)
   figure()
   plot_field(IntensityField, 100)
   plot_geometry(GEO.GeoO, mirror=True, color="w")
   gcf().canvas.set_window_title('Light Intensity Field ({0} micron \
                                 aperture)'.format(GEO.aperture.dr))
   axvline(x=GEO.aperture.dr, color='w', ls=":", linewidth=1)
   axvline(x=-GEO.aperture.dr, color='w', ls=":", linewidth=1)
   xticks(append(xticks()[0], [-GEO.aperture.dr, GEO.aperture.dr]))
   xlabel(u"r [\xb5m]")
   ylabel(u"z [\xb5m]")

   show()

.. rubric:: Example files

You can download the complete files from this tutorial: :download:`tutorial3.xpl <tutorial3.xpl>`, :download:`tutorial3.py <tutorial3.py>`.

.. rubric:: Footnotes
.. [#disable-warning] It is possible to disable warning, for this please refer to the documentation of the generator :xml:tag:`<warnings>` tag.
