.. _sec-full-threshold-analysis-of-VCSEL:

Finding the threshold current of a gallium-arsenide VCSEL
---------------------------------------------------------

In this tutorial we will perform a full thermo-electro-optical modelling of a VCSEL structure to find it's threshold current.

Analysed structure
^^^^^^^^^^^^^^^^^^

We will run our simulations on a structure similar to the one introduced in previous tutorial. We will have to expand it's geometry by adding a heatsink with a temperature boundary condition, contacts with voltage boundary conditions and specify doping levels in the structure. For simplicity, the VCSEL will remain a single mesa structure, however, we will specify it's radius in the ``<defines>`` section of the XPL file under a name ``mesaRadius`` to refer to it later:

.. code-block:: xml

 <plask>

 <defines>
  <define name="mesaRadius" value="10."/>
  <define name="apertureRadius" value="{mesaRadius-6.}"/>
 </defines>

 <materials>
 </materials>

The values defined in the ``<defines>`` section have to be a value of a basic type (f.e. a string: ``value="'Al(0.9)GaAs'"``), or an expression that returns a value. Any value assignment in the XML part of the XPL file can be an expression, in such case it has to be enclosed in ``{}`` (assigning a single predefined value is treated as an expression, therefore it has to include ``{}`` as well). This way we can make the oxide aperture dependant on ``mesaRadius`` and keep it's radius as the second defined value ``apertureRadius``. For now we can skip the ``<materials>`` section and move on to the geometry description and start with adding a 4 µm golden ring contact on top, that is 1 µm away from the mesa edge:

.. code-block:: xml

 <geometry>
  <cylindrical2d axes="rz" name="GeoTE">
   <stack>
    <shelf>
     <gap total="{mesaRadius-1}"/>
     <block dr="4" dz ="0.0500" material="Au" name="n-contact"/>
    </shelf>

It is possible to make a horizontal air gap by using the the ``<gap>`` tag inside a ``<shelf>``. It has to contain either ``size``, or ``total`` attribute. The difference between these attributes is that the ``size`` will place a gap of the size specified as the attribute value, while the ``total`` will calculate the gap size, so the whole shelf have the total size specified as the attribute value. Since we know the contact's distance from the mesa edge, it is more convenient to use the ``total`` attribute, as ``size={mesaRadius-5}`` would stop being valid after changing the contact's width.
Next we want to insert the mesa geometry updated with the predefined values and layer materials updated with dopant concentrations:

.. code-block:: xml

    <stack name="VCSEL">
     <stack name="top-DBR" repeat="24">
      <block dr="{mesaRadius}" dz="0.07003" material="GaAs:Si=2e+18"/>
      <block dr="{mesaRadius}" dz="0.07945" material="Al(0.73)GaAs:Si=2e+18"/>
     </stack>
     <block dr="{mesaRadius}" dz="0.07003" material="GaAs:Si=2e+18"/>
     <block dr="{mesaRadius}" dz="0.03178" material="Al(0.73)GaAs:Si=2e+18"/>
     <shelf>
      <block dr="{apertureRadius}" dz="0.01603" material="AlAs:Si=2e+18"
                                                     name="apertureIn"/>
      <block dr="{mesaRadius-apertureRadius}" dz="0.01603" material="AlxOy"
                                                      name="apertureOut"/>
     </shelf>
     <block dr="{mesaRadius}" dz="0.03178" material="Al(0.73)GaAs:Si=2e+18"/>
     <block dr="{mesaRadius}" dz="0.13756" material="GaAs:Si=5e+17"/>
     <block dr="{mesaRadius}" dz="0.005" material="GaAs" name="junction"
                                                         role="active"/>
     <block dr="{mesaRadius}" dz="0.13756" material="GaAs:C=5e+17"/>
     <stack name="bottom-DBR" repeat="29">
      <block dr="{mesaRadius}" dz="0.07945" material="Al(0.73)GaAs:C=2e+18"/>
      <block dr="{mesaRadius}" dz="0.07003" material="GaAs:C=2e+18"/>
     </stack>
     <block dr="{mesaRadius}" dz="0.07945" material="Al(0.73)GaAs:C=2e+18"/>
    </stack>


We contained it in a stack named "VCSEL" to reuse it when creating a geometry for optical calculations. You probably have noticed changes in the active region with regard to the previous tutorial - it got replaced with a more realistic multiple-quantum-well structure, also the spacer thicknesses were adjusted to maintain a proper resonator. Noteworthy is the fact, that it is possible to assign a ``role`` to a whole container, like we just did with the ``junction`` stack. The ``role="active"`` makes the contents of this stack unimportant to the electrical solver, as it threats it as the junction area described with the junction equation. The ``role="QW"`` is obviously marking a quantum-well layer and is important for the diffusion and gain solvers. Now we have to add a substrate with copper heatsink on the bottom side. The heatsink will be also used as the bottom contact:

.. code-block:: xml

    <zero/>
    <block dr="200." dz="150." material="GaAs:C=2e+18"/>
    <block dr="2500." dz="5000." material="Cu" name="p-contact"/>
   </stack>
  </cylindrical2d>

We included the ``<zero/>`` tag, to keep the thermo-electrical ``GeoTE`` geometry coordinates compatible with the optical geometry ``GeoO`` we are going to add now:

.. code-block:: xml

  <cylindrical2d axes="rz" name="GeoO" top="air" bottom="GaAs" outer="extend">
   <again ref="VCSEL"/>
  </cylindrical2d>

This geometry has it's own coordinates with [0,0] located at it's bottom left corner, therefore adding ``<zero/>`` at a corresponding location in the ``GeoTE`` makes these geometries compatible.
Now we have to define meshes for all the solvers we are going to use:

.. code-block:: xml

 <grids>
  <generator type="rectilinear2d" method="divide" name="default">
   <postdiv by0="2" by1="2"/>
  </generator>
  <mesh type="regular1d" name="meshDiff">
   <axis start="0" stop="{mesaRadius}" num="2000"/>
  </mesh>
  <generator type="rectilinear2d" method="divide" name="gridsOptical">
   <prediv by0="10" by1="3"/>
  </generator>
 </grids>

The first ``default`` mesh will be used by the thermal and electrical solvers. The ``meshDiff`` mesh of type ``regular1d`` is an one-dimensional mesh of evenly spaced ``num``-ber of points between the values provided in ``start`` and ``stop`` attributes. Diffusion solver automatically calculates for the active layers (marked with ``role="QW"`` in our case), therefore it only requires a 1D set of points for the horizontal direction. The last mesh generated will be used by the optical solver. Effective frequency algorithm is relatively fast and not used as frequently as thermal and electrical solvers, so we can make it horizontally dense for higher accuracy.
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
   <junction js="1" beta="18"/>
   <voltage>
    <condition value="2.0"><place object="p-contact" side="bottom"/></condition>
    <condition value="0.0"><place object="n-contact" side="top"/></condition>
   </voltage>
  </electrical>

  <electrical solver="DiffusionCyl" name="DIFFUSION">
   <geometry ref="GeoO"/>
   <mesh ref="meshDiff"/>
   <config fem-method="parabolic" accuracy="0.005"/>
  </electrical>

  <gain solver="FermiCyl" name="GAIN">
   <geometry ref="GeoO"/>
   <config lifetime="0.5" matrix-elem="8"/>
  </gain>

  <optical solver="EffectiveFrequencyCyl" name="OPTICAL">
   <geometry ref="GeoO"/>
   <mesh ref="gridsOptical"/>
  </optical>
 </solvers>

``THERMAL`` and ``ELECTRICAL`` solvers are analogous to these used in the first tutorial, but designed for cylindrical symmetries (f.e. ``StaticCyl`` instead of ``Static2D``). The ``OPTICAL`` solver is similar to that from previous tutorial, but here we also specify a mesh for it, so it does not perform calculations on it's default simplified mesh. It is important to note, that this solver is assigned to a different geometry than ``THERMAL`` and ``ELECTRICAL`` solvers - a geometry we adjusted for optical simulations. ``DIFFUSION`` and ``GAIN`` could be assigned to either full, or optical geometry, but in the second case we limit the calculations range to the mesa radius (instead of calculating in the air outside the mesa for the range of the full geometry, which is the heatsink radius equal to 2500 microns), therefore saving some time and memory.
==>  TODO: diffusion and gain description...

with our solvers defined, we have to connect them properly:

.. code-block:: xml

 <connects>
  <connect in="ELECTRICAL.inTemperature" out="THERMAL.outTemperature"/>
  <connect in="THERMAL.inHeatDensity" out="ELECTRICAL.outHeatDensity"/>

  <connect in="DIFFUSION.inTemperature" out="THERMAL.outTemperature"/>
  <connect in="DIFFUSION.inCurrentDensity" out="THERMAL.outCurrentDensity"/>

  <connect in="GAIN.inTemperature" out="THERMAL.outTemperature"/>
  <connect in="GAIN.inCarriersConcentration"
  out="DIFFUSION.outCarriersConcentration"/>

  <connect in="OPTICAL.inTemperature" out="THERMAL.outTemperature"/>
  <connect in="OPTICAL.inGain" out="GAIN.outGain"/>
 </connects>

These are all the connects we need in our case. First two are for achieving self-consistency in the thermo-electrical part. The final temperature distribution calculated by ``THERMAL`` solver will be then used by all other solvers. Additionally the ``DIFFUSION`` solver has to be provided with the current density distribution from ``ELECTRICAL`` solver, ``GAIN`` requires carriers concentration obtained in ``DIFFUSION`` to calculate gain, which then has to be connected to the ``OPTICAL`` solver in the end.

Manual refinements of divide mesh generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We could now run our calculations, but it is a good habit, to check the geometries for any design flaws and grids for proper density. To do this, we will write a simple script (remember to include it within ``<script><![CDATA[`` and ``]]></script>`` tags), that will just draw the ``GeoTE`` geometry and the ``default`` grid with the boundary conditions:

::

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

Close the XPL file with the ``</plask>`` tag and execute it. You can now see, that the lattice is rather sparse. It could be improved by increasing the values in the ``<postdiv by0="2" by1="2"/>`` line (that corresponds to horizontal and vertical divisions of every element in the geometry), but it would either end with a mesh that is still quite sparse at important locations, or a very dense and calculations-ineffective. Plask allows for a better approach, that is a manual addition of refinements at desired location in desired dimension. Let's modify our ``default`` mesh generator by adding a vertical refinement at the very bottom of the heatsink, where the temperature boundary condition is located, and two horizontal refinements at the inner part of the oxidation, where strong current crowding is expected:

.. code-block:: xml

 <grids>
  <generator type="rectilinear2d" method="divide" name="default">
   <postdiv by0="2" by1="2"/>
   <refinements>
    <axis1 object="p-contact" at="50"/>
    <axis0 object="oxideOut" at="-0.1"/>
    <axis0 object="oxideOut" at="-0.05"/>
   </refinements>
  </generator>

The refinements have to be included in the ``refinements`` section and are described with the ``axis#`` tag, where ``#`` means the dimension (0 - horizontal, 1 - vertical *r* and *z* in our case). The ``at`` attribute places a single refinement line at the location distant by the value provided in the ``at`` attribute from the bottom-left corner of the specified ``object`` in the desired direction (specified with the ``axis#`` tag). So the first refinement will add a single refinement line 50 microns in the *z* direction above the bottom of the ``p-contact`` (heatsink), while the next two will place 2 horizontal refinements to the left of the ``oxideOut`` object's left edge. The last two refinements are defined outside the object they are referred to, which will result in a warning-message when executing the file. We defined these refinements this way on purpose, because this notation is simpler than referring to the ``oxideIn`` object and using expressions with predefined values (``<axis0 object="oxideIn" at="{apertureRadius-0.1}"/>``) and we are sure, that these refinements will be within our geometry. Therefore we can ignore corresponding warnings, but it is always important to check the warning-messages, as they may point to a serious flaw in our code, especially when lots of predefined variables or/and real-time geometry changes are involved. It is possible to disable warnings, for this please refer to the https://phys.p.lodz.pl/doc/plask/xpl/grids.html#tag-warnings.
Instead of the ``at`` attribute, it is also possible to use either ``by``, or ``every`` attribute. ``by`` results in dividing the specified objects into provided number of elements, while ``every`` places refinement lines spaced equally with a distance specified as this attribute value. Adding a single refinement line does not actually result in a single refinement in the final mesh, as it creates an artificial element in the geometry, for which the grid is then generated. This results in densification of the mesh with regard to all the elements. You can see the new mesh by executing the file again.

Threshold current calculations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With having the geometries and meshes prepared, we can move on to the script part. Like in the previous tutorial, we are going to create a separate tutorial3.py file for the scripting purpose and begin it with::

 import sys
 import scipy.optimize

 filename = sys.argv[1]
 loadxpl(filename)

Then we can move directly to defining a function for the brenq root-finding algorithm, but this time, it will take the voltage applied to the structure as it's argument and return the material losses::

 def lossVsVoltage(voltage):
  ELECTRICAL.voltage_boundary[0].value = voltage
  verr = electr.compute(1)
  terr = therm.compute(1)
  while terr > THERMAL.maxerr or verr > ELECTRICAL.maxerr:
   verr = ELECTRICAL.compute(8)
   terr = THERMAL.compute(1)
  DIFFUSION.compute_threshold()
  mode_number = OPTICAL.find_mode(980.)
  mode_wavelength = OPTICAL.outWavelength(mode_number)
  mode_loss = OPTICAL.outLoss(mode_number)
  print_log(LOG_INFO, "voltage = " + str(voltage) + ", current = "
   + ELECTRICAL.get_total_current() + ", material losses " + str(mode_loss))
  return mode_loss




threshold_voltage = scipy.optimize.brentq(lossVsVoltage,0.,2500., xtol=0.1)


lossVsVoltage(threshold_voltage)
threshold_current = abs(ELECTRICAL.get_total_current())





Real-time structure modifications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

set voltage -> current vs aperture


