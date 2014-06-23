<plask>

<defines>
 <define name="mesaRadius" value="10."/>
 <define name="aperture" value="{mesaRadius-6.}"/>
</defines>

<materials>
	<material name="InGaAsQW" base="In(0.22)GaAs">
		<nr>3.621</nr>
		<absp>0</absp>
		<A>110000000</A>
		<B>7e-011-1.08e-12*(T-300)</B>
		<C>1e-029+1.4764e-33*(T-300)</C>
		<D>10+0.01667*(T-300)</D>
	</material>
</materials>

<geometry>
 <cylindrical2d axes="rz" name="GeoTE">
   <stack>
	 <shelf>
	   <gap total="{mesaRadius-1}"/>
	   <block dr="4" dz ="0.0500" material="Au" name="n-contact"/>
	 </shelf>
	 <stack name="VCSEL">
	   <stack name="top-DBR" repeat="24">
		 <block dr="{mesaRadius}" dz="0.07003" material="GaAs:Si=2e+18"/>
		 <block dr="{mesaRadius}" dz="0.07945" material="Al(0.73)GaAs:Si=2e+18"/>
	   </stack>
	   <block dr="{mesaRadius}" dz="0.07003" material="GaAs:Si=2e+18"/>
	   <block dr="{mesaRadius}" dz="0.03178" material="Al(0.73)GaAs:Si=2e+18"/>
	   <shelf>
		 <block dr="{aperture}" dz="0.01603" material="AlAs:Si=2e+18" name="aperture"/>
		 <block dr="{mesaRadius-aperture}" dz="0.01603" material="AlxOy" name="oxide"/>
	   </shelf>
	   <block dr="{mesaRadius}" dz="0.03178" material="Al(0.73)GaAs:Si=2e+18"/>
	   <block dr="{mesaRadius}" dz="0.11756" material="GaAs:Si=5e+17"/>
		<stack role="active" name="junction">
			<block dr="{mesaRadius}" dz="0.005" material="InGaAsQW" role="QW"/>
			<stack repeat="4">
				<block dr="{mesaRadius}" dz="0.005" material="GaAs"/>
				<block dr="{mesaRadius}" dz="0.005" material="InGaAsQW" role="QW"/>
			</stack>
		</stack>
	   <block dr="{mesaRadius}" dz="0.11756" material="GaAs:C=5e+17"/>
	   <stack name="bottom-DBR" repeat="29">
		 <block dr="{mesaRadius}" dz="0.07945" material="Al(0.73)GaAs:C=2e+18"/>
		 <block dr="{mesaRadius}" dz="0.07003" material="GaAs:C=2e+18"/>
	   </stack>
	   <block dr="{mesaRadius}" dz="0.07945" material="Al(0.73)GaAs:C=2e+18"/>
	 </stack>
	 <zero/>
	 <block dr="200." dz="150." material="GaAs:C=2e+18"/>
	 <block dr="2500." dz="5000." material="Cu" name="p-contact"/>
   </stack>
 </cylindrical2d>
 
 <cylindrical2d axes="rz" name="GeoO" top="air" bottom="GaAs" outer="extend">
   <again ref="VCSEL"/>
 </cylindrical2d>

</geometry>

<grids>
 <generator type="rectangular2d" method="divide" name="default">
   <postdiv by0="3" by1="2"/>
   <refinements>
	 <axis1 object="p-contact" at="50"/>
	 <axis0 object="oxide" at="-0.1"/>
	 <axis0 object="oxide" at="-0.05"/>
	 <axis0 object="aperture" at="0.1"/>
   </refinements>
 </generator>

 <mesh type="regular" name="diffusion">
   <axis start="0" stop="{mesaRadius}" num="2000"/>
 </mesh>

 <generator type="rectangular2d" method="divide" name="optical">
   <prediv by0="10" by1="3"/>
 </generator>

	<generator type="rectangular2d" method="divide" name="plots">
		<postdiv by="30"/>
	</generator>
</grids>

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



<script><![CDATA[

figure()
plot_geometry(GEO.GeoTE, set_limits=True)
gcf().canvas.set_window_title("GEO TE")

figure()
plot_geometry(GEO.GeoTE, set_limits=True)
defmesh = MSG.default(GEO.GeoTE.item)
plot_mesh(defmesh, color="0.75")
plot_boundary(ELECTRICAL.voltage_boundary, defmesh, ELECTRICAL.geometry, color="b", marker="D")
plot_boundary(THERMAL.temperature_boundary, defmesh, THERMAL.geometry, color="r")
gcf().canvas.set_window_title("Default mesh")
show()

]]></script>
</plask>