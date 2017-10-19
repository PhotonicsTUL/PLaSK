<plask loglevel="detail">

<defines>
  <define name="aperture" value="8."/>
  <define name="mesa" value="{4 * aperture}"/>
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
  <cylindrical2d name="GeoE" axes="r,z">
    <stack>
      <item right="{mesa/2-1}">
        <rectangle name="n-contact" material="Au" dr="4" dz="0.0500"/>
      </item>
      <stack name="VCSEL">
        <rectangle material="GaAs:Si=2e+18" dr="{mesa/2}" dz="0.0700"/>
        <stack name="top-DBR" repeat="24">
          <rectangle material="Al(0.73)GaAs:Si=2e+18" dr="{mesa/2}" dz="0.0795"/>
          <rectangle material="GaAs:Si=2e+18" dr="{mesa/2}" dz="0.0700"/>
        </stack>
        <shelf>
          <rectangle name="aperture" material="AlAs:Si=2e+18" dr="{aperture/2}" dz="0.0160"/>
          <rectangle name="oxide" material="AlOx" dr="{(mesa-aperture)/2}" dz="0.0160"/>
        </shelf>
        <rectangle material="Al(0.73)GaAs:Si=2e+18" dr="{mesa/2}" dz="0.0635"/>
        <rectangle material="GaAs:Si=5e+17" dr="{mesa/2}" dz="0.1160"/>
        <stack name="junction" role="active">
          <stack repeat="4">
            <rectangle name="QW" role="QW" material="InGaAsQW" dr="{mesa/2}" dz="0.0050"/>
            <rectangle material="GaAs" dr="{mesa/2}" dz="0.0050"/>
          </stack>
          <again ref="QW"/>
        </stack>
        <rectangle material="GaAs:C=5e+17" dr="{mesa/2}" dz="0.1160"/>
        <stack name="bottom-DBR" repeat="30">
          <rectangle material="Al(0.73)GaAs:C=2e+18" dr="{mesa/2}" dz="0.0795"/>
          <rectangle material="GaAs:C=2e+18" dr="{mesa/2}" dz="0.0700"/>
        </stack>
      </stack>
      <zero/>
      <rectangle name="p-contact" material="GaAs:C=2e+18" dr="{mesa/2}" dz="5."/>
    </stack>
  </cylindrical2d>
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
  <cylindrical2d name="GeoO" axes="r,z" outer="extend" bottom="GaAs" top="air">
    <again ref="VCSEL"/>
  </cylindrical2d>
</geometry>

<grids>
  <generator method="divide" name="default" type="rectangular2d">
    <postdiv by0="2"/>
    <warnings outside="no"/>
  </generator>
</grids>

<solvers>
  <meta name="SOLVER" solver="ThermoElectricCyl" lib="shockley">
    <geometry electrical="GeoE" thermal="GeoT"/>
    <mesh electrical="default" thermal="default"/>
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
    <junction beta0="11" js0="1"/>
  </meta>
</solvers>

<script><![CDATA[
import unittest

class GratingTest(unittest.TestCase):

    def testComputations(self):
        SOLVER.compute(save=False)
        self.assertAlmostEqual( SOLVER.get_total_current(), 1.754, 3 )


if __name__ == '__main__':
    import __main__
    __main__.GratingTest = GratingTest
    unittest.main()

]]></script>

</plask>
