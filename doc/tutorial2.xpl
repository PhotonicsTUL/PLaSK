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
        <block dr="6" dz="0.01603" material="AlOx"/>
      </shelf>
      <block dr="10" dz="0.03178" material="Al(0.73)GaAs"/>
      <block dr="10" dz="0.13756" material="GaAs"/>
      <shelf>
        <block dr="4" dz="0.005" role="gain" material="active" name="gain-region"/>
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
