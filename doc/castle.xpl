<plask>

<geometry>
  <cartesian2d name="castle" axes="x,y">
    <stack trancenter="0">
      <item left="-50">
        <shelf>
          <shelf repeat="6">
            <rectangle material="InAs" dx="6" dy="6"/>
            <gap size="6"/>
          </shelf>
          <stack trancenter="0">
            <triangle material="AlAs" ax="-15.5" ay="-20"
                                      bx=" 15.5" by="-20"/>
            <align trancenter="0" vertcenter="0">
              <rectangle material="AlN" dx="25" dy="25"/>
              <item vertcenter="4">
                <circle material="air" radius="5"/>
              </item>
            </align>
          </stack>
        </shelf>
      </item>
      <rectangle material="InN" dx="100" dy="15"/>
      <align trancenter="0" top="0">
        <rectangle material="GaN" dx="100" dy="10"/>
        <shelf repeat="11">
          <gap size="1"/>
          <clip top="0">
            <circle material="AlAs:C=1e20" radius="3.5"/>
          </clip>
          <gap size="1"/>
        </shelf>
      </align>
      <shelf>
        <rectangle name="gate" material="Cu" dx="35" dy="30"/>
        <gap total="100"/>
        <again ref="gate"/>
      </shelf>
      <rectangle material="GaAs" dx="200" dy="10"/>
    </stack>
  </cartesian2d>
</geometry>

<script><![CDATA[
plot_geometry(GEO.castle, fill=True, margin=0.02)
gca().set_aspect('equal')
tight_layout(0.1)
show()
]]></script>

</plask>
