<plask loglevel="detail">

<geometry>
  <cartesian3d name="lattice" axes="x,y">
    <align>
      <item back="-2" left="-2">
        <cuboid material="AlAs" dz="4" dx="5" dy="1"/>
      </item>
      <item back="-1" left="-1" bottom="-1">
        <cuboid material="AlAs" dz="2" dx="2" dy="2"/>
      </item>
      <lattice az="1" ax="0" ay="0" bz="0" bx="1" by="0">
        <segments>-2 -2; -2 3; 2 3; 2 -2 ^ -1 -1; -1 1; 1 1; 1 -1 ^  1 4</segments>
        <cylinder material="AlN" radius="0.05" height="3"/>
      </lattice>
    </align>
  </cartesian3d>
</geometry>

<script><![CDATA[
plot_geometry(GEO.lattice, margin=0.02, plane="xz")
gca().set_aspect('equal')
tight_layout(0.8)
show()
]]></script>

</plask>
