<plask loglevel="detail">

<geometry>
  <cartesian3d name="lattice" axes="x,y,z">
    <align>
      <item back="-2" left="-2">
        <cuboid material="AlAs" dx="4" dy="5" dz="1"/>
      </item>
      <item back="-1" left="-1" bottom="-1">
        <cuboid material="AlAs" dx="2" dy="2" dz="2"/>
      </item>
      <lattice ax="1" ay="0" az="0" bx="0" by="1" bz="0">
        <segments>-2 -2; -2 3; 2 3; 2 -2 ^ -1 -1; -1 1; 1 1; 1 -1 ^  1 4</segments>
        <cylinder material="AlN" radius="0.05" height="4"/>
      </lattice>
    </align>
  </cartesian3d>
</geometry>

<script><![CDATA[
rc.figure.figsize = 6.0, 4.5
plot_geometry(GEO.lattice, margin=0.02, plane="yx")
gca().set_aspect('equal')
tight_layout(0.8)
show()
]]></script>

</plask>
