.. _sec-xpl-geometry-3d-objects:

Physical objects 3D
^^^^^^^^^^^^^^^^^^^

Physical objects are the leafs of the geometry tree. They represent actual objects that have some shape and defined material.

.. xml:tag:: <block3D/> (or <block/>)

   Corresponding Python class: :py:class:`plask.geometry.Block3D`.

   Rectangular block. Its origin is located in the lower back left corner.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr material: Definition of the block material (for solid blocks).
   :attr material-bottom: Definition of the material of the bottom of the block (for blocks with material changing from bottom to top). You should also set ``material-top`` and both materials can differs only in composition or amount of dopant.
   :attr material-top: Definition of the material of top of the block (see also ``material-bottom``).
   :attr material-shape: Exponent for graded materials. Setting this value to anything different than one allows non-linear change the block material.
   :attr required d{X}: where **{X}** is the longitudinal axis name: Longitudinal dimension of the cuboid. (float, µm)
   :attr required d{Y}: where **{Y}** is the transverse axis name: Transversal dimension of the cuboid. (float, µm)
   :attr required d{Z}: where **{Z}** is the vertical axis name: Vertical dimension of the cuboid. (float, µm)
   :attr depth: Alias for ``d{X}``.
   :attr width: Alias for ``d{Y}``.
   :attr height: Alias for ``d{Z}``.
   :attr angle: Rotation angle in the horizontal (longitudinal-vertical) plane [deg]. This rotates the cuboid horizontally, so the longitudinal and transverse dimensions are no longer along the axes, but correspond to the object sides.
   :attr steps-num: Maximum number of the mesh steps in each direction the object is divided into if it is non-uniform.
   :attr steps-dist: Minimum step size if the object is non-uniform.

   Either ``material`` or both ``material-top`` and ``material-bottom`` are required.

.. xml:tag:: <cuboid/>

   Alias for :xml:tag:`<block3d/>`.

.. xml:tag:: <cylinder/>

   Corresponding Python class: :py:class:`plask.geometry.Cylinder`.

   Cylinder with its base lying in the horizontal plane. Its origin is located at the center of the lower circular base.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr material: Definition of the cylinder material (for solid cylinders).
   :attr material-bottom: Definition of the material of the bottom of the cylinder (for cylinders with material changing from bottom to top). You should also set ``material-top`` and both materials can differs only in composition or amount of dopant.
   :attr material-top: Definition of the material of top of the cylinder (see also ``material-bottom``).
   :attr material-shape: Exponent for graded materials. Setting this value to anything different than one allows non-linear change the cylinder material.
   :attr required radius: Radius of the cylinder base.
   :attr required height: Height of the cylinder.
   :attr steps-num: Maximum number of the mesh steps in each direction the object is divided into if it is non-uniform.
   :attr steps-dist: Minimum step size if the object is non-uniform.

.. xml:tag:: <prism/>

   Corresponding Python class: :py:class:`plask.geometry.Prism`.

   Prism with base defined by a with vertices at specified lateral points.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr material-bottom: Definition of the material of the bottom of the polygon (for polygons with material changing from bottom to top). You should also set ``material-top`` and both materials can differs only in composition or amount of dopant.
   :attr material-top: Definition of the material of top of the polygon (see also ``material-bottom``).
   :attr material-shape: Exponent for graded materials. Setting this value to anything different than one allows non-linear change the polygon material.
   :attr required height: Prism height in the vertical direction. (float, µm)
   :attr steps-num: Maximum number of the mesh steps in each direction the object is divided into if it is non-uniform.
   :attr steps-dist: Minimum step size if the object is non-uniform.

   .. xml:contents::

      Polygon base longitudinal and transverse vertices separated by semicolons. Each vertex is defined by two coordinates separated by space.

   Example:

   .. _lis-polygon:
   .. code-block:: xml

      <cartesian3d name="example" axes="xyz">
        <prism height="2" material="GaAs"> -2 -2; -2 3; 2 3; 2 -2 </poly-prism>
      </cartesian3d>

.. xml:tag:: <sphere/> (or <circle3d/>)

   Corresponding Python class: :py:class:`plask.geometry.Sphere`.

   Sphere with centre at point (0, 0, 0).

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr material: Definition of the sphere material (for solid triangles).
   :attr material-bottom: Definition of the material of the bottom of the sphere (for spheres with material changing from bottom to top). You should also set ``material-top`` and both materials can differs only in composition or amount of dopant.
   :attr material-top: Definition of the material of top of the sphere (see also ``material-bottom``).
   :attr material-shape: Exponent for graded materials. Setting this value to anything different than one allows non-linear change the sphere material.
   :attr required radius: Radius of the sphere. (float, µm)
   :attr steps-num: Maximum number of the mesh steps in each direction the object is divided into if it is non-uniform.
   :attr steps-dist: Minimum step size if the object is non-uniform.

.. xml:tag:: <triangular-prism/>

   Corresponding Python class: :py:class:`plask.geometry.TriangularPrism`.

   Prism with triangular base having one vertex at point (0, 0, 0) and specified height.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr material: Definition of the triangle material (for solid triangles).
   :attr material-bottom: Definition of the material of the bottom of the triangle (for triangles with material changing from bottom to top). You should also set ``material-top`` and both materials can differs only in composition or amount of dopant.
   :attr material-top: Definition of the material of top of the triangle (see also ``material-bottom``).
   :attr material-shape: Exponent for graded materials. Setting this value to anything different than one allows non-linear change the triangle material.
   :attr required a{X}: where **{X}** is the longitudinal axis name: Coordinate of the first base triangle vertex. (float, µm)
   :attr required a{Y}: where **{Y}** is the transverse axis name: Coordinate of the first base triangle vertex. (float, µm)
   :attr required b{X}: where **{X}** is the longitudinal axis name: Coordinate of the second base triangle vertex. (float, µm)
   :attr required b{Y}: where **{Y}** is the transverse axis name: Coordinate of the second base triangle vertex. (float, µm)
   :attr required height: Prism height in the vertical direction. (float, µm)
   :attr steps-num: Maximum number of the mesh steps in each direction the object is divided into if it is non-uniform.
   :attr steps-dist: Minimum step size if the object is non-uniform.

.. xml:tag:: <tube/>

   Corresponding Python class: :py:class:`plask.geometry.Tube`.

   Tube (i.e. hollow tube)  with its base lying in the horizontal plane. Its origin is located at the center of the lower circular base.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr material: Definition of the tube material (for solid tubes).
   :attr material-bottom: Definition of the material of the bottom of the tube (for tubes with material changing from bottom to top). You should also set ``material-top`` and both materials can differs only in composition or amount of dopant.
   :attr material-top: Definition of the material of top of the tube (see also ``material-bottom``).
   :attr material-shape: Exponent for graded materials. Setting this value to anything different than one allows non-linear change the tube material.
   :attr required inner-radius: Inner radius of the tube base.
   :attr required outer-radius: Outer radius of the tube base.
   :attr required height: Height of the tube.
   :attr steps-num: Maximum number of the mesh steps in each direction the object is divided into if it is non-uniform.
   :attr steps-dist: Minimum step size if the object is non-uniform.
