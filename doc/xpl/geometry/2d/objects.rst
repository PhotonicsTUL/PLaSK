Physical objects 2D
^^^^^^^^^^^^^^^^^^^

Physical objects are the leafs of the geometry tree. They represent actual objects that have some shape and defined material.

.. xml:tag:: <block2d/> (or <block/>)

   Corresponding Python class: :py:class:`plask.geometry.Block2D`.

   Rectangular block. Its origin is located at the lower left corner.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr material: Definition of the block material (for solid blocks).
   :attr material-bottom: Definition of the material of the bottom of the block (for blocks with material changing from bottom to top). You should also set ``material-top`` and both materials can differs only in composition or amount of dopant.
   :attr material-top: Definition of the material of top of the block (see also ``material-bottom``).
   :attr material-shape: Exponent for graded materials. Setting this value to anything different than one allows non-linear change the block material.
   :attr required d{X}: where **{X}** is the transverse axis name: Horizontal dimension of the rectangle. (float [µm])
   :attr required d{Y}: where **{Y}** is the vertical axis name: Vertical dimension of the rectangle. (float [µm])
   :attr width: Alias for ``d{X}``.
   :attr height: Alias for ``d{Y}``.
   :attr steps-num: Maximum number of the mesh steps in each direction the object is divided into if it is non-uniform.
   :attr steps-dist: Minimum step size if the object is non-uniform.

   Either ``material`` or both ``material-top`` and ``material-bottom`` are required.

.. xml:tag:: <rectangle/>

   Alias for :xml:tag:`<block2d/>`.

.. xml:tag:: <triangle/>

   Corresponding Python class: :py:class:`plask.geometry.Triangle`.

   Triangle with one vertex at point (0, 0).

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr material: Definition of the triangle material (for solid triangles).
   :attr material-bottom: Definition of the material of the bottom of the triangle (for triangles with material changing from bottom to top). You should also set ``material-top`` and both materials can differs only in composition or amount of dopant.
   :attr material-top: Definition of the material of top of the triangle (see also ``material-bottom``).
   :attr material-shape: Exponent for graded materials. Setting this value to anything different than one allows non-linear change the triangle material.
   :attr required a{X}: where **{X}** is the transverse axis name: Coordinate of the first triangle vertex. (float [µm])
   :attr required a{Y}: where **{Y}** is the vertical axis name: Coordinate of the first triangle vertex. (float [µm])
   :attr required b{X}: where **{X}** is the transverse axis name: Coordinate of the second triangle vertex. (float [µm])
   :attr required b{Y}: where **{Y}** is the vertical axis name: Coordinate of the second triangle vertex. (float [µm])
   :attr steps-num: Maximum number of the mesh steps in each direction the object is divided into if it is non-uniform.
   :attr steps-dist: Minimum step size if the object is non-uniform.

.. xml:tag:: <circle2d/> (or <circle/>)

   Corresponding Python class: :py:class:`plask.geometry.Circle`.

   Circle with centre at point (0, 0).

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr material-bottom: Definition of the material of the bottom of the circle (for circles with material changing from bottom to top). You should also set ``material-top`` and both materials can differs only in composition or amount of dopant.
   :attr material-top: Definition of the material of top of the circle (see also ``material-bottom``).
   :attr material-shape: Exponent for graded materials. Setting this value to anything different than one allows non-linear change the circle material.
   :attr required radius: Radius of the circle. (float [µm])
   :attr steps-num: Maximum number of the mesh steps in each direction the object is divided into if it is non-uniform.
   :attr steps-dist: Minimum step size if the object is non-uniform.
