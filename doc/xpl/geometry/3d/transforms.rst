.. _sec-xpl-geometry-3d-transforms:

Transforms 3D
^^^^^^^^^^^^^

Transforms always contain a single geometry object (possibly container) as their content and perform some transformation of this object.


.. xml:tag:: <arrange3d> (or <arrange>)

   Corresponding Python class: :py:class:`plask.geometry.Arrange3D`.

   Container in which replicates a single item in an equally-spaced row of its repetitions.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr d{X}: where **{X}** is the transverse axis name: Longitudinal component of the spacing vector, by which the adjacent repetitions are shifted. (float [µm])
   :attr d{Y}: where **{Y}** is the vertical axis name: Transverse dimension of the spacing vector, by which the adjacent repetitions are shifted. (float [µm])
   :attr d{Z}: where **{Z}** is the vertical axis name: Vertical dimension of the spacing vector, by which the adjacent repetitions are shifted. (float [µm])
   :attr required count: Number of repetitions of the container item.

   .. xml:contents::

      A single :ref:`three-dimensional geometry object <sec-xpl-Geometry-objects-3D>`. Object to arrange in a row.


.. xml:tag:: <clip3d> (or <clip>)

   Corresponding Python class: :py:class:`plask.geometry.Clip3D`.

   Clip child object to given cuboid.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr back: back edge of the clipping cuboid (-INF by default)
   :attr front: front edge of the clipping cuboid (+INF by default)
   :attr left: left edge of the clipping cuboid (-INF by default)
   :attr right: right edge of the clipping cuboid (+INF by default)
   :attr bottom: bottom edge of the clipping cuboid (-INF by default)
   :attr top: top edge of the clipping cuboid (+INF by default)

   .. xml:contents::

       A single :ref:`three-dimensional geometry object <sec-xpl-Geometry-objects-3D>`. Object to clip.


.. xml:tag:: <extrusion>

   Corresponding Python class: :py:class:`plask.geometry.Extrusion`.

   Extrusion of two-dimensional object into third dimension. 2D objects are defined in the plane defined by the transverse and vertical axes. Hence, the extrusion is performed into the longitudinal direction.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr required length: Length of the extrusion.

   .. xml:contents::

       A single :ref:`two-dimensional geometry object <sec-xpl-Geometry-objects-2D>`.


.. xml:tag:: <flip3d> (or <flip>)

   Corresponding Python class: :py:class:`plask.geometry.Flip3D`.

   Mirror reflection of the object along specified axis.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr required axis: Name of the inverted axis (i.e. perpendicular to the reflection plane).

   .. xml:contents::

       A single :ref:`three-dimensional geometry object <sec-xpl-Geometry-objects-3D>`.



.. xml:tag:: <intersection3d> (or <intersection>)

   Corresponding Python class: :py:class:`plask.geometry.Intersection3D`.

   Clip child object to given shape.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.

   .. xml:contents::

       Two :ref:`three-dimensional geometry objects <sec-xpl-Geometry-objects-3D>`. First is a child of this, second is a clipping shape (materials are not required in its leafs).


.. xml:tag:: <lattice>

   Corresponding Python class: :py:class:`plask.geometry.Lattice`.

   Lattice container that arranges its children in two-dimensional lattice.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr required a{X}: where **{X}** is the longitudinal axis name: Coordinate of the first basis vector. (float [µm])
   :attr required a{Y}: where **{Y}** is the transverse axis name: Coordinate of the first basis vector. (float [µm])
   :attr required a{Z}: where **{Y}** is the vertical axis name: Coordinate of the first basis vector. (float [µm])
   :attr required b{X}: where **{X}** is the longitudinal axis name: Coordinate of the second basis vector. (float [µm])
   :attr required b{Y}: where **{Y}** is the transverse axis name: Coordinate of the second basis vector. (float [µm])
   :attr required b{Z}: where **{Y}** is the vertical axis name: Coordinate of the second basis vector. (float [µm])

   .. xml:contents::

       :xml:tag:`segments  [in lattice]` tag followed by a :ref:`three-dimensional geometry object <sec-xpl-Geometry-objects-3D>`.

       .. xml:tag:: <segments> [in lattice]

          .. xml:contents::

             One or more polygons separated by ``^`` characters. Each polygon is formed by two or more vertices, separated by ``;`` characters. Each vertex consists of two space-separated integers.

   Example:

   .. _lis-lattice:
   .. code-block:: xml

      <cartesian3d name="lattice" axes="xyz">
        <lattice ax="{sqrt(3)/2}" ay="0.5" az="0" bx="0" by="1" bz="0">
          <segments>
            -2 -2; -2 3; 2 3; 2 -2 ^
            -1 -1; -1 1; 1 1; 1 -1 ^
             1 4
          </segments>
          <cylinder material="AlN" radius="0.3" height="3"/>
        </lattice>
      </cartesian3d>



   .. _fig-lattice-polygons:
   .. figure:: lattice_helper.*
      :scale: 100%
      :align: center

      Points described by the :xml:tag:`segments  [in lattice]` tag with the following content: ``-2 -2; -2 3; 2 3; 2 -2 ^ -1 -1; -1 1; 1 1; 1 -1 ^ 1 4``, same as :ref:`in listing with example <lis-lattice>`. The first polygon (``-2 -2; -2 3; 2 3; 2 -2``) is the red rectangle. The second one (``-1 -1; -1 1; 1 1; 1 -1``) is green. Since the green polygin lies inside the red one, it is subtracted from it (XOR operation). Points lying at polygoin edges (both red and green rectangles) are always included in the final lattice. The last segment (``1 4``) comprises of only one point. It is also added to final figure (XOR operation).


   .. _fig-lattice-final:
   .. figure:: lattice_final.*
      :scale: 100%
      :align: center

      Lattice described by :ref:`the example code <lis-lattice>`. Top view. Each ``(x,y)`` point described in the  :xml:tag:`segments  [in lattice]` tag (and showed in :ref:`the previous figure <fig-lattice-polygons>`) is combined with basis vectors ``a`` and ``b`` to obtain the location of the lattice contents instance.


.. xml:tag:: <revolution>

   Corresponding Python class: :py:class:`plask.geometry.Revolution`.

   Revolution of the two-dimensional object around its local vertical axis. The horizontal axis of the 2D object becomes a radial axis of the resulting compound cylinder. Vertical axes of the 2D object remains the vertical axis of the resulting block.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr auto-clip: If true item will be implicitly clipped to non-negative tran. coordinates. (bool, false by default)

   .. xml:contents::

       A single :ref:`two-dimensional geometry object <sec-xpl-Geometry-objects-2D>`. If auto-clip is off, all the boundaries of its bounding box must have their horizontal coordinates larger or equal to zero i.e. all the object must be located at the right-hand half of the plane. If auto-clip is on, they will be implicitly clipped.



.. xml:tag:: <mirror3d> (or <mirror>)

   Corresponding Python class: :py:class:`plask.geometry.Mirror3D`.

   Object mirrored along specified axis. In other words this is transformed object together with its flipped version. The bounding box of the object cannot span at bot sides of zero along inverted axis.

   :attr name: Object name for further reference.
   :attr role: Object role. Important for some solvers.
   :attr required axis: Name of the inverted axis (i.e. perpendicular to the reflection plane).

   .. xml:contents::

       A single :ref:`three-dimensional geometry object <sec-xpl-Geometry-objects-3D>`.





.. xml:tag:: <translation3D> (or <translation>)

   Corresponding Python class: :py:class:`plask.geometry.Translation3D`.

   A simple shift of the object. Note that the bounding box is shifted as well, so in containers that place their items basing on their bounding boxes, this transformation will have no effect.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr {X}: where **{X}** is the longitudinal axis name: Longitudinal position of the origin of transformed element. (float [µm])
   :attr {Y}: where **{Y}** is the transverse axis name: Transversal position of the origin of transformed element. (float [µm])
   :attr {Z}: where **{Z}** is the vertical axis name: Vertical position of the origin of transformed element. (float [µm])

   .. xml:contents::

       A single :ref:`three-dimensional geometry object <sec-xpl-Geometry-objects-3D>`.
