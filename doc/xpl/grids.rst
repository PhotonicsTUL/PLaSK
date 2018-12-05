.. _sec-xpl-grids:

Section <grids>
===============

.. xml:tag:: <grids>

In this section one can define computational meshes for use by solvers. It can be done by one of the two ways: either by specifying the mesh directly or, by creating a generator that will automatically construct the required mesh basing on the structure geometry when the calculations in the solver using particular generator are about to begin. Hence the two allowed tags in this section are ``<mesh>`` and ``<generator>``. The contents of these tags are determined by the particular mesh or generator type, while their attributes are always the same:

.. xml:tag:: <generator>

   Specification of the mesh generator.

   :attr required name: Object name for further reference.
   :attr required type: Type of the mesh to generate.
   :attr required method: Generation method i.e. the type of the generator.

   .. xml:contents::

       The content of this element depends on the values of the type and method tag. It specifies generator configuration (if any). See below for details.

.. xml:tag:: <mesh>

   Specification of the mesh.

   :attr required name: Name of the mesh for reference in configuration of the solvers.
   :attr required type: Type of the mesh.

   .. xml:contents::

       The content of this element depends on the value of the type tag. See below for details.

Possible <mesh> contents for different types
--------------------------------------------

.. xml:tag:: <mesh type="ordered"> [ordered]

   One-dimensional rectangular mesh.

   .. xml:contents::

      .. xml:tag:: <axis> [in ordered mesh]

         Specification of the horizontal axis.

         If any of the following attributes are specified, the points along this axis are equally distributed like in regular meshes. In such a case the contents must be empty.

         :attr start: Position of the first point on the axis. (float [µm])
         :attr stop: Position of the last point on the axis. (float [µm])
         :attr num: Number of the equally distributed points along the axis. (integer)

         .. xml:contents::

            Comma-separated list of the mesh points along this axis.


.. xml:tag:: <mesh type="regular"> [regular]

   One-dimensional rectangular mesh with regular intervals.

   .. xml:contents::

      .. xml:tag:: <axis> [in regular mesh]

         Specification of the horizontal axis.

         :attr required start: Position of the first point on the axis. (float [µm])
         :attr required stop: Position of the last point on the axis. (float [µm])
         :attr required num: Number of the equally distributed points along the axis. (integer)

.. xml:tag:: <mesh type="regular1d"> [regular1d]

   Deprecated alias to ordered :xml:tag:`<mesh> [regular]`.



.. xml:tag:: <mesh type="rectangular2d"> [rectangular2d]

   Two-dimensional rectangular mesh.

   .. xml:contents::

      .. xml:tag:: <axis0> [in rectangular2d mesh]

         Specification of the horizontal axis.

         :attr type: Type of axis: ``ordered`` or ``regular``. If not given, axis will be ``regular`` only if any of the ``start``, ``stop`` or ``num`` attributes are specified (in other case it will be ``ordered``).
         :attr start: Position of the first point on the axis. (float [µm])
         :attr stop: Position of the last point on the axis. (float [µm])
         :attr num: Number of the equally distributed points along the axis. (integer)

         .. xml:contents::

            Comma-separated list of the mesh points along this axis. Allowed and required only for ``rectangular`` axis.

      .. xml:tag:: <axis1> [in rectangular2d mesh]

         Specification of the vertical axis.

         Attributes and contents are in the same format as in :xml:tag:`<axis0> [in rectangular2d mesh]`.

.. xml:tag:: <mesh type="regular2d"> [regular2d]

   Deprecated alias to :xml:tag:`<mesh> [rectangular2d]`.

.. xml:tag:: <mesh type="triangular2d"> [triangular2d]

   Two-dimensional triangular mesh.

   .. xml:contents::

      Either a sequence of ``<triangle>`` tags or a sequence of ``<node>`` tags followed by sequence of ``<element>`` tags.

      The sequence of ``<node>`` tags describes a 0-indexed vector of nodes.

      .. xml:tag:: <triangle> [in triangular2d mesh]

         Specification of a triangular element of the mesh; a triangle.

         :attr a0: horizontal coordinate of the first vertex of the triangle. (float [µm])
         :attr a1: vertical coordinate of the first vertex of the triangle. (float [µm])
         :attr b0: horizontal coordinate of the second vertex of the triangle. (float [µm])
         :attr b1: vertical coordinate of the second vertex of the triangle. (float [µm])
         :attr c0: horizontal coordinate of the third vertex of the triangle. (float [µm])
         :attr c1: vertical coordinate of the third vertex of the triangle. (float [µm])

      .. xml:tag:: <node> [in triangular2d mesh]

         Node specification; a 2D point; a vertex of triangular element.

         :attr tran: horizontal coordinate of the point. (float [µm])
         :attr vert: vertical coordinate of the point. (float [µm])

      .. xml:tag:: <element> [in triangular2d mesh]

         Specification of a triangular element of the mesh; a triangle.

         :attr a: index of the first vertex of the triangle in the vector of nodes. (integer)
         :attr b: index of the second vertex of the triangle in the vector of nodes. (integer)
         :attr c: index of the third vertex of the triangle in the vector of nodes. (integer)


.. xml:tag:: <mesh type="rectangular3d"> [rectangular3d]

   Three-dimensional rectangular mesh.

   .. xml:contents::

      .. xml:tag:: <axis0> [in rectangular3d mesh]

         Specification of the longitudinal axis.

         :attr type: Type of axis: ``ordered`` or ``regular``. If not given, axis will be ``regular`` only if any of the ``start``, ``stop`` or ``num`` attributes are specified (in other case it will be ``ordered``).
         :attr start: Position of the first point on the axis. (float [µm])
         :attr stop: Position of the last point on the axis. (float [µm])
         :attr num: Number of the equally distributed points along the axis. (integer)

         .. xml:contents::

            Comma-separated list of the mesh points along this axis. Allowed and required only for ``rectangular`` axis.

      .. xml:tag:: <axis1> [in rectangular3d mesh]

         Specification of the transversal axis.

         Attributes and contents are in the same format as in :xml:tag:`<axis0> [in rectangular3d mesh]`.

      .. xml:tag:: <axis2> [in rectangular3d mesh]

         Specification of the vertical axis.

         Attributes and contents are in the same format as in :xml:tag:`<axis0> [in rectangular3d mesh]`.

.. xml:tag:: <mesh type="regular3d"> [regular3d]

   Deprecated alias to :xml:tag:`<mesh> [rectangular3d]`.


Possible <generator> contents for different types and methods
-------------------------------------------------------------

.. xml:tag:: <generator type="ordered" method="divide"> [ordered, divide]

   Generator that divides each geometry object along both axes into a specified number of elements, ensuring that two adjacent do not differ in size more than twice.

   .. xml:contents::

      .. xml:tag:: <options/> [in ordered, divide generator]

         :attr gradual: Turn on/off smooth mesh step (i.e. if disabled, the adjacent elements of the generated mesh may differ more than by the factor of two). (bool, ``yes`` by default)

      .. xml:tag:: <prediv/> [in ordered, divide generator]

         Set number of the initial divisions of each geometry object.

         :attr by: Number of parts each object is divided into along horizontal axis.

      .. xml:tag:: <postdiv/> [in ordered, divide generator]

         Set number of the final divisions of each geometry object.

         :attr by: Number of parts each object is divided into along horizontal axis.

      .. xml:tag:: <refinements> [in ordered, divide generator]

         Specify list of additional refinements of the generated mesh.

         .. xml:contents::

            .. xml:tag:: <axis0/> [in ordered, divide generator]

               Add refinement to the horizontal axis.

               :attr required object: Name of the geometry object to add additional division to.
               :attr path: Path name, specifying particular instance of the object given in the object attribute.
               :attr at: If this attribute is present, a single refinement line is placed at the position specified in it (in the local object coordinates).
               :attr by: If this attribute is present, multiple refinement lines are placed dividing the object into a specified number of equal parts.
               :attr every: If this attribute is present, multiple refinement lines are places at distance from each other specified in the attribute value.

               Exactly one of ``at``, ``by``, or ``every`` attribute must be present.

      .. xml:tag:: <warnings/>

         Control printing of the warnings.

         :attr missing: Warn if any refinement references to non-existing object. Defaults to ‘yes’. (bool)
         :attr multiple: Warn if any refinement references to multiple objects. Defaults to ‘yes’. (bool)
         :attr outside: Warn if refining line lies outside of the specified object. Defaults to ‘yes’. (bool)


.. xml:tag:: <generator type="ordered" method="regular"> [ordered, regular]

   Generator creating the mesh with lines at transverse edges of all objects and fine regular division of each object with spacing approximately equal to the specified one.

   .. xml:tag:: <spacing/> [in ordered, regular generator]

      :attr every: Approximate single element size.

   .. xml:tag:: <boundaries/> [in ordered, regular generator]

      :attr split: Split mesh lines at object boundaries. This is useful mostly for plotting discontinous material parameters. Defaults to ‘no’. (bool)


.. xml:tag:: <generator type="ordered" method="simple"> [ordered, simple]

   Simple generator creating the rectangular mesh consisting of ordered axes with lines at the edges of bounding boxes of each object of the geometry.

   .. xml:tag:: <boundaries/> [in ordered, simple generator]

      :attr split: Split mesh lines at object boundaries. This is useful mostly for plotting discontinous material parameters. Defaults to ‘no’. (bool)


.. xml:tag:: <generator type="ordered" method="smooth"> [ordered, smooth]

   Generator that divides each geometry object along both axes with elements of given size near a boundary and increasing geometrically by given factor towards the middle of an object.

   .. xml:contents::

      .. xml:tag:: <steps/> [in ordered, smooth generator]

         Parametrs specifying element size.

         :attr small: Maximum size of the smallest elements near objects edges.

         :attr factor: Factor by which sizes of elements increase towards objects centers.

      .. xml:tag:: <refinements> [in ordered, smooth generator]

         Specify list of additional refinements of the generated mesh.

         .. xml:contents::

            .. xml:tag:: <axis0/> [in ordered, smooth generator]

               Add refinement to the horizontal axis.

               :attr required object: Name of the geometry object to add additional division to.
               :attr path: Path name, specifying particular instance of the object given in the object attribute.
               :attr at: If this attribute is present, a single refinement line is placed at the position specified in it (in the local object coordinates).
               :attr by: If this attribute is present, multiple refinement lines are placed dividing the object into a specified number of equal parts.
               :attr every: If this attribute is present, multiple refinement lines are places at distance from each other specified in the attribute value.

               Exactly one of ``at``, ``by``, or ``every`` attribute must be present.

      .. xml:tag:: <warnings/>

         Control printing of the warnings.

         :attr missing: Warn if any refinement references to non-existing object. Defaults to ‘yes’. (bool)
         :attr multiple: Warn if any refinement references to multiple objects. Defaults to ‘yes’. (bool)
         :attr outside: Warn if refining line lies outside of the specified object. Defaults to ‘yes’. (bool)


.. xml:tag:: <generator type="rectangular2d" method="divide"> [rectangular2d, divide]

   Generator that divides each geometry object along both axes into a specified number of elements, ensuring that two adjacent do not differ in size more than twice.

   .. xml:contents::

      .. xml:tag:: <gradual/> [in rectangular2d, divide generator]

         Turn on/off smooth mesh step (i.e. if disabled, the adjacent elements of the generated mesh may differ more than by the factor of two).

         Gradual is enabled by default.

         :attr all: enable (if ``yes``) or disable (if ``no``) gradual for all axes (bool, ``yes`` by default)

      .. xml:tag:: <prediv/> [in rectangular2d, divide generator]

         Set number of the initial divisions of each geometry object.

         :attr by0: Number of parts each object is divided into along horizontal axis.
         :attr by1: Number of parts each object is divided into along vertical axis.
         :attr by: Set values of ``by0`` and ``by1`` both at once. It this attribute is specified, no other ones are allowed.

      .. xml:tag:: <postdiv/> [in rectangular2d, divide generator]

         Set number of the final divisions of each geometry object.

         It has same attributes as :xml:tag:`<prediv/> [in rectangular2d, divide generator]`.

      .. xml:tag:: <refinements> [in rectangular2d, divide generator]

         Specify list of additional refinements of the generated mesh.

         .. xml:contents::

            .. xml:tag:: <axis0/> [in rectangular2d, divide generator]

               Add refinement to the horizontal axis.

               :attr required object: Name of the geometry object to add additional division to.
               :attr path: Path name, specifying particular instance of the object given in the object attribute.
               :attr at: If this attribute is present, a single refinement line is placed at the position specified in it (in the local object coordinates).
               :attr by: If this attribute is present, multiple refinement lines are placed dividing the object into a specified number of equal parts.
               :attr every: If this attribute is present, multiple refinement lines are places at distance from each other specified in the attribute value.

               Exactly one of ``at``, ``by``, or ``every`` attribute must be present.

            .. xml:tag:: <axis1/> [in rectangular2d, divide generator]

               Add refinement to the vertical axis.

               It has same attributes as :xml:tag:`<axis0/> [in rectangular2d, divide generator]`.

      .. xml:tag:: <warnings/>

         Control printing of the warnings.

         :attr missing: Warn if any refinement references to non-existing object. Defaults to ‘yes’. (bool)
         :attr multiple: Warn if any refinement references to multiple objects. Defaults to ‘yes’. (bool)
         :attr outside: Warn if refining line lies outside of the specified object. Defaults to ‘yes’. (bool)


.. xml:tag:: <generator type="rectangular2d" method="regular"> [rectangular2d, regular]

   Generator creating the mesh with lines at transverse edges of all objects and fine regular division of each object with spacing approximately equal to the specified one.

   .. xml:tag:: <spacing/> [in rectangular2d, regular generator]

      :attr every: Approximate single element size along all axes (this is mutaally exclusive with all other attributes).
      :attr every0: Approximate single element size along horizontal axis.
      :attr every1: Approximate single element size along vertical axis.

   .. xml:tag:: <boundaries/> [in rectangular2d, regular generator]

      :attr split: Split mesh lines at object boundaries. This is useful mostly for plotting discontinous material parameters. Defaults to ‘no’. (bool)


.. xml:tag:: <generator type="rectangular2d" method="simple"> [rectangular2d, simple]

   Simple generator creating the rectangular mesh consisting of ordered axes with lines at the edges of bounding boxes of each object of the geometry.

   .. xml:tag:: <boundaries/> [in rectangular2d, simple generator]

      :attr split: Split mesh lines at object boundaries. This is useful mostly for plotting discontinous material parameters. Defaults to ‘no’. (bool)


.. xml:tag:: <generator type="rectangular2d" method="smooth"> [rectangular2d, smooth]

   Generator that divides each geometry object along both axes with elements of given size near a boundary and increasing geometrically by given factor towards the middle of an object.

   .. xml:contents::

      .. xml:tag:: <options/> [in rectangular2d, smooth generator]

         :attr aspect: Maximum aspect ratio for the rectangular and cubic elements generated by this generator. If set the mesh elements are additionally divided to ensure their aspect not larger than this value. (float)

      .. xml:tag:: <steps/> [in rectangular2d, smooth generator]

         Parameters specifying element size.

         :attr small: Maximum size of the smallest elements near objects edges.

         :attr factor: Factor by which sizes of elements increase towards objects centers.

      .. xml:tag:: <refinements> [in rectangular2d, smooth generator]

         Specify list of additional refinements of the generated mesh.

         .. xml:contents::

            .. xml:tag:: <axis0/> [in rectangular2d, smooth generator]

               Add refinement to the horizontal axis.

               :attr required object: Name of the geometry object to add additional division to.
               :attr path: Path name, specifying particular instance of the object given in the object attribute.
               :attr at: If this attribute is present, a single refinement line is placed at the position specified in it (in the local object coordinates).
               :attr by: If this attribute is present, multiple refinement lines are placed dividing the object into a specified number of equal parts.
               :attr every: If this attribute is present, multiple refinement lines are places at distance from each other specified in the attribute value.

               Exactly one of ``at``, ``by``, or ``every`` attribute must be present.

      .. xml:tag:: <warnings/>

         Control printing of the warnings.

         :attr missing: Warn if any refinement references to non-existing object. Defaults to ‘yes’. (bool)
         :attr multiple: Warn if any refinement references to multiple objects. Defaults to ‘yes’. (bool)
         :attr outside: Warn if refining line lies outside of the specified object. Defaults to ‘yes’. (bool)

.. xml:tag:: <generator type="triangular2d" method="triangle"> [triangular2d, triangle]

   Generator which creates triangular mesh by *Triangle* library authored by **Jonathan Richard Shewchuk**.

   Citing official *Triangle* documentation: *Triangle* generates exact Delaunay triangulations, constrained Delaunay triangulations, conforming Delaunay triangulations, Voronoi diagrams, and high-quality triangular meshes. The latter can be generated with no small or large angles, and are thus suitable for finite element analysis.

   See: https://www.cs.cmu.edu/~quake/triangle.html

   .. xml:contents::

      .. xml:tag:: <options/> [in triangular2d, triangle generator]

         :attr maxarea: A maximum triangle area constraint. (float [µm²])
         :attr minangle: Minimum angle. (float [°])

         Citing official *Triangle* documentation (http://www.cs.cmu.edu/~quake/triangle.q.html): If the minimum angle is 20.7° or smaller, the triangulation algorithm is theoretically guaranteed to terminate (assuming infinite precision arithmetic - *Triangle* may fail to terminate if you run out of precision). In practice, the algorithm often succeeds for minimum angles up to 33°. It usually does not terminate for angles above 34°. For some meshes, however, it may be necessary to reduce the minimum angle to well below 20° to avoid problems associated with insufficient floating-point precision.


.. xml:tag:: <generator type="rectangular3d" method="divide"> [rectangular3d, divide]

   Generator that divides each geometry object along both axes into a specified number of elements, ensuring that two adjacent do not differ in size more than twice.

   .. xml:contents::

      .. xml:tag:: <gradual/> [in rectangular3d, divide generator]

         Turn on/off smooth mesh step (i.e. if disabled, the adjacent elements of the generated mesh may differ more than by the factor of two).

         Gradual is enabled by default.

         :attr all: enable (if ``yes``) or disable (if ``no``) gradual for all axes (bool, ``yes`` by default)

      .. xml:tag:: <no-gradual/> [in rectangular3d, divide generator]

         Turn off smooth mesh step (i.e. the adjacent elements of the generated mesh may differ more than by the factor of two).

         Deprecated alias to ``<gradual all="false"/>``.

      .. xml:tag:: <prediv/> [in rectangular3d, divide generator]

         Set number of the initial divisions of each geometry object.

         :attr by0: Number of parts each object is divided into along longitudinal axis.
         :attr by1: Number of parts each object is divided into along transverse axis.
         :attr by2: Number of parts each object is divided into along vertical axis.
         :attr by: Set values of ``by0``, ``by1`` and ``by2`` at once. It this attribute is specified, no other ones are allowed.

      .. xml:tag:: <postdiv/> [in rectangular3d, divide generator]

         Set number of the final divisions of each geometry object.

         It has same attributes as :xml:tag:`<prediv/> [in rectangular3d, divide generator]`.

      .. xml:tag:: <refinements> [in rectangular3d, divide generator]

         Specify list of additional refinements of the generated mesh.

         .. xml:contents::

            .. xml:tag:: <axis0/> [in rectangular3d, divide generator]

               Add refinement to the longitudinal axis.

               :attr required object: Name of the geometry object to add additional division to.
               :attr path: Path name, specifying particular instance of the object given in the object attribute.
               :attr at: If this attribute is present, a single refinement line is placed at the position specified in it (in the local object coordinates).
               :attr by: If this attribute is present, multiple refinement lines are placed dividing the object into a specified number of equal parts.
               :attr every: If this attribute is present, multiple refinement lines are places at distance from each other specified in the attribute value.

               Exactly one of ``at``, ``by``, or ``every`` attribute must be present.

            .. xml:tag:: <axis1/> [in rectangular3d, divide generator]

               Add refinement to the transverse axis.

               It has same attributes as :xml:tag:`<axis0/> [in rectangular3d, divide generator]`.

            .. xml:tag:: <axis2/> [in rectangular3d, divide generator]

               Add refinement to the vertical axis.

               It has same attributes as :xml:tag:`<axis0/> [in rectangular3d, divide generator]`.

      .. xml:tag:: <warnings/>

         Control printing of the warnings.

         :attr missing: Warn if any refinement references to non-existing object. Defaults to ‘yes’. (bool)
         :attr multiple: Warn if any refinement references to multiple objects. Defaults to ‘yes’. (bool)
         :attr outside: Warn if refining line lies outside of the specified object. Defaults to ‘yes’. (bool)


.. xml:tag:: <generator type="rectangular3d" method="regular"> [rectangular3d, regular]

   Generator creating the mesh with lines at transverse edges of all objects and fine regular division of each object with spacing approximately equal to the specified one.

   .. xml:tag:: <spacing/> [in rectangular3d, regular generator]

      :attr every: Approximate single element size along all axes (this is mutaally exclusive with all other attributes).
      :attr every0: Approximate single element size along longitudinal axis.
      :attr every1: Approximate single element size along transverse axis.
      :attr every2: Approximate single element size along vertical axis.

   .. xml:tag:: <boundaries/> [in rectangular3d, regular generator]

      :attr split: Split mesh lines at object boundaries. This is useful mostly for plotting discontinous material parameters. Defaults to ‘no’. (bool)


.. xml:tag:: <generator type="rectangular3d" method="simple"> [rectangular3d, simple]

   Simple generator creating the rectangular mesh consisting of ordered axes with lines at the edges of bounding boxes of each object of the geometry.

   .. xml:tag:: <boundaries/> [in rectangular3d, simple generator]

      :attr split: Split mesh lines at object boundaries. This is useful mostly for plotting discontinous material parameters. Defaults to ‘no’. (bool)


.. xml:tag:: <generator type="rectangular3d" method="smooth"> [rectangular3d, smooth]

   Generator that divides each geometry object along both axes with elements of given size near a boundary and increasing geometrically by given factor towards the middle of an object.

   .. xml:contents::

      .. xml:tag:: <options/> [in rectangular3d, smooth generator]

         :attr aspect: Maximum aspect ratio for the rectangular and cubic elements generated by this generator. If set the mesh elements are additionally divided to ensure their aspect not larger than this value. (float)

      .. xml:tag:: <steps/> [in rectangular3d, smooth generator]

         Parameters specifying element size.

         :attr small: Maximum size of the smallest elements near objects edges.

         :attr factor: Factor by which sizes of elements increase towards objects centers.

      .. xml:tag:: <refinements> [in rectangular3d, smooth generator]

         Specify list of additional refinements of the generated mesh.

         .. xml:contents::

            .. xml:tag:: <axis0/> [in rectangular3d, smooth generator]

               Add refinement to the horizontal axis.

               :attr required object: Name of the geometry object to add additional division to.
               :attr path: Path name, specifying particular instance of the object given in the object attribute.
               :attr at: If this attribute is present, a single refinement line is placed at the position specified in it (in the local object coordinates).
               :attr by: If this attribute is present, multiple refinement lines are placed dividing the object into a specified number of equal parts.
               :attr every: If this attribute is present, multiple refinement lines are places at distance from each other specified in the attribute value.

               Exactly one of ``at``, ``by``, or ``every`` attribute must be present.

            .. xml:tag:: <axis1/> [in rectangular3d, divide generator]

               Add refinement to the transverse axis.

               It has same attributes as :xml:tag:`<axis0/> [in rectangular3d, divide generator]`.

            .. xml:tag:: <axis2/> [in rectangular3d, divide generator]

               Add refinement to the vertical axis.

               It has same attributes as :xml:tag:`<axis0/> [in rectangular3d, divide generator]`.

      .. xml:tag:: <warnings/>

         Control printing of the warnings.

         :attr missing: Warn if any refinement references to non-existing object. Defaults to ‘yes’. (bool)
         :attr multiple: Warn if any refinement references to multiple objects. Defaults to ‘yes’. (bool)
         :attr outside: Warn if refining line lies outside of the specified object. Defaults to ‘yes’. (bool)
