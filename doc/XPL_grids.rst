.. _sec-XPL-grids:

Section <grids>
===============

.. xml:tag:: <grids>

In this section one can define computational meshes for use by solvers. It can be done by one of the two ways: either by specifying the mesh directly or, by creating a generator that will automatically construct the required mesh basing on the structure geometry when the calculations in the solver using particular generator are about to begin. Hence the two allowed tags in this section are ``<mesh>`` and ``<generator>``. The contents of these tags are determined by the particular mesh or generator type, while their attributes are always the same:

.. xml:tag:: <generator>

   Specification of the mesh generator.

   :attr required name: Object name for further reference.
   :attr required type: Type of the mesh to generate.
   :attr required method: Generation method i.e. the type of the generator.

   :Contents: The content of this element depends on the values of the type and method tag. It specifies generator configuration (if any). See below for details.

.. xml:tag:: <mesh>

   Specification of the mesh.

   :attr required name: Name of the mesh for reference in configuration of the solvers.
   :attr required type: Type of the mesh.

   :Contents: The content of this element depends on the value of the type tag. See below for details.

Possible <mesh> contents for different types
--------------------------------------------

.. xml:tag:: <mesh type="rectilinear1d"> [rectilinear1d]

   One-dimensional rectangular mesh with regular intervals.

   :Contents:

   .. xml:tag:: <axis> [in rectilinear1d mesh]

      Specification of the horizontal axis.

      If any of the following attributes are specified, the points along this axis are equally distributed like in regular meshes. In such a case the contents must be empty.

      :attr start: Position of the first point on the axis. (float [µm])
      :attr stop: Position of the last point on the axis. (float [µm])
      :attr num: Number of the equally distributed points along the axis. (integer)

      :Contents: Comma-separated list of the mesh points along this axis.

.. xml:tag:: <mesh type="rectilinear2d"> [rectilinear2d]

   Two-dimensional rectangular mesh with regular intervals.

   :Contents:

   .. xml:tag:: <axis0> [in rectilinear2d mesh]

      Specification of the horizontal axis.

      If any of the following attributes are specified, the points along this axis are equally distributed like in regular meshes. In such a case the contents must be empty.

      :attr start: Position of the first point on the axis. (float [µm])
      :attr stop: Position of the last point on the axis. (float [µm])
      :attr num: Number of the equally distributed points along the axis. (integer)

      :Contents: Comma-separated list of the mesh points along this axis.

   .. xml:tag:: <axis1> [in rectilinear2d mesh]

      Specification of the vertical axis.

      Attributes and contents are in the same format as in :xml:tag:`<axis0> [in rectilinear2d mesh]`.

.. xml:tag:: <mesh type="rectilinear3d"> [rectilinear3d]

   Three-dimensional rectangular mesh with regular intervals.

   :Contents:

   .. xml:tag:: <axis0> [in rectilinear3d mesh]

      Specification of the longitudinal axis.

      If any of the following attributes are specified, the points along this axis are equally distributed like in regular meshes. In such a case the contents must be empty.

      :attr start: Position of the first point on the axis. (float [µm])
      :attr stop: Position of the last point on the axis. (float [µm])
      :attr num: Number of the equally distributed points along the axis. (integer)

      :Contents: Comma-separated list of the mesh points along this axis.

   .. xml:tag:: <axis1> [in rectilinear3d mesh]

      Specification of the transversal axis.

      Attributes and contents are in the same format as in :xml:tag:`<axis0> [in rectilinear3d mesh]`.

   .. xml:tag:: <axis2> [in rectilinear3d mesh]

      Specification of the vertical axis.

      Attributes and contents are in the same format as in :xml:tag:`<axis0> [in rectilinear3d mesh]`.

.. xml:tag:: <mesh type="regular1d"> [regular1d]

   One-dimensional rectangular mesh with regular intervals.

   :Contents:

   .. xml:tag:: <axis> [in regular1d mesh]

      Specification of the horizontal axis.

      :attr required start: Position of the first point on the axis. (float [µm])
      :attr required stop: Position of the last point on the axis. (float [µm])
      :attr required num: Number of the equally distributed points along the axis. (integer)

.. xml:tag:: <mesh type="regular2d"> [regular2d]

   Two-dimensional rectangular mesh with regular intervals.

   :Contents:

   .. xml:tag:: <axis0> [in regular2d mesh]

      Specification of the horizontal axis.

      :attr required start: Position of the first point on the axis. (float [µm])
      :attr required stop: Position of the last point on the axis. (float [µm])
      :attr required num: Number of the equally distributed points along the axis. (integer)

   .. xml:tag:: <axis1> [in regular2d mesh]

      Specification of the vertical axis.

      Attributes and contents are in the same format as in :xml:tag:`<axis0> [in regular2d mesh]`.

.. xml:tag:: <mesh type="regular3d">

   Three-dimensional rectangular mesh with regular intervals.

   :Contents:

   .. xml:tag:: <axis0> [in regular3d mesh]

      Specification of the longitudinal axis.

      :attr required start: Position of the first point on the axis. (float [µm])
      :attr required stop: Position of the last point on the axis. (float [µm])
      :attr required num: Number of the equally distributed points along the axis. (integer)

   .. xml:tag:: <axis1> [in regular3d mesh]

      Specification of the transversal axis.

      Attributes and contents are in the same format as in :xml:tag:`<axis0> [in regular3d mesh]`.

   .. xml:tag:: <axis2> [in regular3d mesh]

      Specification of the vertical axis.

      Attributes and contents are in the same format as in :xml:tag:`<axis0> [in regular3d mesh]`.


Possible <generator> contents for different types and methods
-------------------------------------------------------------

.. xml:tag:: <generator type="rectilinear1d" method="divide"> [rectilinear1d, divide]

   Generator that divides each geometry object along both axes into a specified number of elements, ensuring that two adjacent do not differ in size more than twice.

   :Contents:

   .. xml:tag:: <no-gradual/> [in rectilinear1d, divide generator]

      Turn off smooth mesh step (i.e. the adjacent elements of the generated mesh may differ more than by the factor of two).

   .. xml:tag:: <prediv/> [in rectilinear1d, divide generator]

      Set number of the initial divisions of each geometry object.

      :attr by: Number of parts each object is divided into along horizontal axis.

   .. xml:tag:: <postdiv/> [in rectilinear1d, divide generator]

      Set number of the final divisions of each geometry object.

      :attr by: Number of parts each object is divided into along horizontal axis.

   .. xml:tag:: <refinements> [in rectilinear1d, divide generator]

      Specify list of additional refinements of the generated mesh.

      :Contents:

      .. xml:tag:: <axis0/> [in rectilinear1d, divide generator]

         Add refinement to the horizontal axis.

         :attr required object: Name of the geometry object to add additional division to.
         :attr path: Path name, specifying particular instance of the object given in the object attribute.
         :attr at: If this attribute is present, a single refinement line is placed at the position specified in it (in the local object coordinates).
         :attr by: If this attribute is present, multiple refinement lines are placed dividing the object into a specified number of equal parts.
         :attr every: If this attribute is present, multiple refinement lines are places at distance from each other specified in the attribute value.

         Exactly one of ``at``, ``by``, or ``every`` attribute must be present.

   .. xml:tag:: <warnings/>

      Control printing of the warnings.

      :attr missing: Warn if any refinement references to non-existing object. Defaults to true. (boolean)
      :attr multiple: Warn if any refinement references to multiple objects. Defaults to true. (boolean)
      :attr outside: Warn if refining line lies outside of the specified object. Defaults to true. (boolean)

.. xml:tag:: <generator type="rectilinear1d" method="simple"> [rectilinear1d, simple]

   Simple generator creating the rectangular rectilinear mesh lines at the edges of bounding box of each object of the geometry. This generator has no configuration.

.. xml:tag:: <generator type="rectilinear2d" method="divide"> [rectilinear1d, divide]

   Generator that divides each geometry object along both axes into a specified number of elements, ensuring that two adjacent do not differ in size more than twice.

   :Contents:

   .. xml:tag:: <no-gradual/> [in rectilinear2d, divide generator]

      Turn off smooth mesh step (i.e. the adjacent elements of the generated mesh may differ more than by the factor of two).

   .. xml:tag:: <prediv/> [in rectilinear2d, divide generator]

      Set number of the initial divisions of each geometry object.

      :attr by0: Number of parts each object is divided into along horizontal axis.
      :attr by1: Number of parts each object is divided into along vertical axis.
      :attr by: Set values of ``by0`` and ``by1`` both at once. It this attribute is specified, no other ones are allowed.

   .. xml:tag:: <postdiv/> [in rectilinear2d, divide generator]

      Set number of the final divisions of each geometry object.

      It has same attributes as :xml:tag:`<prediv/> [in rectilinear2d, divide generator]`.

   .. xml:tag:: <refinements> [in rectilinear2d, divide generator]

      Specify list of additional refinements of the generated mesh.

      :Contents:

      .. xml:tag:: <axis0/> [in rectilinear2d, divide generator]

         Add refinement to the horizontal axis.

         :attr required object: Name of the geometry object to add additional division to.
         :attr path: Path name, specifying particular instance of the object given in the object attribute.
         :attr at: If this attribute is present, a single refinement line is placed at the position specified in it (in the local object coordinates).
         :attr by: If this attribute is present, multiple refinement lines are placed dividing the object into a specified number of equal parts.
         :attr every: If this attribute is present, multiple refinement lines are places at distance from each other specified in the attribute value.

         Exactly one of ``at``, ``by``, or ``every`` attribute must be present.

      .. xml:tag:: <axis1/> [in rectilinear2d, divide generator]

         Add refinement to the vertical axis.

         It has same attributes as :xml:tag:`<axis0/> [in rectilinear2d, divide generator]`.

   .. xml:tag:: <warnings/>

      Control printing of the warnings.

      :attr missing: Warn if any refinement references to non-existing object. Defaults to true. (boolean)
      :attr multiple: Warn if any refinement references to multiple objects. Defaults to true. (boolean)
      :attr outside: Warn if refining line lies outside of the specified object. Defaults to true. (boolean)

.. xml:tag:: <generator type="rectilinear2d" method="simple"> [rectilinear2d, simple]

   Simple generator creating the rectangular rectilinear mesh lines at the edges of bounding box of each object of the geometry. This generator has no configuration.

.. xml:tag:: <generator type=”rectilinear3d” method=”divide”> [rectilinear3d, divide]

   Generator that divides each geometry object along both axes into a specified number of elements, ensuring that two adjacent do not differ in size more than twice.

   :Contents:

   .. xml:tag:: <no-gradual/> [in rectilinear3d, divide generator]

      Turn off smooth mesh step (i.e. the adjacent elements of the generated mesh may differ more than by the factor of two).

   .. xml:tag:: <prediv/> [in rectilinear3d, divide generator]

      Set number of the initial divisions of each geometry object.

      :attr by0: Number of parts each object is divided into along longitudinal axis.
      :attr by1: Number of parts each object is divided into along trnasverse axis.
      :attr by2: Number of parts each object is divided into along vertical axis.
      :attr by: Set values of ``by0``, ``by1`` and ``by2`` at once. It this attribute is specified, no other ones are allowed.

   .. xml:tag:: <postdiv/> [in rectilinear3d, divide generator]

      Set number of the final divisions of each geometry object.

      It has same attributes as :xml:tag:`<prediv/> [in rectilinear3d, divide generator]`.

   .. xml:tag:: <refinements> [in rectilinear3d, divide generator]

      Specify list of additional refinements of the generated mesh.

      :Contents:

      .. xml:tag:: <axis0/> [in rectilinear3d, divide generator]

         Add refinement to the longitudinal axis.

         :attr required object: Name of the geometry object to add additional division to.
         :attr path: Path name, specifying particular instance of the object given in the object attribute.
         :attr at: If this attribute is present, a single refinement line is placed at the position specified in it (in the local object coordinates).
         :attr by: If this attribute is present, multiple refinement lines are placed dividing the object into a specified number of equal parts.
         :attr every: If this attribute is present, multiple refinement lines are places at distance from each other specified in the attribute value.

         Exactly one of ``at``, ``by``, or ``every`` attribute must be present.

      .. xml:tag:: <axis1/> [in rectilinear3d, divide generator]

         Add refinement to the transverse axis.

         It has same attributes as :xml:tag:`<axis0/> [in rectilinear3d, divide generator]`.

      .. xml:tag:: <axis2/> [in rectilinear3d, divide generator]

         Add refinement to the vertical axis.

         It has same attributes as :xml:tag:`<axis0/> [in rectilinear3d, divide generator]`.

   .. xml:tag:: <warnings/>

      Control printing of the warnings.

      :attr missing: Warn if any refinement references to non-existing object. Defaults to true. (boolean)
      :attr multiple: Warn if any refinement references to multiple objects. Defaults to true. (boolean)
      :attr outside: Warn if refining line lies outside of the specified object. Defaults to true. (boolean)

.. xml:tag:: <generator type="rectilinear3d" method="simple"> [rectilinear3d, simple]

   Simple generator creating the rectangular rectilinear mesh lines at the edges of bounding box of each object of the geometry. This generator has no configuration.
