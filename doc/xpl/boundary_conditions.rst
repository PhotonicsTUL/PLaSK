.. _sec-xpl-Boundary-conditions:

Boundary conditions
-------------------

Most of the solvers have some boundary conditions. They differ by name or type, but all share the same structure: some value is set at mesh points in some region (usually the edge of the whole geometry, or the edge of some geometrical object. Hence, the structure of boundary conditions is usually the same and looks as follows [#different-boundary-conditions]_:

.. xml:tag:: <boundary_conditions> []

   Some boundary conditions specifications.

   .. xml:tag:: <condition>

      Specification of one boundary condition.

      :attr required value: Value of the boundary condition. In some boundary condition value is given in different attributes. In such case, this attribute should be replaced with the specific ones.
      :attr placename: Name of the boundary condition location for further reference.
      :attr placeref: Set location of boundary conditions to some location previously named with ``placename``.
      :attr place: Set one of standard location of boundary condition. The value of this attribute depends on the mesh.

      .. xml:contents::

         .. xml:tag:: <place>

            Set location of boundary condition. This tag can be used instead of the ``place`` attribute if more detailed description of the boundary condition location is required. Its attributes are mesh-specific. Below there are most common examples of attribute sets for rectangular meshes:

            **Boundary conditions at the side of some object:**

            :attr required object: Name of the geometry object to set boundary conditions at.
            :attr path: Path name, specifying particular instance of the object given in the object attribute.
            :attr required side: Side of the object to set boundary conditions at. (``left``, ``right``, ``top``, ``bottom``, ``back``, ``front``, ``all`` [supported by Triangular2D mesh only])

            **Boundary conditions at some line (2D meshes) [supported by rectangular meshes only]:**

            :attr required line: Direction of the line. (``vertical`` or ``horizontal``)
            :attr required at: Location of the line i.e. its position on the perpendicular axis.
            :attr required start: Position of the start of the line on the parallel axis.
            :attr required stop: Position of the end of the line on the parallel axis.

.. rubric:: Footnotes

.. [#different-boundary-conditions] In some cases where structure of boundary conditions description is different, it is shown in the reference of particular solver.

