Thermal solvers
---------------

.. xml:tag:: <thermal solver="Static2D"> [Static2D]

   Corresponding Python class: :py:class:`thermal.fem.Static2D`.

   Two-dimensional static thermal solver in Cartesian geometry, based on finite-element method.

   .. xml:contents::

      .. xml:tag:: <geometry/> [in Static2D thermal solver]

         Geometry for use by this solver.

         :attr required ref: Name of the geometry defined in the :xml:tag:`<geometry>` section.

      .. xml:tag:: <mesh/> [in Static2D thermal solver]

         Mesh used by this solver.

         :attr required ref: Name of the mesh defined in the :xml:tag:`<grids>` section.

      .. xml:tag:: <loop/> [in Static2D thermal solver]

         Configuration of the self-consistent loop.

         :attr inittemp: Initial temperature. (float [K])
         :attr maxerr: Maximum allowed error. (float [K])

      .. xml:tag:: <matrix/> [in Static2D thermal solver]

         Configuration of the matrix solver.

         :attr algorithm: Solution algorithm. Defaults to ``cholesky``. (``cholesky``, ``gauss``, or ``iterative``)
         :attr itererr: Maximum allowed residual error for the iterative algorithm.
         :attr iterlim: Maximum number of iterations for the iterative algorithm.
         :attr logfreq: Frequency of logging iterative solver progress.
         .. :attr preconditioner: Preconditioner for the iterative (conjugate gradient) algorithm. (``jacobi`` or ``factor``)

      .. xml:tag:: <temperature> [in Static2D thermal solver]

         Boundary conditions: constant temperature. See subsection :ref:`sec-xpl-Boundary-conditions`.

      .. xml:tag:: <heatflux> [in Static2D thermal solver]

         Boundary conditions: constant heat flux. See subsection :ref:`sec-xpl-Boundary-conditions`.

      .. xml:tag:: <convection> [in Static2D thermal solver]

         Boundary conditions: convection. See subsection :ref:`sec-xpl-Boundary-conditions`.

         This boundary condition does not have ``value`` attribute. Use ``coeff`` for convection coefficient and ``ambient`` for ambient temperature instead.

      .. xml:tag:: <radiation> [in Static2D thermal solver]

         Boundary conditions: radiation. See subsection :ref:`sec-xpl-Boundary-conditions`.

         This boundary condition does not have ``value`` attribute. Use ``emissivity`` for surface emissivity and ``ambient`` for ambient temperature instead.

.. xml:tag:: <thermal solver="StaticCyl"> [StaticCyl]

      Corresponding Python class: :py:class:`thermal.fem.StaticCyl`.

      Two-dimensional static thermal solver in cylindrical geometry, based on finite-element method.

      .. xml:contents::

          See :xml:tag:`<thermal solver="Static2D"> [Static2D]`.

.. xml:tag:: <thermal solver="Static3D"> [Static3D]

      Corresponding Python class: :py:class:`thermal.fem3d.Static3D`.

      Three-dimensional static thermal solver, based on finite-element method.

      .. xml:contents::

          See :xml:tag:`<thermal solver="Static2D"> [Static2D]`.
