Electrical solvers
------------------

.. xml:tag:: <electrical solver="Shockley2D"> [Shockley2D]

   Corresponding Python class: :py:class:`electrical.fem.Shockley2D`.

   Two-dimensional phenomenological solver in Cartesian geometry based Shockley equation and using finite-element method.

   .. xml:contents::

      .. xml:tag:: <geometry/> [in Shockley2D electrical solver]

         Geometry for use by this solver.

         :attr required ref: Name of the geometry defined in the :xml:tag:`<geometry>` section.

      .. xml:tag:: <mesh/> [in Shockley2D electrical solver]

         Mesh used by this solver.

         :attr required ref: Name of the mesh defined in the :xml:tag:`<grids>` section.

      .. xml:tag:: <loop/> [in Shockley2D electrical solver]

         Configuration of the self-consistent loop.

         :attr maxerr: Maximum allowed current density error. (float [%])

      .. xml:tag:: <matrix/> [in Shockley2D electrical solver]

         Configuration of the matrix solver.

         :attr algorithm: Solution algorithm. Defaults to ``cholesky``. (``cholesky``, ``gauss``, or ``iterative``)
         :attr itererr: Allowed residual error for the iterative algorithm.
         :attr iterlim: Maximum number of iterations for the iterative algorithm.
         :attr logfreq: Frequency of logging iterative solver progress.
         .. :attr preconditioner: Preconditioner for the iterative (conjugate gradient) algorithm. (``jacobi`` or ``factor``)

      .. xml:tag:: <junction/> [in Shockley2D electrical solver]

         Configuration of the effective model of p-n junction.

         :attr js#: Reverse bias current density. *#* can be a junction number. If the number is omitted, 0 is assumed. (float [A/m\ :sup:`2`])
         :attr beta#: Junction coefficient. This is an inverse of the junction thermal voltage. *#* can be a junction number. If the number is omitted, 0 is assumed. (float [1/V])
         :attr pnjcond: Initial vertical conductivity of the junctions. (float [S/m])
         :attr heat: Method of determination of the heat generated in the junction. (``joules`` or ``wavelength``)
         :attr wavelength: Emitted wavelength if ``heat`` is set to ``wavelength``.

      .. xml:tag:: <contacts/> [in Shockley2D electrical solver]

         Properties of the contacts.

         :attr pcond: p-contact conductivity. (float [S/m])
         :attr ncond: n-contact conductivity. (float [S/m])

      .. xml:tag:: <voltage> [in Shockley2D electrical solver]

         Boundary conditions: electric potential. See subsection :ref:`sec-xpl-Boundary-conditions`.

.. xml:tag:: <electrical solver="ShockleyCyl"> [ShockleyCyl]

      Corresponding Python class: :py:class:`electrical.fem.ShockleyCyl`.

      Two-dimensional phenomenological solver in cylindrical geometry based Shockley equation and using finite-element method.

      .. xml:contents::

          See :xml:tag:`<electrical solver="Shockley2D"> [Shockley2D]`.

.. xml:tag:: <electrical solver="Shockley3D"> [Shockley3D]

      Corresponding Python class: :py:class:`electrical.fem3d.Shockley3D`.

      Three-dimensional phenomenological solver in Cartesian geometry based Shockley equation and using finite-element method.

      .. xml:contents::

          See :xml:tag:`<electrical solver="Shockley2D"> [Shockley2D]`.

.. xml:tag:: <electrical solver="Diffusion2D"> [Diffusion2D]

   Corresponding Python class: :py:class:`electrical.diffusion.Diffusion2D`.

   Two-dimensional diffusion solver in Cartesian geometry.

   .. xml:contents::

      .. xml:tag:: <geometry/> [in Diffusion2D electrical solver]

         Geometry for use by this solver.

         :attr required ref: Name of the geometry defined in the :xml:tag:`<geometry>` section.

      .. xml:tag:: <mesh/> [in Diffusion2D electrical solver]

         One-dimensional horizontal initial mesh used by this solver.

         :attr required start: Position of the first mesh point. (float [µm])
         :attr required stop: Position of the last mesh point. (float [µm])
         :attr required num: Number of the mesh points. (integer)

      .. xml:tag:: <config/> [in Diffusion2D electrical solver]

         :attr fem-method: Order of the finite-element method. (``linear`` or ``parabolic``)
         :attr accuracy: Required relative accuracy. (float [%])
         :attr abs-accuracy: Required absolute minimal concentration accuracy. (float [cm\ :sup:`-3`])
         :attr interpolation: Current density interpolation method name.
         :attr maxiters: Maximum number of allowed iterations before attempting to refine mesh. (integer)
         :attr maxrefines: Maximum number of allowed mesh refinements. (integer)

.. xml:tag:: <electrical solver="DiffusionCyl"> [DiffusionCyl]

      Corresponding Python class: :py:class:`electrical.diffusion.DiffusionCyl`.

      Two-dimensional diffusion solver in cylindrical geometry.

      .. xml:contents::

          See :xml:tag:`<electrical solver="Diffusion2D"> [Diffusion2D]`.
