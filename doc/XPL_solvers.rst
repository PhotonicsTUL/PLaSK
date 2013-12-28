.. _sec-XPL-solvers:

Section <solvers>
=================

.. xml:tag:: <solvers>

In this section used computational solvers are defined and configured. Also here, the :ref:`data filters <sec-data-filters>` are set-up, as in general, they are only special kinds of solvers. Each XML element in this section correspond to separate solver. The content of such element depends strongly on particular solver, while its name and attributes are standard (although there are differences in attributes of strict computational solvers and :ref:`data filters <sec-data-filters>`). The details of XML content of this section is presented below.

The computational solvers are declared with an XML tag, which name is the category of the solver, e.g. *thermal*, *electrical*, *gain*, or *optical* and that has the standard set of attributes:

.. xml:tag:: <category> []

   Definition of computational solver.

   :attr required name: Solver name. In Python script there is a automatically created solver object with such name. (identifier string)
   :attr required solver: Actual solver type. In Python script this defines the class of the solver object.
   :attr lib: Library in which this solver is implemented. For most standard solvers, PLaSK can automatically determine its proper value. For other solver types this attribute is required.

   .. xml.. xml:contents::

       The contents of each solver depends on the category and the solver type (i.e. the tag name and the value of the solver attribute). It is specified in the following subsections.


.. _sec-XPL-Boundary-conditions:

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
            :attr required side: Side of the object to set boundary conditions at. (``left``, ``right``, ``top``, ``bottom``, ``back``, ``front``)

            **Boundary conditions at some line (2D meshes):**

            :attr required line: Direction of the line. (``vertical`` or ``horizontal``)
            :attr required at: Location of the line i.e. its position on the perpendicular axis.
            :attr required start: Position of the start of the line on the parallel axis.
            :attr required stop: Position of the end of the line on the parallel axis.


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
         :attr itererr: Allowed residual error for the iterative algorithm.
         :attr iterlim: Maximum number of iterations for the iterative algorithm.
         :attr logfreq: Frequency of logging iterative solver progress.
         .. :attr preconditioner: Preconditioner for the iterative (conjugate gradient) algorithm. (``jacobi`` or ``factor``)

      .. xml:tag:: <temperature> [in Static2D thermal solver]

         Boundary conditions: constant temperature. See subsection :ref:`sec-XPL-Boundary-conditions`.

      .. xml:tag:: <heatflux> [in Static2D thermal solver]

         Boundary conditions: constant heat flux. See subsection :ref:`sec-XPL-Boundary-conditions`.

      .. xml:tag:: <convection> [in Static2D thermal solver]

         Boundary conditions: convection. See subsection :ref:`sec-XPL-Boundary-conditions`.

         This boundary condition does not have ``value`` attribute. Use ``coeff`` for convection coefficient and ``ambient`` for ambient temperature instead.

      .. xml:tag:: <radiation> [in Static2D thermal solver]

         Boundary conditions: radiation. See subsection :ref:`sec-XPL-Boundary-conditions`.

         This boundary condition does not have ``value`` attribute. Use ``emissivity`` for surface emissivity and ``ambient`` for ambient temperature instead.

.. xml:tag:: <thermal solver="StaticCyl"> [StaticCyl]

      Corresponding Python class: :py:class:`thermal.fem.StaticCyl`.

      Two-dimensional static thermal solver in cylindrical geometry, based on finite-element method.

      .. xml.. xml:contents::

          See :xml:tag:`<thermal solver="Static2D"> [Static2D]`.

.. xml:tag:: <thermal solver="Static3D"> [Static3D]

      Corresponding Python class: :py:class:`thermal.fem3d.Static3D`.

      Three-dimensional static thermal solver, based on finite-element method.

      .. xml.. xml:contents::

          See :xml:tag:`<thermal solver="Static2D"> [Static2D]`.


Electrical solvers
------------------

.. xml:tag:: <electrical solver="Shockley2D"> [Shockley2D]

   Corresponding Python class: :py:class:`electrical.fem.Shockley2D`.

   Two-dimensional phenomenological solver in Cartesian geometry, based on finite-element method.

   .. xml:contents::

      .. xml:tag:: <geometry/> [in Shockley2D electrical solver]

         Geometry for use by this solver.

         :attr required ref: Name of the geometry defined in the :xml:tag:`<geometry>` section.

      .. xml:tag:: <mesh/> [in Shockley2D electrical solver]

         Mesh used by this solver.

         :attr required ref: Name of the mesh defined in the :xml:tag:`<grids>` section.

      .. xml:tag:: <loop/> [in Shockley2D electrical solver]

         Configuration of the self-consistent loop.

         :attr maxerr: Maximum allowed error. (float [%])

      .. xml:tag:: <matrix/> [in Shockley2D electrical solver]

         Configuration of the matrix solver.

         :attr algorithm: Solution algorithm. Defaults to ``cholesky``. (``cholesky``, ``gauss``, or ``iterative``)
         :attr itererr: Allowed residual error for the iterative algorithm.
         :attr iterlim: Maximum number of iterations for the iterative algorithm.
         :attr logfreq: Frequency of logging iterative solver progress.
         .. :attr preconditioner: Preconditioner for the iterative (conjugate gradient) algorithm. (``jacobi`` or ``factor``)

      .. xml:tag:: <junction/> [in Shockley2D electrical solver]

         Configuration of the effective model of p-n junction.

         :attr js: Reverse bias current density. (float [A/m\ :sup:`2`])
         :attr Shockley: Junction coefficient.
         :attr pnjcond: Initial vertical conductivity of the junction. (float [S/m])
         :attr heat: Method of determination of the heat generated in the junction. (``joules`` or ``wavelength``)
         :attr wavelength: Emitted wavelength if ``heat`` is set to ``wavelength``.

      .. xml:tag:: <contacts/> [in Shockley2D electrical solver]

         Properties of the contacts.

         :attr pcond: p-contact conductivity. (float [S/m])
         :attr ncond: n-contact conductivity. (float [S/m])

      .. xml:tag:: <voltage> [in Shockley2D electrical solver]

         Boundary conditions: electric potential. See subsection :ref:`sec-XPL-Boundary-conditions`.

.. xml:tag:: <electrical solver="ShockleyCyl"> [ShockleyCyl]

      Corresponding Python class: :py:class:`electrical.fem.ShockleyCyl`.

      Two-dimensional phenomenological solver in cylindrical geometry, based on finite-element method.

      .. xml.. xml:contents::

          See :xml:tag:`<electrical solver="Shockley2D"> [Shockley2D]`.

.. xml:tag:: <electrical solver="Shockley3D"> [Shockley3D]

      Corresponding Python class: :py:class:`electrical.fem3d.Shockley3D`.

      Three-dimensional phenomenological solver in Cartesian geometry, based on finite-element method.

      .. xml.. xml:contents::

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

      .. xml.. xml:contents::

          See :xml:tag:`<electrical solver="Diffusion2D"> [Diffusion2D]`.


Gain solvers
------------

.. xml:tag:: <gain solver="Fermi2D"> [Fermi2D]

   Corresponding Python class: :py:class:`gain.simple.Fermi2D`.

   Simple gain solver based on Fermi Golden Rule for two-dimensional Cartesian geometry.

   .. xml:contents::

      .. xml:tag:: <geometry/> [in Fermi2D gain solver]

         Geometry for use by this solver.

         :attr required ref: Name of the geometry defined in the :xml:tag:`<geometry>` section.

      .. xml:tag:: <mesh/> [in Fermi2D gain solver]

         Optional mesh used by this solver. If it is set then the gain is computed only in the mesh points and interpolated in-between. Otherwise, the full gain calculation is performed in each requested point.

         :attr required ref: Name of the mesh defined in the :xml:tag:`<grids>` section.

      .. xml:tag:: <config/> [in Fermi2D gain solver]

         Configuration of the self-consistent loop.

         :attr lifetime: Carriers lifetime.
         :attr matrix-elem: Value of the matrix element in gain computations (if not set it is estimated automatically).

      .. xml:tag:: <levels/> [in Fermi2D gain solver]

         Custom energy levels in quantum wells. If this tag is used all levels must be set.

         :attr required le: Comma-separated list of electron levels.
         :attr required hh: Comma-separated list of heavy hole levels.
         :attr required lh: Comma-separated list of light hole levels.

.. xml:tag:: <gain solver="FermiCyl"> [FermiCyl]

   Corresponding Python class: :py:class:`gain.simple.FermiCyl`.

   Simple gain solver based on Fermi Golden Rule for two-dimensional cylindrical geometry.

   .. xml.. xml:contents::

       See :xml:tag:`<gain solver="Fermi2D"> [Fermi2D]`.


Optical solvers
---------------

.. xml:tag:: <optical solver=”EffectiveIndex2D”> [EffectiveIndex2D]

   Corresponding Python class: :py:class:`optical.effective.EffectiveIndex2D`.

   Scalar optical solver based on effective index method.

   .. xml.. xml:contents::

      .. xml:tag:: <geometry/> [in EffectiveIndex2D optical solver]

         Geometry for use by this solver.

         :attr required ref: Name of the geometry defined in the :xml:tag:`<geometry>` section.

      .. xml:tag:: <mesh/> [in EffectiveIndex2D optical solver]

         Mesh used by this solver.

         :attr required ref: Name of the mesh defined in the :xml:tag:`<grids>` section.

      .. xml:tag:: <mode> [in EffectiveIndex2D optical solver]

         Mode properties.

         :attr polarization: Light polatization. (``TE`` or ``TM``)
         :attr symmetry: Mode symmetry with respect to vertical symmetry axis (if present). (``none``, ``positive``, or ``negative``)
         :attr wavelength: Light wavelength. (float [nm])

      .. xml:tag:: <root> [in EffectiveIndex2D optical solver]

         Parameters of the global root-finding algorithm.

         :attr tolx: Tolerance on effective index. (float [-])
         :attr tolf-min: Minimum value of the determinant sufficient to assume convergence. (float [a.u.])
         :attr tolf-max: Maximum value of the determinant required to assume convergence. (float [a.u.])
         :attr maxstep: Maximum step in one iteration of root finding. (float [-])
         :attr maxiter: Maximum number of root finding iterations. (integer)

      .. xml:tag:: <stripe-root> [in EffectiveIndex2D optical solver]

         Parameters of root-finding algorithm for one stripe.

         It has same attributes as :xml:tag:`<root> [in EffectiveIndex2D optical solver]`.

      .. xml:tag:: <mirrors> [in EffectiveIndex2D optical solver]

         Mirror losses.

         :attr required R1: Reflectivity of the first mirror. (float [-])
         :attr required R2: Reflectivity of the second mirror. (float [-])

      .. xml:tag:: <outer> [in EffectiveIndex2D optical solver]

         Configuration of handling area outside of the computational domain.

         :attr required distance: Distance from the computational domain boundaries where material for the outermost layer is sampled. (float [µm])

.. xml:tag:: <optical solver=”EffectiveFrequencyCyl”> [EffectiveFrequencyCyl]

   Corresponding Python class: :py:class:`optical.effective.EffectiveFrequencyCyl`.

   Scalar optical solver based on effective index method.

   .. xml.. xml:contents::

      .. xml:tag:: <geometry/> [in EffectiveFrequencyCyl optical solver]

         Geometry for use by this solver.

         :attr required ref: Name of the geometry defined in the :xml:tag:`<geometry>` section.

      .. xml:tag:: <mesh/> [in EffectiveFrequencyCyl optical solver]

         Mesh used by this solver.

         :attr required ref: Name of the mesh defined in the :xml:tag:`<grids>` section.

      .. xml:tag:: <mode> [in EffectiveFrequencyCyl optical solver]

         Mode properties.

         :attr lam0: Approximate wavelength. (float [nm])
         :attr k0: Approximate normalized frequency. (float [1/µm])
         :attr emission: Direction of emission, necessary for over-threshold power computations (``top`` or ``bottom``)
         :attr vlam: "vertical wavelength" i.e. the wavelength what would be in the absence of lateral confinement; setting this value helps to find models in very long resonators (float [nm])

         Attributes ``lam0`` and ``k0`` are mutually exclusive.

      .. xml:tag:: <root> [in EffectiveFrequencyCyl optical solver]

         Parameters of the global root-finding algorithm.

         :attr tolx: Tolerance on effective index. (float [-])
         :attr tolf-min: Minimum value of the determinant sufficient to assume convergence. (float [a.u.])
         :attr tolf-max: Maximum value of the determinant required to assume convergence. (float [a.u.])
         :attr maxstep: Maximum step in one iteration of root finding. (float [-])
         :attr maxiter: Maximum number of root finding iterations. (integer)

      .. xml:tag:: <stripe-root> [in EffectiveFrequencyCyl optical solver]

         Parameters of root-finding algorithm for one stripe.

         It has same attributes as :xml:tag:`<root> [in EffectiveFrequencyCyl optical solver]`.

      .. xml:tag:: <outer> [in EffectiveFrequencyCyl optical solver]

         Configuration of handling area outside of the computational domain.

         :attr required distance: Distance from the computational domain boundaries where material for the outermost layer is sampled. (float [µm])


.. _sec-data-filters:

Data filters
------------

.. xml:tag:: <filter/>

   Filter is a special kind of solver which "solves" the problem using another solvers.

   It calculates its output using input of similar type and changing it in some way,
   for example translating it from one space to another (2D -> 3D, 3D -> 2D, etc.).

   Typically filter has one or more inputs and one output (output provider, named ``out``).

   :attr required name: Solver (filter) name. In Python script there is a automatically created solver object with such name. (identifier string)
   :attr required for: name of property type that this filter will pass (``out`` will provide data of this type), e.g.: ``temperature``.
   :attr required geometry: Name of the geometry defined in the :xml:tag:`<geometry>` section. Filter will provide data in coordinates of given geometry.

   Some information about connecting filters with solvers are in :xml:tag:`connects` sections.




