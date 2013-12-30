Optical solvers
---------------

.. xml:tag:: <optical solver=”EffectiveIndex2D”> [EffectiveIndex2D]

   Corresponding Python class: :py:class:`optical.effective.EffectiveIndex2D`.

   Scalar optical solver based on the effective index method.

   .. xml:contents::

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

   Scalar optical solver based on the effective index method.

   .. xml:contents::

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
