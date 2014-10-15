Optical solvers
---------------

.. xml:tag:: <optical solver="EffectiveIndex2D"> [EffectiveIndex2D]

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

         :attr polarization: Longitudnally-propagating mode polatization. (``TE`` or ``TM``)
         :attr wavelength: Mode wavelength. (float [nm])
         :attr vat: Horizontal position of at which the vertical part of the field is calculated. (float [µm])

      .. xml:tag:: <root> [in EffectiveIndex2D optical solver]

         Parameters of the global root-finding algorithm.

         :attr method: Root finding method (‘muller’ or ‘broyden’)
         :attr tolx: Maximum change of the effective index which is allowed for convergent solution. (float [-])
         :attr tolf-min: Minimum value of the determinant sufficient to assume convergence. (float [a.u.])
         :attr tolf-max: Maximum value of the determinant required to assume convergence. (float [a.u.])
         :attr maxstep: Maximum step in one iteration of root finding (Broyden method only). (float [a.u.])
         :attr maxiter: Maximum number of root finding iterations. (int)
         :attr alpha: Parameter ensuring sufficient decrease of determinant in each step (Broyden method only). (float [a.u.])
         :attr lambda: Minimum decrease ratio of one step (Broyden method only). (float [a.u.])
         :attr initial-range: Initial range size (Muller method only).  (complex [a.u.])

      .. xml:tag:: <stripe-root> [in EffectiveIndex2D optical solver]

         Parameters of root-finding algorithm for one stripe.

         It has same attributes as :xml:tag:`<root> [in EffectiveIndex2D optical solver]`.

      .. xml:tag:: <mirrors> [in EffectiveIndex2D optical solver]

         Mirror losses.

         :attr required R1: Reflectivity of the first mirror. (float [-])
         :attr required R2: Reflectivity of the second mirror. (float [-])

      .. xml:tag:: <outer> [in EffectiveIndex2D optical solver]

         Configuration of handling area outside of the computational domain.

         :attr required dist: Distance from the computational domain boundaries where material for the outermost layer is sampled. (float [µm])



.. xml:tag:: <optical solver="EffectiveFrequencyCyl"> [EffectiveFrequencyCyl]

   Corresponding Python class: :py:class:`optical.effective.EffectiveFrequencyCyl`.

   Scalar optical solver based on the effective frequency method.

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
         :attr vat: Radial position of at which the vertical part of the field is calculated. (float or ``none``)

         Attributes ``lam0`` and ``k0`` are mutually exclusive.

      .. xml:tag:: <root> [in EffectiveFrequencyCyl optical solver]

         Parameters of the global root-finding algorithm.

         :attr method: Root finding method (‘muller’ or ‘broyden’)
         :attr tolx: Maximum change of the effective frequency parameter which is allowed for convergent solution. (float [-])
         :attr tolf-min: Minimum value of the determinant sufficient to assume convergence. (float [a.u.])
         :attr tolf-max: Maximum value of the determinant required to assume convergence. (float [a.u.])
         :attr maxstep: Maximum step in one iteration of root finding (Broyden method only). (float [a.u.])
         :attr maxiter: Maximum number of root finding iterations. (integer)
         :attr alpha: Parameter ensuring sufficient decrease of determinant in each step (Broyden method only).  (float [a.u.])
         :attr lambda: Minimum decrease ratio of one step (Broyden method only). (float [a.u.])
         :attr initial-range: Initial range size (Muller method only).  (complex [a.u.])

      .. xml:tag:: <stripe-root> [in EffectiveFrequencyCyl optical solver]

         Parameters of root-finding algorithm for one stripe.

         It has same attributes as :xml:tag:`<root> [in EffectiveFrequencyCyl optical solver]`.

      .. xml:tag:: <outer> [in EffectiveFrequencyCyl optical solver]

         Configuration of handling area outside of the computational domain.

         :attr required dist: Distance from the computational domain boundaries where material for the outermost layer is sampled. (float [µm])



.. xml:tag:: <optical solver="Fourier2D"> [Fourier2D]

   Corresponding Python class: :py:class:`optical.slab.Fourier2D`.

   Scalar optical solver based on the effective index method.

   .. xml:contents::

      .. xml:tag:: <geometry/> [in Fourier2D optical solver]

         Geometry for use by this solver.

         :attr required ref: Name of the geometry defined in the :xml:tag:`<geometry>` section.

      .. xml:tag:: <expansion> [in Fourier2D optical solver]

         Details on Fourier expansion used in computations

         :attr size: Expansion size. (integer)
         :attr refine: Number of refinement points for refractive index averaging. (integer)
         :attr smooth: Smoothing parameter for material boundaries (increases convergence). (float)
         :attr group-layers: Should similar layers be grouped for better performance. (true)

      .. xml:tag:: <interface> [in Fourier2D optical solver]

         Matching interface position in the stack.

         :attr index: Index of the layer, below which the interface is located. (integer)
         :attr position: Interface will be located as close as possible to the vertical coordinate specified in this attribute. (float)
         :attr object: Name of the geometry object below which the interface is located.
         :attr path: Optional path name, specifying particular instance of the object given in the object attribute.

         Attributes ``index``, ``position``, or ``object``/``path`` pair are mutually exclusive.

      .. xml:tag:: <vpml> [in Fourier2D optical solver]

         Vertical absorbing perfectly matched layer boundary conditions parameters.

         :attr factor: PML scaling factor. (complex)
         :attr shift: PML shift from the structure. (float [µm])
         :attr size: PML size. (float [µm])

      .. xml:tag:: <pml> [in Fourier2D optical solver]

         Side absorbing perfectly matched layer boundary conditions parameters.

         :attr factor: PML scaling factor. (complex)
         :attr order: PML shape order (0 → flat, 1 → linearly increasing, 2 → quadratic, etc.). (float)
         :attr shift: PML shift from the structure. (float [µm])
         :attr size: PML size. (float [µm])

      .. xml:tag:: <mode> [in Fourier2D optical solver]

         Mode properties.

         :attr wavelength: Light wavelength. (float [nm])
         :attr k-tran: Transverse wave-vector component. (float [1/µm])
         :attr k-long: Longitudinal wave-vector component. (float [1/µm])
         :attr symmetry: Mode symmetry. Give a symmetric field component here (e.g. ``Etran``, ``Hx``).
         :attr polarization: Mode polarization. Give an existing field component here (e.g. ``Etran``, ``Hx``).

      .. xml:tag:: <root> [in Fourier2D optical solver]

         Parameters of the global root-finding algorithm.

         :attr method: Root finding method (‘muller’ or ‘broyden’)
         :attr tolx: Tolerance on effective index. (float [-])
         :attr tolf-min: Minimum value of the determinant sufficient to assume convergence. (float [a.u.])
         :attr tolf-max: Maximum value of the determinant required to assume convergence. (float [a.u.])
         :attr maxstep: Maximum step in one iteration of root finding. (float [a.u.])
         :attr maxiter: Maximum number of root finding iterations. (integer)
         :attr alpha: Parameter ensuring sufficient decrease of determinant in each step (Broyden method only).  (float [a.u.])
         :attr lambda: Minimum decrease ratio of one step (Broyden method only). (float [a.u.])
         :attr initial-range: Initial range size (Muller method only).  (complex [a.u.])

      .. xml:tag:: <mirrors> [in Fourier2D optical solver]

         Mirror losses.

         :attr required R1: Reflectivity of the first mirror. (float [-])
         :attr required R2: Reflectivity of the second mirror. (float [-])

      .. xml:tag:: <outer> [in Fourier2D optical solver]

         Configuration of handling area outside of the computational domain.

         :attr required dist: Distance from the computational domain boundaries where material for the outermost layer is sampled. (float [µm])



.. xml:tag:: <optical solver="Fourier3D"> [Fourier3D]

   Corresponding Python class: :py:class:`optical.slab.Fourier3D`.

   Scalar optical solver based on the effective index method.

   .. xml:contents::

      .. xml:tag:: <geometry/> [in Fourier3D optical solver]

         Geometry for use by this solver.

         :attr required ref: Name of the geometry defined in the :xml:tag:`<geometry>` section.

      .. xml:tag:: <expansion> [in Fourier3D optical solver]

         Details on Fourier expansion used in computations

         :attr size: Expansion size along longitudinal and transverse axes. (one or two integers)
         :attr size-long: Expansion size along longitudinal axis. You cannot use this attribute together with :attr:`size`. (integer)
         :attr size-tran: Expansion size along transverse axis. You cannot use this attribute together with :attr:`size`. (integer)
         :attr refine: Number of refinement points for refractive index averaging along longitudinal and transverse axes. (one or two integers)
         :attr refine-long: Number of refinement points for refractive index averaging along longitudinal axis. You cannot use this attribute together with :attr:`refine`. (integer)
         :attr refine-tran: Number of refinement points for refractive index averaging along transverse axis. You cannot use this attribute together with :attr:`refine`. (integer)
         :attr smooth: Smoothing parameter for material boundaries (increases convergence). (float)
         :attr group-layers: Should similar layers be grouped for better performance. (true)

      .. xml:tag:: <interface> [in Fourier3D optical solver]

         Matching interface position in the stack.

         :attr index: Index of the layer, below which the interface is located. (integer)
         :attr position: Interface will be located as close as possible to the vertical coordinate specified in this attribute. (float)
         :attr object: Name of the geometry object below which the interface is located.
         :attr path: Optional path name, specifying particular instance of the object given in the object attribute.

         Attributes ``index``, ``position``, or ``object``/``path`` pair are mutually exclusive.

      .. xml:tag:: <vpml> [in Fourier3D optical solver]

         Vertical absorbing perfectly matched layer boundary conditions parameters.

         :attr factor: PML scaling factor. (complex)
         :attr shift: PML shift from the structure. (float [µm])
         :attr size: PML size. (float [µm])

      .. xml:tag:: <pmls> [in Fourier3D optical solver]

         Side absorbing perfectly matched layer boundary conditions parameters. Adding attributes to this tag sets PML parameters for both longitudinal and transverse boundary.

         :attr factor: PML scaling factor. (complex)
         :attr order: PML shape order (0 → flat, 1 → linearly increasing, 2 → quadratic, etc.). (float)
         :attr shift: PML shift from the structure. (float [µm])
         :attr size: PML size. (float [µm])

         .. xml:contents::

            .. xml:tag:: <long> [in Fourier3D optical solver <pmls>]

               Set parameters of PMLs in longitudinal direction. The attributes are the same as above.

            .. xml:tag:: <tran> [in Fourier3D optical solver <pmls>]

               Set parameters of PMLs in transverse direction. The attributes are the same as above.

      .. xml:tag:: <mode> [in Fourier3D optical solver]

         Mode properties.

         :attr wavelength: Light wavelength. (float [nm])
         :attr k-tran: Transverse wave-vector component. (float [1/µm])
         :attr k-long: Longitudinal wave-vector component. (float [1/µm])
         :attr symmetry: Mode symmetry. Give a symmetric field component here (e.g. ``Etran``, ``Hx``).

      .. xml:tag:: <root> [in Fourier3D optical solver]

         Parameters of the root-finding algorithm.

         :attr method: Root finding method (‘muller’ or ‘broyden’)
         :attr tolx: Tolerance on effective index. (float [-])
         :attr tolf-min: Minimum value of the determinant sufficient to assume convergence. (float [a.u.])
         :attr tolf-max: Maximum value of the determinant required to assume convergence. (float [a.u.])
         :attr maxstep: Maximum step in one iteration of root finding. (float [a.u.])
         :attr maxiter: Maximum number of root finding iterations. (integer)
         :attr alpha: Parameter ensuring sufficient decrease of determinant in each step (Broyden method only).  (float [a.u.])
         :attr lambda: Minimum decrease ratio of one step (Broyden method only). (float [a.u.])
         :attr initial-range: Initial range size (Muller method only).  (complex [a.u.])

      .. xml:tag:: <outer> [in Fourier3D optical solver]

         Configuration of handling area outside of the computational domain.

         :attr required dist: Distance from the computational domain boundaries where material for the outermost layer is sampled. (float [µm])

