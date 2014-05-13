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

         :attr lifetime: Carriers lifetime. (float)
         :attr matrix-elem: Value of the matrix element in gain computations (if not set it is estimated automatically). (float)
         :attr strained: Boolean attribute indicated if the solver should consider strain in the active region. If set to *yes* than there must a layer with the role "\ *substrate*\ " in the geometry. The strain is computed by comparing the atomic lattice constants of the substrate and the quantum wells. (*yes* or *no*)

      .. xml:tag:: <levels/> [in Fermi2D gain solver]

         Custom energy levels in quantum wells. If this tag is used all levels must be set.

         :attr required el: Comma-separated list of electron levels. (list of floats)
         :attr required hh: Comma-separated list of heavy hole levels. (list of floats)
         :attr required lh: Comma-separated list of light hole levels. (list of floats)

.. xml:tag:: <gain solver="FermiCyl"> [FermiCyl]

   Corresponding Python class: :py:class:`gain.simple.FermiCyl`.

   Simple gain solver based on Fermi Golden Rule for two-dimensional cylindrical geometry.

   .. xml:contents::

       See :xml:tag:`<gain solver="Fermi2D"> [Fermi2D]`.
