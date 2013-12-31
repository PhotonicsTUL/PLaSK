.. _sec-xpl-materials:

Section <materials>
===================

.. xml:tag:: <materials>

This section contains specification of custom materials that can be used together with the library materials in the structure geometry. The only allowed tag in this section — that can, however, appear more than once — is the tag ``<material>``:

.. xml:tag:: <material>

   Corresponding Python class: :py:class:`Material`.

   Definition of a custom material.

   :attr required name: Name of the material. As all custom materials are simple materials, it can be an arbitrary identifier string. However, it may also contain a doping specification without the doping amount.
   :attr kind: (either **kind** or **base** is required) Kind of the new material. Any of: *semiconductor*, *dielectric*, *oxide*, *metal*, *liquid crystal*.
   :attr base: (either **kind** or **base** is required) Textual specification of the base material. The doping amount information can be skipped from it, in which case the doping amount will have to be specified when the custom material is used.

   .. xml:contents::

      The content of this element is the list of user-defined material properties. Each element of such list is a tag specifying the particular property which content is a mathematical expression computing this property. Each such expression can use several variables: the ones specified below next to each tag and ``dc`` or ``cc`` that will contain the user specified doping amounts: dopant or carriers concentration, respectively (at most one of ``cc`` or ``dc`` is defined, never both).

      Some properties are anisotropic and can have different values for lateral and vertical components. In such case, two separate values may (but do not have to) be defined in the contents of the material property tag and they should be separated with a comma.

      The accepted material properties are as follows:

      .. xml:tag:: <A>

         Monomolecular recombination coefficient *A* [1/s].

         Variables: ``T`` — temperature [K].

      .. xml:tag:: <absb>

         Absorption coefficient *α* [cm\ :sup:`-1`].

         Variables: ``wl`` — wavelength [nm], ``T`` — temperature [K].

      .. xml:tag:: <ac>

         Hydrostatic deformation potential for the conduction band *a*\ :sub:`c` [eV].

         Variables: ``T`` — temperature [K].

      .. xml:tag:: <av>

         Hydrostatic deformation potential for the valence band *a*\ :sub:`v` [eV].

         Variables: ``T`` — temperature [K].

      .. xml:tag:: <B>

         Radiative recombination coefficient *B* [m\ :sup:`3`/s].

         Variables: ``T`` — temperature [K].

      .. xml:tag:: <b>

         Shear deformation potential *b* [eV].

         Variables: ``T`` — temperature [K].

      .. xml:tag:: <C>

         Auger recombination coefficient *C* [m\ :sup:`6`/s].

         Variables: ``T`` — temperature [K].

      .. xml:tag:: <c11>

         Elastic constant *c*\ :sub:`11` [GPa].

         Variables: ``T`` — temperature [K].

      .. xml:tag:: <c12>

         Elastic constant *c*\ :sub:`12` [GPa].

         Variables: ``T`` — temperature [K].

      .. xml:tag:: <CB>

         Conduction band level *CB* [eV].

         Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
         ``point`` — point in the Brillouin zone [-].

      .. xml:tag:: <chi>

         Electron affinity *χ* [eV].

         Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
         ``point`` — point in the Brillouin zone [-].

      .. xml:tag:: <cond>

         Electrical conductivity sigma in-plane (lateral) and cross-plane (vertical) direction [S/m].

         Variables: ``T`` — temperature [K].

      .. xml:tag:: <condtype>

         Electrical conductivity type. In semiconductors this indicates what type of carriers :xml:tag:`<Nf>` refers to.

      .. xml:tag:: <cp>

         Specific heat at constant pressure [J/(kg K)].

         Variables: ``T`` — temperature [K].

      .. xml:tag:: <D>

         Ambipolar diffusion coefficient *D* [m\ :sup:`2`/s].

         Variables: ``T`` — temperature [K].

      .. xml:tag:: <dens>

         Density [kg/m\ :sup:`3`].

         Variables: ``T`` — temperature [K].

      .. xml:tag:: <Dso>

         Split-off energy *D*\ :sub:`so` [eV].

         Variables: ``T`` — temperature [K], ``e`` — lateral strain [-].

      .. xml:tag:: <EactA>

         Acceptor ionization energy *E*\ :sub:`actA` [eV].

         Variables: ``T`` — temperature [K].

      .. xml:tag:: <EactD>

         Donor ionization energy *E*\ :sub:`actD` [eV].

         Variables: ``T`` — temperature [K].

      .. xml:tag:: <Eg>

         Energy gap *E*\ :sub:`g` [eV].

         Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
         ``point`` — point in the Brillouin zone [-].

      .. xml:tag:: <eps>

         Donor ionization energy *ε*\ :sub:`R` [-].

         Variables: ``T`` — temperature [K].

      .. xml:tag:: <lattC>

         Lattice constant [Å].

         Variables: ``T`` — temperature [K], ``x`` — lattice parameter [-].

      .. xml:tag:: <Me>

         Electron effective mass *M*\ :sub:`e` in in-plane (lateral)
         and cross-plane (vertical) direction [*m*\ :sub:`0`].

         Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
         ``point`` — point in the irreducible Brillouin zone [-].

      .. xml:tag:: <Mh>

         Hole effective mass *M*\ :sub:`h` in in-plane (lateral)
         and cross-plane (vertical) direction [*m*\ :sub:`0`].

         Variables: ``T`` — temperature [K], ``e`` — lateral strain [-].

      .. xml:tag:: <Mhh>

         Heavy hole effective mass *M*\ :sub:`hh` in in-plane (lateral)
         and cross-plane (vertical) direction [*m*\ :sub:`0`].

         Variables: ``T`` — temperature [K], ``e`` — lateral strain [-].

      .. xml:tag:: <Mlh>

         Light hole effective mass *M*\ :sub:`lh` in in-plane (lateral)
         and cross-plane (vertical) direction [*m*\ :sub:`0`].

         Variables: ``T`` — temperature [K], ``e`` — lateral strain [-].

      .. xml:tag:: <mob>

         Majority carriers mobility in-plane (lateral) and cross-plane (vertical) direction
         [m\ :sup:`2`/(V s)].

         Variables: T — temperature [K].

      .. xml:tag:: <Mso>

         Split-off mass *M*\ :sub:`so`` [*m*\ :sub:`0`].

         Variables: ``T`` — temperature [K], ``e`` — lateral strain [-].

      .. xml:tag:: <Nc>

         Effective density of states in the conduction band *N*\ :sub:`c` [cm\ :sup:`-3`].

         Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
         ``point`` — point in the Brillouin zone [-].

      .. xml:tag:: <Nf>

         Free carrier concentration *N* [cm\ :sup:`-3`].

         Variables: ``T`` — temperature [K].

      .. xml:tag:: <Ni>

         Intrinsic carrier concentration *N*\ :sub:`i` [cm\ :sup:`-3`].

         Variables: ``T`` — temperature [K].

      .. xml:tag:: <Nr>

         Complex refractive index *n*\ :sub:`R` [-].

         Variables: ``wl`` — wavelength [nm], ``T`` — temperature [K].

      .. xml:tag:: <nr>

         Real refractive index *n*\ :sub:`R` [-].

         Variables: ``wl`` — wavelength [nm], ``T`` — temperature [K].

      .. xml:tag:: <Nr-tensor>

         Anisotropic complex refractive index tensor *n*\ :sub:`R` [-].
         Tensor must have the form [ *n*\ :sub:`00`, *n*\ :sub:`11`, *n*\ :sub:`22`, *n*\ :sub:`01`, *n*\ :sub:`10` ].

         Variables: ``wl`` — wavelength [nm], ``T`` — temperature [K].

      .. xml:tag:: <Nv>

         Effective density of states in the valance band *N*\ :sub:`v` [cm\ :sup:`-3`].

         Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
         ``point`` — point in the Brillouin zone [-].

      .. xml:tag:: <thermk>

         Thermal conductivity in in-plane (lateral) and cross-plane (vertical) direction *k* [W/(m K)].

         Variables: ``T`` — temperature [K], ``h`` — layer thickness [µm].

      .. xml:tag:: <VB>

         Valance band level offset *VB* [eV].

         Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
         ``hole`` — hole type (``'H'`` or ``'L'``) [-].

