.. _sec-xpl-materials:

Section <materials>
===================

.. xml:tag:: <materials>

This section contains specification of custom materials that can be used together with the library materials in the structure geometry. The only allowed tag in this section — that can, however, appear more than once — is the tag ``<material>``:

.. xml:tag:: <material>

   Corresponding Python class: :py:class:`Material`.

   Definition of a custom material.

   :attr required name: Name of the material. As all custom materials are simple materials, it can be an arbitrary identifier string. However, it may also contain a doping specification without the doping amount (e.g. "``myGaAs``" or "``newAlAs:Si``").
   :attr required base: Textual specification of the base material. The doping amount information can be skipped from it, in which case the doping amount will have to be specified when the custom material is used. The following bases are always available: *semiconductor*, *dielectric*, *oxide*, *metal*, *liquid_crystal*.
   :attr alloy: If you specify attribute ``alloy="yes"``, your material will be an alloy. Its name must then consist of element names with an optional custom label after the "``_``" character. For example: "``GaInN``" "``AlGaAs_custom``".

   .. xml:contents::

      The content of this element is the list of user-defined material properties. Each element of such list is a tag specifying the particular property which content is a mathematical expression computing this property. Each such expression can use several variables: the ones specified below next to each tag and ``self`` that refers to the material itself and allows to access its doping (``self.doping``) and composition in case of alloys (e.g. ``self.Ga`` for the amount of gallium). You may also access the parameters of base materials using ``super()`` function (e.g. ``super().thermk(T)``). If the expression does not use any variables, it is evaluated only once when XPL file is being loaded.

      Some properties are anisotropic and can have different values for lateral and vertical components. In such case, two separate values may (but do not have to) be defined in the contents of the material property tag and they should be separated with a comma.

      .. rubric:: Example:

      .. code-block:: xml

         <materials>
           <material name="MyMaterial" base="Semiconductor">
             <nr>3.5 + 0.01*T</nr>
             <absp>10.</absp>
           </material>
         <materials>

      The accepted material properties are as follows:

      .. xml:tag:: <A>

         Monomolecular recombination coefficient *A* (1/s).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <absb>

         Absorption coefficient *α* (cm\ :sup:`-1`).

         Variables: ``lam`` — wavelength (nm), ``T`` — temperature (K).

      .. xml:tag:: <ac>

         Hydrostatic deformation potential for the conduction band *a*\ :sub:`c` (eV).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <av>

         Hydrostatic deformation potential for the valence band *a*\ :sub:`v` (eV).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <B>

         Radiative recombination coefficient *B* (cm\ :sup:`3`/s).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <b>

         Shear deformation potential *b* (eV).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <C>

         Auger recombination coefficient *C* (cm\ :sup:`6`/s).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <Ce>

         Auger recombination coefficient *C* for electrons (cm\ :sup:`6`/s).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <Ch>

         Auger recombination coefficient *C* for holes (cm\ :sup:`6`/s).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <c11>

         Elastic constant *c*\ :sub:`11` (GPa).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <c12>

         Elastic constant *c*\ :sub:`12` (GPa).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <c13>

         Elastic constant *c*\ :sub:`13` (GPa).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <c33>

         Elastic constant *c*\ :sub:`33` (GPa).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <c44>

         Elastic constant *c*\ :sub:`44` (GPa).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <CB>

         Conduction band level *CB* (eV).

         Variables: ``T`` — temperature (K), ``e`` — lateral strain (-),
         ``point`` — point in the Brillouin zone ('*' means minimum bandgap).

      .. xml:tag:: <chi>

         Electron affinity *χ* (eV).

         Variables: ``T`` — temperature (K), ``e`` — lateral strain (-),
         ``point`` — point in the Brillouin zone ('*' means minimum bandgap).

      .. xml:tag:: <cond>

         Electrical conductivity sigma in-plane (lateral) and cross-plane (vertical) direction (S/m).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <condtype>

         Electrical conductivity type. In semiconductors this indicates what type of carriers :xml:tag:`<Nf>` refers to. Value of this property is not interpreted as Python code. Instead, one of the following values is required: ``n``, ``i``, ``p``, ``other`` (or: ``N``, ``I``, ``P``, ``OTHER``).

      .. xml:tag:: <cp>

         Specific heat at constant pressure (J/(kg K)).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <D>

         Ambipolar diffusion coefficient *D* (cm\ :sup:`2`/s).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <dens>

         Density (kg/m\ :sup:`3`).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <Dso>

         Split-off energy *D*\ :sub:`so` (eV).

         Variables: ``T`` — temperature (K), ``e`` — lateral strain (-).

      .. xml:tag:: <e13>

         Piezoelectric constant *e*\ :sub:`13` (C/m\ :sup:`2`).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <e15>

         Piezoelectric constant *e*\ :sub:`15` (C/m\ :sup:`2`).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <e33>

         Piezoelectric constant *e*\ :sub:`33` (C/m\ :sup:`2`).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <EactA>

         Acceptor ionization energy *E*\ :sub:`actA` (eV).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <EactD>

         Donor ionization energy *E*\ :sub:`actD` (eV).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <Eg>

         Energy band gap *E*\ :sub:`g` (eV).

         Variables: ``T`` — temperature (K), ``e`` — lateral strain (-),
         ``point`` — point in the Brillouin zone ('*' means minimum bandgap).

      .. xml:tag:: <eps>

         Dielectric constant *ε*\ :sub:`R` (-).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <lattC>

         Lattice constant (Å).

         Variables: ``T`` — temperature (K), ``x`` — lattice parameter (-).

      .. xml:tag:: <Me>

         Electron effective mass *M*\ :sub:`e` in in-plane (lateral)
         and cross-plane (vertical) direction (*m*\ :sub:`0`).

         Variables: ``T`` — temperature (K), ``e`` — lateral strain (-),
         ``point`` — point in the irreducible Brillouin zone (-).

      .. xml:tag:: <Mh>

         Hole effective mass *M*\ :sub:`h` in in-plane (lateral)
         and cross-plane (vertical) direction (*m*\ :sub:`0`).

         Variables: ``T`` — temperature (K), ``e`` — lateral strain (-).

      .. xml:tag:: <Mhh>

         Heavy hole effective mass *M*\ :sub:`hh` in in-plane (lateral)
         and cross-plane (vertical) direction (*m*\ :sub:`0`).

         Variables: ``T`` — temperature (K), ``e`` — lateral strain (-).

      .. xml:tag:: <Mlh>

         Light hole effective mass *M*\ :sub:`lh` in in-plane (lateral)
         and cross-plane (vertical) direction (*m*\ :sub:`0`).

         Variables: ``T`` — temperature (K), ``e`` — lateral strain (-).

      .. xml:tag:: <mob>

         Majority carriers mobility in-plane (lateral) and cross-plane (vertical) direction
         (cm\ :sup:`2`/(Vs)).

         Variables: T — temperature (K).

      .. xml:tag:: <mobe>

         Electron mobility in-plane (lateral) and cross-plane (vertical) direction
         (cm\ :sup:`2`/(Vs)).

         Variables: T — temperature (K).

      .. xml:tag:: <mobh>

         Hole mobility in-plane (lateral) and cross-plane (vertical) direction
         (cm\ :sup:`2`/(Vs)).

         Variables: T — temperature (K).

      .. xml:tag:: <Mso>

         Split-off mass *M*\ :sub:`so` (*m*\ :sub:`0`).

         Variables: ``T`` — temperature (K), ``e`` — lateral strain (-).

      .. xml:tag:: <Na>

         Acceptor concentration *N*\ :sub:`a` (cm\ :sup:`-3`).

      .. xml:tag:: <Nc>

         Effective density of states in the conduction band *N*\ :sub:`c` (cm\ :sup:`-3`).

         Variables: ``T`` — temperature (K), ``e`` — lateral strain (-),
         ``point`` — point in the Brillouin zone ('*' means minimum bandgap).

      .. xml:tag:: <Nd>

         Donor concentration *N*\ :sub:`d` (cm\ :sup:`-3`).

      .. xml:tag:: <Nf>

         Free carrier concentration *N* (cm\ :sup:`-3`).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <Ni>

         Intrinsic carrier concentration *N*\ :sub:`i` (cm\ :sup:`-3`).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <Nr>

         Complex refractive index *n*\ :sub:`R` (-).

         Variables: ``lam`` — wavelength (nm), ``T`` — temperature (K), n — injected carriers concentration (cm\ :sup:`-3`).

      .. xml:tag:: <nr>

         Real refractive index \ :sup:`2`*n*\ :sub:`R` (-).

         Variables: ``lam`` — wavelength (nm), ``T`` — temperature (K), n — injected carriers concentration (cm\ :sup:`-3`).

      .. xml:tag:: <NR>

         Anisotropic complex refractive index tensor *n*\ :sub:`R` (-).
         Tensor must have the form [ *n*\ :sub:`00`, *n*\ :sub:`11`, *n*\ :sub:`22`, *n*\ :sub:`01`, *n*\ :sub:`10` ].

         Variables: ``lam`` — wavelength (nm), ``T`` — temperature (K), n — injected carriers concentration (cm\ :sup:`-3`).

         .. warning::

            This parameter is used only by solvers that can consider refractive index anisotropy properly. It is strongly advised to also define :xml:tag:`Nr`.

      .. xml:tag:: <Nv>

         Effective density of states in the valance band *N*\ :sub:`v` (cm\ :sup:`-3`).

         Variables: ``T`` — temperature (K), ``e`` — lateral strain (-),
         ``point`` — point in the Brillouin zone ('*' means minimum bandgap).

      .. xml:tag:: <Psp>

         Spontaneous polarization (C/m\ :sup:`2`)

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <taue>

         Monomolecular electrons lifetime *τ* (ns).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <tauh>

         Monomolecular holes lifetime *τ* (ns).

         Variables: ``T`` — temperature (K).

      .. xml:tag:: <thermk>

         Thermal conductivity in in-plane (lateral) and cross-plane (vertical) direction *k* (W/(m K)).

         Variables: ``T`` — temperature (K), ``h`` — layer thickness (µm).

      .. xml:tag:: <VB>

         Valance band level offset *VB* (eV).

         Variables: ``T`` — temperature (K), ``e`` — lateral strain (-),
         ``point`` — point in the Brillouin zone ('*' means minimum bandgap),
         ``hole`` — hole type (``'H'`` or ``'L'``).

      .. xml:tag:: <y1>

         Luttinger parameter *γ*\ :sup:`1` (-).

      .. xml:tag:: <y2>

         Luttinger parameter *γ*\ :sup:`2` (-).

      .. xml:tag:: <y3>

         Luttinger parameter *γ*\ :sup:`3` (-).


.. xml:tag:: <library>

   Load binary materials library.

   :attr required name: Name of the library to load. This should be the name of the library file without the extension (``.so`` or ``.dll``). It is resolved relative to the current working directory, however you can specify the absolute path (still without the extension).


.. xml:tag:: <module>

   Import Python module with materials library.

   :attr required name: Name of the module to load. This should be the name of the module file without the ``.py`` extension. The module is imported by Python using standard rules, i.e. it is searched in the current directory, the directory specified by the ``PYTHONPATH`` environmental variable or in one of the system directories.
