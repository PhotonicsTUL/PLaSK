.. _sec-API:

****************
PLaSK Python API
****************


.. _sec-API-geometry:

Geometry
========

Related XML section: :ref:`<geometry> <sec-XPL-geometry>`.

.. automodule:: plask.geometry
   :members:
   :show-inheritance:
   :undoc-members:


.. _sec-API-mesh:

Mesh
====

Related XML section: :ref:`<grids> <sec-XPL-grids>`.

.. automodule:: plask.mesh
   :members:
   :show-inheritance:
   :undoc-members:


.. _sec-API-material:

material
========

Related XML section: :ref:`<materials> <sec-XPL-materials>`.

.. automodule:: plask.material
   :members:
   :show-inheritance:
   :undoc-members:



.. _sec-API-pylab:

pylab
=====
.. automodule:: plask.pylab
   :members: plot_field, plot_vectors, plot_stream, plot_geometry, plot_mesh, plot_boundary, plot_material_param
   :show-inheritance:
   :undoc-members:


.. _sec-API-hdf5:

hdf5
====
.. automodule:: plask.hdf5
   :members:
   :show-inheritance:
   :undoc-members:


.. _sec-API-solvers:

Solvers
=======

Related XML section: :ref:`<solvers> <sec-XPL-solvers>`.

.. _sec-API-solvers-thermal:

Thermal
-------

.. _sec-API-solvers-thermal-fem:

fem
^^^

Related XML tags: :xml:tag:`<thermal solver="Static2D"> [Static2D]`, :xml:tag:`<thermal solver="StaticCyl"> [StaticCyl]`.

.. automodule:: thermal.fem
   :members:
   :show-inheritance:
   :undoc-members:


.. _sec-API-solvers-thermal-fem3d:

fem3d
^^^^^

Related XML tag: :xml:tag:`<thermal solver="Static3D"> [Static3D]`.

.. automodule:: thermal.fem3d
   :members:
   :show-inheritance:
   :undoc-members:

.. _sec-API-solvers-electrical:

Electrical
----------


.. _sec-API-solvers-electrical-fem:

fem
^^^

Related XML tags: :xml:tag:`<electrical solver="Shockley2D"> [Shockley2D]`, :xml:tag:`<thermal solver="StaticCyl"> [StaticCyl]`.

.. automodule:: electrical.fem
   :members:
   :show-inheritance:
   :undoc-members:


.. _sec-API-solvers-electrical-fem3d:

fem3d
^^^^^

Related XML tag: :xml:tag:`<electrical solver="Shockley3D"> [Shockley3D]`.

.. automodule:: electrical.fem3d
   :members:
   :show-inheritance:
   :undoc-members:



.. _sec-API-solvers-electrical-diffusion:

diffusion
^^^^^^^^^

Related XML tags: :xml:tag:`<electrical solver="Diffusion2D"> [Diffusion2D]`, :xml:tag:`<electrical solver="DiffusionCyl"> [DiffusionCyl]`.

.. automodule:: electrical.diffusion
   :members:
   :show-inheritance:
   :undoc-members:



.. _sec-API-solvers-gain:

Gain
----



.. _sec-API-solvers-gain-simple:

simple
^^^^^^

Related XML tags: :xml:tag:`<gain solver="Fermi2D"> [Fermi2D]`, :xml:tag:`<gain solver="FermiCyl"> [FermiCyl]`.

.. automodule:: gain.simple
   :members:
   :show-inheritance:
   :undoc-members:



.. _sec-API-solvers-optical:

Optical
-------

.. _sec-API-solvers-optical-effective:

effective
^^^^^^^^^

Related XML tags: :xml:tag:`<optical solver=”EffectiveIndex2D”> [EffectiveIndex2D]`, :xml:tag:`<optical solver=”EffectiveFrequencyCyl”> [EffectiveFrequencyCyl]`.

.. automodule:: optical.effective
   :members:
   :show-inheritance:
   :undoc-members:



.. _sec-API-solvers-optical-slab:

slab
^^^^

.. automodule:: optical.slab
   :members:
   :show-inheritance:
   :undoc-members:


