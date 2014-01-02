.. _sec-api:

################
PLaSK Python API
################

PLaSK Python API consists of two parts. The first one is the :mod:`plask` module that, together with its submodules, contains all the core API that allows to define geometries, create meshes, manage materials, etc. The second one is a set of packages containing computational solvers in the separate binary libraries. Such separation enables loading only the minimal set of necessary binaries, effectively saving computer memory.

This section contains API reference manual of both these parts. Its content is automatically generated from the in-line Python documentation of all the referenced objects.

.. toctree::
   :maxdepth: 2

   api-plask
   api-solvers
