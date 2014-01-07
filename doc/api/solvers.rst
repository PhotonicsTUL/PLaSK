.. _sec-api-solvers:

*******
Solvers
*******

Solvers are organized into packages named after solver category. Each package contains binary modules with proper solvers and helper functions that can create solver classes without a need to explicitly import the proper binary module.

The available packages are as follows:

.. autosummary::
   :toctree: solvers
   :template: module.rst

   thermal
   electrical
   gain
   optical
