.. _sec-api-plask:

*****************
``plask`` package
*****************

.. automodule:: plask

Modules
=======

.. autosummary::
   :toctree: plask
   :template: module.rst

   geometry
   mesh
   material
   flow
   phys

Classes
=======

.. autosummary::
   :toctree: plask
   :template: class.rst
   :nosignatures:

   config
   vec
   Data
   Manager
   StepProfile
   Solver
   XplWriter

Functions
=========

.. autosummary::

   loadxpl
   runxpl
   print_log

   save_field
   load_field

   plot_profile
   plot_field
   plot_vectors
   plot_stream
   plot_geometry
   plot_mesh
   plot_boundary

Constants
=========

=========== ==================================================================================================================================
``JOBID``   Job index: either a job ID in batch submission system or a float indicating lauch time (that can be converted to ``time.time()``).
``ARRAYID`` Array index in batch submission system. ``None`` if the job is not a part of an array.
``PROCID``  Process index for parallel computations (eg. MPI). ``None`` if this is a regular or serial job.
=========== ==================================================================================================================================


Descriptions
============

.. rubric:: Function Details

.. autofunction::   loadxpl
.. autofunction::   runxpl
.. autofunction::   print_log

.. autofunction::   save_field
.. autofunction::   load_field

.. autofunction::   plot_profile
.. autofunction::   plot_field
.. autofunction::   plot_vectors
.. autofunction::   plot_stream
.. autofunction::   plot_geometry
.. autofunction::   plot_mesh
.. autofunction::   plot_boundary
