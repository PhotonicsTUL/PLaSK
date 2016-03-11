Optical Analysis of Photonic Crystal Slab Waveguide
---------------------------------------------------

Analyzed structure
^^^^^^^^^^^^^^^^^^

All the previous tutorials dealed with two-dimensional geometry. Here we will define a three-dimensional structure and perform its optical analysis. This tutorial is base on the article ‘Numerical Methods for modeling Photonic-Crystal VCSELs’ [Dems-2010]_, which compares several advanced method for analysis of a PC-VCSEL.

Photonic crystal etched in the VCSEL structre breaks its axial symmetry, and, hence, the two-dimensional cylindirical approximation cannot be used any more. Furthermore, strong refractive index contrast between semiconductor layers and photonic crystal pattern causes strong light scattering and a typical linear polarization (LP) approximation is no longer valid. In consequence, application of popular simplified models is impossible, especially in situations where one needs to determine not only the resonant wavelength of the PC-VCSEL cavity, but also the cavity Q-factor or the gain characteristics of the laser.

.. _fig-tutorial5-pc-vcsel:
.. figure:: tutorial5-pc-vcsel.*
   :scale: 120%
   :align: center

   Schematic structure of PC-VCSEL modelled in :file:`tutorial5.xpl` file.

The structure for this tutorial is a gallium arsenide PC-VCSEL presented in the figure :ref:`schematic structure of PC-VCSEL <fig-tutorial5-pc-vcsel>`. It consists of single-wavelength long GaAs cavity, sandwiched by 24/29 pairs of top/bottom GaAs/AlGaAs distributed Bragg reflectors (DBRs). The optical gain is provided by three layers of 8 nm-thick InGaAs quantum wells separated by 5 nm GaAs barriers and located together in the anti-node position. The diameter of the optical gain region is set to be the same as the photonic crystal pitch Λ. Details for layer thickness and refractive index values can be found in the table below:

+--------+---------------------+----------------+------------------+
| Layer                        | Thickness [nm] | Refractive index |
+========+=====================+================+==================+
|        | Top DBR             | 69.40          | 3.53             |
|        |                     |                |                  |
|        | 24.5 pairs          | 79.55          | 3.08             |
+--------+---------------------+----------------+------------------+
|        |                     | 121.71         | 3.53             |
+        +---------------------+----------------+------------------+
| Cavity | Multi quantum wells | 3×8.00 / 2×5.00| 3.56 / 3.35      |
+        +---------------------+----------------+------------------+
|        |                     | 121.71         | 3.53             |
+--------+---------------------+----------------+------------------+
|        | Bottom DBR          | 79.55          | 3.08             |
|        |                     |                |                  |
|        | 29.5 pairs          | 69.40          | 3.53             |
+--------+---------------------+----------------+------------------+
|        | Substrate           | infinite       | 3.53             |
+--------+---------------------+----------------+------------------+

In the quantum wells the imaginary part of the refractive index is *n*\ :sub:`g` for *r* ≤ *Λ* and –0.01 for *r* > *Λ*.

The photonic crystal is etched in the structure and consists of three rings of a triangular lattice with one missing hole in the center. In the analysis the holes are etched from the top edge of the laser structure down to the specified number of DBR pairs.

Epitaxial layers
~~~~~~~~~~~~~~~~

Defining epitaxial layers is very similar to the previous tutorials. However, this time we operate in 3D, which means we will have sligtly different geometrical objects and we will need to specify three coordinates or dimensions istead of two.

Open new PLaSK window, and switch to the *Geometry* tab. Next press |list-add| and select ``Cartesian3D`` from the list. It will create three-dimensional geometry (:xml:tag:`cartesian3d`). In the bottom left of the window, define a name ‘\ *main*\ ’ for the geometry, and type ‘\ *GaAs*\ ’ in the ‘Bottom’ field of the Border Settings (bottom-left one). Next, press |list-add| again, select ``Item`` submenu and choose ``Stack``. Select the new stack in the geometry tree and a Cuboid object.


.. [Dems-2010]
   M. Dems, I.-S. Chung, P. Nyakas, S. Bischoff, K. Panajotov,
   ‘Numerical Methods for Modeling Photonic-Crystal VCSELs,’
   Opt. Express 18 (2010), pp. 16042-16054

   
.. |list-add| image:: list-add.png
   :align: middle
   :alt: +

