/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#ifndef PLASK__PLASK_HPP
#define PLASK__PLASK_HPP

/** @file
This file allows for easy include of all plask features.

It also contains manual main page in doxygen format.

\mainpage PLaSK Developer Manual

\section plask_about About
\b PLaSK (<b>P</b>hotonic <b>La</b>ser <b>S</b>imulation <b>K</b>it) is a complete software for performing simulations of semiconductor lasers and some other photonic devices.
It consist of several computational solvers, which are used to perform numerical analysis of various physical phenomena in investigated structures, e.g. heat transfer, current flow,
gain in quantum wells, optical modes, etc. Due to the interactions between these phenomena, PLaSK offers a feasible way of performing self-consitent calculations considering these
interactions. The open architecture of the PLaSK framework allows to set-up almost any kind of computations using provided solvers or newly-written ones.

PLaSK project consists of:
- \b plask-core: The core library which provides framework for structure geometry description, material database, and data exchange between solvers. It also contains Python bindings,
  which allows to use Python scripts for setting-up the computation logic and the executable to run the simulation. All the other components depend on it.
- \b plask-solvers: PLaSK computational solvers.
- \b plask-gui: Graphical user interface and visual experiment designer which eases definition of an analyzed structure and helps in setting-up and performing simulations.

This documentation is intended for developers who want to write new PLaSK solvers or enhance the core functionality. However, before you begin it is advisable to read the User Manual
in order to get familiar with the PLaSK architecture and the way the computations are performed. Here you will find the technical details of the software.

\section plask_source_code PLaSK source code

PLaSK is designed using modern programming technologies. The core and computational solvers are written in C++ language using its latest 2011 standard. It means that it can be compiled
only with compilers supporting this standard. The encouraged choices are GCC (ver. 4.6 and above) or ICC. However other compilers migh work as well.
The also requires the Boost C++ library and the solvers might need some other libraries to be installed as well (eg. BLAS and LAPACK). The user interface is provided through Python
bindings and the communication between C++ and Python parts requires the Boost Python library. The graphical experiment designer is written in Python and uses the Qt graphics library
(with PySide bindings) and Matplotlib for presenting the calculation results in attractive form. You should get familiar with all these technologies if you want to make modifications to PLaSK.

\remarks Although it is possible to compile and use PLaSK without Python, it is strongly discouraged, as all the simulation logic would have to be written in C++ and compiled for every
structure analyzed. Such approach is also hardly documented and not tested.


\section plask_tutorials Tutorials

Here you can find the tutorials explaining how you can extend PLaSK by writing new solver, creating providers and receivers for data exchange between solvers or implement a new mesh.
You should start reading from section \ref solvers "How to implement an own solver" as it explains the most basic particulars of the internal PLaSK architecture.

- \subpage style "Coding style guide" — please read it first!
- \subpage solvers "How to write a calculation solver?"
    - \subpage providers "How to use providers and receivers for data exchange?"
- \subpage providers_writing "How to implement new providers and receivers for new data types?"
- \subpage meshes "How to use meshes and write an own ones?"
    - \subpage interpolation "All about interpolation."
    - \subpage boundaries "All about boundaries and boundary conditions."
- \subpage geometry "All about geometry."
*/

#include <plask/config.hpp>

#include "memory.hpp"
#include "memalloc.hpp"
#include "math.hpp"
#include "exceptions.hpp"
#include "solver.hpp"
#include "vec.hpp"
#include "axes.hpp"
#include "manager.hpp"

#include "vector/tensor2.hpp"
#include "vector/tensor3.hpp"

#include "material/material.hpp"
#include "material/db.hpp"
#include "material/info.hpp"

#include "log/log.hpp"
#include "log/data.hpp"
#include "log/id.hpp"

#include "utils/xml.hpp"

#include "parallel.hpp"

//this contains all geometry stuff
#include "geometry/geometry.hpp"

#include "mesh/mesh.hpp"
#include "mesh/utils.hpp"
#include "mesh/interpolation.hpp"
#include "mesh/boundary_conditions.hpp"
#include "mesh/rectangular.hpp"
#include "mesh/generator_rectangular.hpp"
#include "mesh/rectangular_spline.hpp"
#include "mesh/rectangular_masked.hpp"
#include "mesh/rectangular_masked_spline.hpp"
#include "mesh/triangular2d.hpp"
#include "mesh/extruded_triangular3d.hpp"
#include "mesh/basic.hpp"

#include "provider/provider.hpp"
#include "provider/providerfor.hpp"
#include "provider/combined_provider.hpp"

#include "phys/constants.hpp"
#include "phys/functions.hpp"

#include "properties/thermal.hpp"
#include "properties/electrical.hpp"
#include "properties/optical.hpp"
#include "properties/gain.hpp"
#include "properties/energylevels.hpp"

#include "filters/filter.hpp"

#include "utils/openmp.hpp"

#include "utils/warnings.hpp"


#endif // PLASK__PLASK_HPP
