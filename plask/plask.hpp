#ifndef PLASK__PLASK_HPP
#define PLASK__PLASK_HPP

/** @file
This file allows for easy include all plask features.

It also includes manual main page in doxygen format.

\mainpage PLaSK developer manual

\section plask_about About
\b PLaSK (<b>P</b>hotonic <b>La</b>ser <b>S</b>imulation <b>K</b>it) is a complete software for performing simulations of semiconductor lasers and some other photonic devices.
It consist of several computational modules, which are used to perform numerical analysis of various physical phenomena in investigated structures, e.g. heat transfer, current flow,
gain in quantum wells, optical modes, etc. Due to the interactions between these phenomena, PLaSK offers a feasible way of performing self-consitent calculations considering these
interactions. The open architecture of the PLaSK framework allows to set-up almost any kind of computations using provided modules or newly-written ones.

PLaSK project consists of:
- \b plask-core: The core library which provides framework for structure geometry description, material database, and data exchange between modules. It also contains Python bindings,
  which allows to use Python scripts for setting-up the computation logic and the executable to run the simulation. All the other components depend on it.
- \b plask-modules: PLaSK computational modules.
- \b plask-gui: Graphical user interface and visual experiment designer which eases definition of an analyzed structure and helps in setting-up and performing simulations.

This documentation is intended for developers who want to write new PLaSK modules or enhance the core functionality. However, before you begin it is advisable to read the User Manual
in order to get familiar with the PLaSK architecture and the way the computations are performed. Here you will find the technical details of the software.

\section plask_source_code PLaSK source code

PLaSK is designed using modern programming technologies. The core and computational modules are written in C++ language using its latest 2011 standard. It means that it can be compiled
only with compilers supporting this standard. The encouraged choices are GCC (ver. 4.6 and above) or ICC. However other compilers migh work as well.
The also requires the Boost C++ library and the modules might need some other libraries to be installed as well (eg. BLAS and LAPACK). The user interface is provided through Python
bindings and the communication between C++ and Python parts require the Boost Python library. The graphical experiment designer is written in Python and uses the Qt graphics library
(with PySide bindings) and Matplotlib for presenting the calculation results in attractive form. You should get familiar with all these technologies if you want to make modifications to PLaSK.

\remarks Although it is possible to compile and use PLaSK without Python, it is strongly discouraged, as all the simulation logic would have to be written in C++ and compiled for every
structure analyzed. Such approach is also hardly documented and not tested.


\section plask_tutorials Tutorials

Here you can find the tutorials expalinig how you can extend PLaSK by writing new module, creating providers and receivers for data exchange between modules or implement a new mesh.
You should start reading from section \ref modules "How to implement an own module" as it explains the most basic particulars of the internal PLaSK architecture.

- \subpage modules "How to write a calculation module?"
    - \subpage providers "How to use providers and receivers for data exchange?"
- \subpage providers_writing "How to implement new providers and receivers for new data types?"
- \subpage meshes "How to use meshes and write an own ones?"
    - \subpage interpolation "All about interpolation."
    - \subpage boundaries "All about boundaries and boundary conditions."
- \subpage geometry "All about geometry."
*/

#include <plask/config.h>

#include "memory.h"
#include "math.h"
#include "exceptions.h"
#include "module.h"
#include "vec.h"
#include "axes.h"
#include "space.h"

#include "geometry/geometry.h"
#include "geometry/primitives.h"
#include "geometry/element.h"
#include "geometry/manager.h"
#include "geometry/transform.h"
#include "geometry/leaf.h"
#include "geometry/transform_space_cartesian.h"
#include "geometry/transform_space_cylindric.h"
#include "geometry/container.h"
#include "geometry/reader.h"
#include "geometry/border.h"

#include "material/material.h"
#include "material/db.h"
#include "material/info.h"
#include "material/air.h"

#include "log/log.h"
#include "log/data.h"
#include "log/id.h"


#include "mesh/mesh.h"
#include "mesh/interpolation.h"
#include "mesh/rectilinear.h"

#include "provider/provider.h"
#include "provider/temperature.h"
#include "provider/optical.h"

#endif // PLASK__PLASK_HPP
