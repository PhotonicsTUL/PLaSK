#include "../python_globals.h"
#include "../python_numpy.h"
#include "../python_mesh.h"

#include <boost/python/stl_iterator.hpp>

#include <plask/mesh/mesh.h>
#include <plask/mesh/interpolation.h>
#include <plask/mesh/triangular2d.h>
#include <plask/mesh/generator_triangular.h>

namespace plask { namespace python {

void register_mesh_triangular() {

    py::class_<TriangularMesh2D, shared_ptr<TriangularMesh2D>, py::bases<MeshD<2>>> triangularMesh2D("TriangularMesh2D",
        u8"Two-dimensional triangular mesh\n\n",
        py::no_init);


}

} } // namespace plask::python
