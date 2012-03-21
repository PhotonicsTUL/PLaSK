#include "globals.h"
#include <algorithm>
#include <boost/python/stl_iterator.hpp>

#include <plask/mesh/mesh.h>
#include <plask/mesh/interpolation.h>
#include <plask/mesh/rectilinear1d.h>

namespace plask { namespace python {

void register_mesh_rectilinear();

//////////

void register_mesh()
{
    py::object mesh_module { py::handle<>(py::borrowed(PyImport_AddModule("plask.mesh"))) };
    py::scope().attr("mesh") = mesh_module;
    py::scope scope = mesh_module;

    py::enum_<InterpolationMethod> pyInterpolationMethod("interpolation", "Available interpolation methods.");
    for (int i = (int)DEFAULT_INTERPOLATION; i != (int)__ILLEGAL_INTERPOLATION_METHOD__; ++i) {
        pyInterpolationMethod.value(interpolationMethodNames[i], (InterpolationMethod)i);
    }

    register_mesh_rectilinear();

}

}} // namespace plask::python