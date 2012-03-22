#include "globals.h"
#include <algorithm>
#include <boost/python/stl_iterator.hpp>

#include <plask/mesh/mesh.h>
#include <plask/mesh/interpolation.h>
#include <plask/mesh/rectilinear1d.h>

namespace plask { namespace python {

void register_mesh_rectilinear();

void register_mesh()
{
    py::object mesh_module { py::handle<>(py::borrowed(PyImport_AddModule("plask.meshes"))) };
    py::scope().attr("meshes") = mesh_module;
    py::scope scope = mesh_module;

    py::enum_<InterpolationMethod> pyInterpolationMethod("interpolation", "Available interpolation methods.");
    for (int i = (int)DEFAULT_INTERPOLATION; i != (int)__ILLEGAL_INTERPOLATION_METHOD__; ++i) {
        pyInterpolationMethod.value(interpolationMethodNames[i], (InterpolationMethod)i);
    }

    py::class_<Mesh<2>, shared_ptr<Mesh<2>>, boost::noncopyable>("Mesh2D", "Base class for every two-dimensional mesh", py::no_init)
        .def("size", &Mesh<2>::size, "Return the size of the mesh")
    ;

    py::class_<Mesh<3>, shared_ptr<Mesh<3>>, boost::noncopyable>("Mesh3D", "Base class for every two-dimensional mesh", py::no_init)
        .def("size", &Mesh<3>::size, "Return the size of the mesh")
    ;

    register_mesh_rectilinear();

}

}} // namespace plask::python