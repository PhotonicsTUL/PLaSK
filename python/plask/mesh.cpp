#include "python_globals.h"
#include <algorithm>
#include <boost/python/stl_iterator.hpp>

#include <plask/mesh/mesh.h>
#include <plask/mesh/interpolation.h>
#include <plask/mesh/rectilinear1d.h>

namespace plask { namespace python {

void register_mesh_rectangular();

template <typename T>
static bool __nonempty__(const T& self) { return !self.empty(); }

/// Generic declaration of base class of xD mesh generator
template <int mesh_dim>
py::class_<MeshGeneratorD<mesh_dim>, shared_ptr<MeshGeneratorD<mesh_dim>>, py::bases<MeshGenerator>, boost::noncopyable>
ExportMeshGenerator(const char* name) {
   // py::scope scope = parent;
    //std::string name = py::extract<std::string>(parent.attr("__name__"));
    std::string nameD = boost::lexical_cast<std::string>(mesh_dim) + "D";
    py::class_<MeshGeneratorD<mesh_dim>, shared_ptr<MeshGeneratorD<mesh_dim>>, py::bases<MeshGenerator>, boost::noncopyable>
    pyclass(name, ("Base class for all " + nameD +  " mesh generators.").c_str(), py::no_init);
    pyclass.def("__call__", &MeshGeneratorD<mesh_dim>::operator(), "Generate mesh for given geometry or load it from the cache", py::arg("geometry"));
    pyclass.def("generate", &MeshGeneratorD<mesh_dim>::generate, "Generate mesh for given geometry omitting the cache", py::arg("geometry"));
    pyclass.def("clear_cache", &MeshGeneratorD<mesh_dim>::clearCache, "Clear cache of generated meshes");
    return pyclass;
}

void register_mesh()
{
    py_enum<InterpolationMethod> pyInterpolationMethod("interpolation", "Available interpolation methods.");
    for (unsigned method = INTERPOLATION_DEFAULT; method != __ILLEGAL_INTERPOLATION_METHOD__; ++method) {
        pyInterpolationMethod.value(interpolationMethodNames[method], (InterpolationMethod)method);
    }

    py::object mesh_module { py::handle<>(py::borrowed(PyImport_AddModule("plask.mesh"))) };
    py::scope().attr("mesh") = mesh_module;
    py::scope scope = mesh_module;

    scope.attr("__doc__") =
        "Meshes classes.\n\n"
    ;

    py::class_<Mesh, shared_ptr<Mesh>, boost::noncopyable>("Mesh", "Base class for all meshes", py::no_init)
        .def("__len__", &Mesh::size)
        .def("__nonzero__", &__nonempty__<Mesh>, "Return True if the mesh is empty")
    ;

    py::class_<MeshD<1>, shared_ptr<MeshD<1>>, py::bases<Mesh>, boost::noncopyable> mesh1d("Mesh1D",
        "Base class for every one-dimensional transverse mesh in two-dimensional geometry", py::no_init); mesh1d
        .def("__iter__", py::range(&MeshD<1>::begin, &MeshD<1>::end))
    ;
    mesh1d.attr("dim") = 1;

    py::class_<MeshD<2>, shared_ptr<MeshD<2>>, py::bases<Mesh>, boost::noncopyable> mesh2d("Mesh2D",
        "Base class for every two-dimensional mesh", py::no_init); mesh2d
        .def("__iter__", py::range(&MeshD<2>::begin, &MeshD<2>::end))
    ;
    mesh2d.attr("dim") = 2;

    py::class_<MeshD<3>, shared_ptr<MeshD<3>>, py::bases<Mesh>, boost::noncopyable> mesh3d("Mesh3D",
        "Base class for every two-dimensional mesh", py::no_init); mesh3d
        .def("__iter__", py::range(&MeshD<3>::begin, &MeshD<3>::end))
    ;
    mesh3d.attr("dim") = 3;

    py::class_<MeshGenerator, shared_ptr<MeshGenerator>, boost::noncopyable>("MeshGenerator",
        "Base class for all mesh generators", py::no_init)
    ;

    ExportMeshGenerator<1>("Generator1D");
    ExportMeshGenerator<2>("Generator2D");
    ExportMeshGenerator<3>("Generator3D");

    register_mesh_rectangular();

    register_vector_of<RectilinearAxis>("Ordered");

}

}} // namespace plask::python
