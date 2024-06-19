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
#include "python_globals.hpp"
#include <algorithm>
#include <boost/python/stl_iterator.hpp>

#include "plask/mesh/mesh.hpp"
#include "plask/mesh/interpolation.hpp"
#include "plask/mesh/ordered1d.hpp"

#include "python_mesh.hpp"
#include "python_util/raw_constructor.hpp"

namespace plask { namespace python {

void register_mesh_rectangular();
void register_mesh_triangular();
void register_mesh_extruded_triangular();


template <typename T>
static bool __nonempty__(const T& self) { return !self.empty(); }

template <int dim> typename MeshD<dim>::LocalCoords MeshWrap<dim>::at(std::size_t index) const {
    return this->template call_python<typename MeshD<dim>::LocalCoords>("__getitem__", index);
}

template <int dim> size_t MeshWrap<dim>::size() const {
    return this->template call_python<size_t>("__len__");
}

template <int dim> shared_ptr<MeshD<dim>> MeshWrap<dim>::__init__(py::tuple args, py::dict kwargs) {
    if (py::len(args) > 1)
        throw TypeError(u8"__init__() takes exactly 1 non-keyword arguments ({:d} given)", py::len(args));
    if (py::len(kwargs) > 0)
        throw TypeError(u8"__init__() got an unexpected keyword argument '{}'",
                        py::extract<std::string>(kwargs.keys()[0])());

    py::object self(args[0]);
    return plask::make_shared<MeshWrap<dim>>(self.ptr());
}


template <int dim>
struct UnstructuredMesh: public MeshD<dim> {

    py::object points;

    UnstructuredMesh(const py::object& points): points(points) {}

    typename MeshD<dim>::LocalCoords at(std::size_t index) const override {
        OmpLockGuard lock(python_omp_lock);
        return py::extract<typename MeshD<dim>::LocalCoords>(points[index]);
    }

    size_t size() const override {
        OmpLockGuard lock(python_omp_lock);
        return py::len(points);
    }

    static void register_class() {
        py::class_<UnstructuredMesh<dim>, shared_ptr<UnstructuredMesh<dim>>,
                   py::bases<MeshD<dim>>>(NAME, DOCSTRING, py::init<const py::object&>());
    }

    static const char* const NAME;

    static const char* const DOCSTRING;
};

template<> const char* const UnstructuredMesh<2>::NAME = "Unstructured2D";

template<> const char* const UnstructuredMesh<3>::NAME = "Unstructured3D";

template<> const char* const UnstructuredMesh<2>::DOCSTRING =
    u8"Unstructured2D(points)\n\n"
    u8"Two-dimensional unstructured mesh.\n\n"
    u8"Args:\n"
    u8"    points: List of mesh points. Each element in this list should be\n"
    u8"            a two-dimensional vector or sequence of floats with point\n"
    u8"            coordinates.\n";

template<> const char* const UnstructuredMesh<3>::DOCSTRING =
    u8"Unstructured3D(points)\n\n"
    u8"Three-dimensional unstructured mesh.\n\n"
    u8"Args:\n"
    u8"    points: List of mesh points. Each element in this list should be\n"
    u8"            a three-dimensional vector or sequence of floats with point\n"
    u8"            coordinates.\n";


template <int dim>
static shared_ptr<MeshD<dim>> MeshGenerator_generate(MeshGeneratorD<dim>& self, shared_ptr<const GeometryD<(dim==1)?2:dim>> geometry) {
    return self.generate(geometry->getChild());
}

template <int dim>
static shared_ptr<MeshD<dim>> MeshGenerator__call__(MeshGeneratorD<dim>& self, shared_ptr<const GeometryD<(dim==1)?2:dim>> geometry) {
    return self(geometry->getChild());
}

/// Generic declaration of base class of xD mesh generator
template <int mesh_dim>
py::class_<MeshGeneratorD<mesh_dim>, shared_ptr<MeshGeneratorD<mesh_dim>>, py::bases<MeshGenerator>, boost::noncopyable>
ExportMeshGenerator(const char* name) {
   // py::scope scope = parent;
    //std::string name = py::extract<std::string>(parent.attr("__name__"));
    std::string nameD = boost::lexical_cast<std::string>(mesh_dim) + "D";
    py::class_<MeshGeneratorD<mesh_dim>, shared_ptr<MeshGeneratorD<mesh_dim>>, py::bases<MeshGenerator>, boost::noncopyable>
    pyclass(name, ("Base class for all " + nameD +  " mesh generators.").c_str(), py::no_init);
    pyclass.def("__call__", &MeshGenerator__call__<mesh_dim>, py::arg("geometry"));
    pyclass.def("__call__", &MeshGeneratorD<mesh_dim>::operator(),
                u8"Generate mesh for given geometry object or load it from the cache.\n\n"
                u8"Args:\n"
                u8"    geometry: Geometry to generate mesh at.\n"
                u8"    object: Geometry object to generate mesh at.\n",
                py::arg("object"));
    pyclass.def("generate", &MeshGenerator_generate<mesh_dim>, py::arg("geometry"));
    pyclass.def("generate", &MeshGeneratorD<mesh_dim>::generate,
                u8"Generate mesh for given geometry object omitting the cache.\n\n"
                u8"Args:\n"
                u8"    geometry: Geometry to generate mesh at.\n"
                u8"    object: Geometry object to generate mesh at.\n",
                py::arg("object"));
    pyclass.def("clear_cache", &MeshGeneratorD<mesh_dim>::clearCache, u8"Clear cache of generated meshes");
    return pyclass;
}

template <int DIM>
static typename std::conditional<DIM==1, double, Vec<DIM, double>>::type
MeshXD__getitem__(const MeshD<DIM>& self, py::object index) {
    int indx = py::extract<int>(index);
    if (indx < 0) indx += int(self.size());
    if (indx < 0 || size_t(indx) >= self.size()) throw IndexError("mesh index out of range");
    return self.at(indx);
}

void register_mesh()
{
    py_enum<InterpolationMethod> pyInterpolationMethod;
    for (unsigned method = INTERPOLATION_DEFAULT; method != __ILLEGAL_INTERPOLATION_METHOD__; ++method) {
        pyInterpolationMethod.value(interpolationMethodNames[method], (InterpolationMethod)method);
    }

    py::object mesh_module { py::handle<>(py::borrowed(PyImport_AddModule("plask.mesh"))) };
    py::scope().attr("mesh") = mesh_module;
    py::scope scope = mesh_module;

    scope.attr("__doc__") =
        "Meshes classes.\n\n"
    ;

    py::class_<MeshBase, shared_ptr<MeshBase>, boost::noncopyable>("Mesh", u8"Base class for all meshes", py::no_init);

    py::class_<Mesh, shared_ptr<Mesh>, py::bases<MeshBase>, boost::noncopyable>("Mesh",
        u8"Base class for all meshes", py::no_init)
        .def("__len__", &Mesh::size)
        .def("__nonzero__", &__nonempty__<Mesh>, u8"Return True if the mesh is empty")
    ;
    py::implicitly_convertible<shared_ptr<Mesh>, shared_ptr<const Mesh>>();

    py::class_<MeshD<1>, shared_ptr<MeshD<1>>, py::bases<Mesh>, boost::noncopyable> mesh1d("Mesh1D",
        u8"Base class for every one-dimensional transverse mesh in two-dimensional geometry", py::no_init); mesh1d
        .def("__init__", raw_constructor(&MeshWrap<1>::__init__))
        .def("__iter__", py::range(&MeshD<1>::begin, &MeshD<1>::end))
        .def("__getitem__", MeshXD__getitem__<1>);
    ;
    mesh1d.attr("dim") = 1;
    py::implicitly_convertible<shared_ptr<MeshD<1>>, shared_ptr<const MeshD<1>>>();

    py::class_<MeshD<2>, shared_ptr<MeshD<2>>, py::bases<Mesh>, boost::noncopyable> mesh2d("Mesh2D",
        u8"Base class for every two-dimensional mesh", py::no_init); mesh2d
        .def("__init__", raw_constructor(&MeshWrap<2>::__init__))
        .def("__iter__", py::range(&MeshD<2>::begin, &MeshD<2>::end))
        .def("__getitem__", MeshXD__getitem__<2>)
    ;
    mesh2d.attr("dim") = 2;
    py::implicitly_convertible<shared_ptr<MeshD<2>>, shared_ptr<const MeshD<2>>>();

    py::class_<MeshD<3>, shared_ptr<MeshD<3>>, py::bases<Mesh>, boost::noncopyable> mesh3d("Mesh3D",
        u8"Base class for every two-dimensional mesh", py::no_init); mesh3d
        .def("__init__", raw_constructor(&MeshWrap<3>::__init__))
        .def("__iter__", py::range(&MeshD<3>::begin, &MeshD<3>::end))
        .def("__getitem__", MeshXD__getitem__<3>)
    ;
    mesh3d.attr("dim") = 3;
    py::implicitly_convertible<shared_ptr<MeshD<3>>, shared_ptr<const MeshD<3>>>();

    py::class_<MeshGenerator, shared_ptr<MeshGenerator>, py::bases<MeshBase>, boost::noncopyable>("MeshGenerator",
        u8"Base class for all mesh generators", py::no_init)
    ;
    py::implicitly_convertible<shared_ptr<MeshGenerator>, shared_ptr<const MeshGenerator>>();

    ExportMeshGenerator<1>("Generator1D");
    ExportMeshGenerator<2>("Generator2D");
    ExportMeshGenerator<3>("Generator3D");

    UnstructuredMesh<2>::register_class();
    UnstructuredMesh<3>::register_class();

    register_mesh_rectangular();
    register_mesh_triangular();
    register_mesh_extruded_triangular();

    register_vector_of<OrderedAxis>("Ordered");

}

}} // namespace plask::python
