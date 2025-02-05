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
#ifndef PLASK__PYTHON_MESH_H
#define PLASK__PYTHON_MESH_H

#include <cmath>

// Important contains
#include "globals.hpp"
#include "boundaries.hpp"
#include "method_override.hpp"
#include <plask/mesh/mesh.hpp>
#include <plask/mesh/boundary.hpp>

namespace plask { namespace python {

namespace py = boost::python;

/// Custom Python mesh
template <int dim>
struct MeshWrap: public MeshD<dim>, Overriden<MeshD<dim>> {

    MeshWrap(PyObject* self): Overriden<MeshD<dim>> (self) {}

    typename MeshD<dim>::LocalCoords at(std::size_t index) const override;

    size_t size() const override;

    static shared_ptr<MeshD<dim>> __init__(py::tuple args, py::dict kwargs);
};

/// Generic declaration of boundary class for a specific mesh type
template <typename Boundary>
struct ExportBoundary {

    typedef typename Boundary::MeshType MeshType;

    struct PythonPredicate {

        py::object pyfun;

        PythonPredicate(PyObject* obj) : pyfun(py::object(py::handle<>(py::incref(obj)))) {}

        bool operator()(const MeshType& mesh, std::size_t indx) const {
            py::object pyresult = pyfun(boost::ref(mesh), indx);
            bool result;
            try {
                result = py::extract<bool>(pyresult);
            } catch (py::error_already_set&) {
                throw TypeError(u8"boundary predicate did not return Boolean value");
            }
            return result;
        }

        static void* convertible(PyObject* obj) {
            if (PyCallable_Check(obj)) return obj;
            return nullptr;
        }
        static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
            void* storage = ((boost::python::converter::rvalue_from_python_storage<Boundary>*)data)->storage.bytes;
            PythonPredicate predicate(obj);
            new (storage) Boundary { makePredicateBoundary<Boundary>(predicate) };
            data->convertible = storage;
        }

    };

    static BoundaryNodeSet Boundary__call__(
        const Boundary& self, const MeshType& mesh, shared_ptr<const GeometryD<MeshType::DIM>> geometry
    ) {
        return self(mesh, geometry);
    }

    static BoundaryNodeSet BoundarySets__sum__(BoundaryNodeSet self, BoundaryNodeSet right) {
        return new UnionBoundarySetImpl(std::move(self), std::move(right));
    }

    static BoundaryNodeSet BoundarySets__prod__(BoundaryNodeSet self, BoundaryNodeSet right) {
        return new IntersectionBoundarySetImpl(std::move(self), std::move(right));
    }

    static BoundaryNodeSet BoundarySets__diff__(BoundaryNodeSet self, BoundaryNodeSet right) {
        return new DiffBoundarySetImpl(std::move(self), std::move(right));
    }

    static Boundary Boundary__sum__(Boundary self, Boundary right) {
        return Boundary(UnionBoundary<MeshType>(std::move(self), std::move(right)));
    }

    static Boundary Boundary__prod__(Boundary self, Boundary right) {
        return Boundary(IntersectionBoundary<MeshType>(std::move(self), std::move(right)));
    }

    static Boundary Boundary__diff__(Boundary self, Boundary right) {
        return Boundary(DiffBoundary<MeshType>(std::move(self), std::move(right)));
    }

    ExportBoundary(py::object mesh_class) {

        py::scope scope = mesh_class;

        std::string name = py::extract<std::string>(mesh_class.attr("__name__"));

        if (py::converter::registry::lookup(py::type_id<BoundaryNodeSet>()).m_class_object == nullptr) {
            py::class_<BoundaryNodeSet, shared_ptr<BoundaryNodeSet>>("BoundaryInstance",
                ("Boundary specification for particular "+name+" mesh object").c_str(), py::no_init)
                .def("__contains__", &BoundaryNodeSet::contains)
                .def("__iter__", py::range(&BoundaryNodeSet::begin, &BoundaryNodeSet::end))
                .def("__len__", &BoundaryNodeSet::size)
                .def("__or__",  &BoundarySets__sum__,  py::arg("other"), u8"union of sets of indices included in self and other")
                .def("__add__", &BoundarySets__sum__,  py::arg("other"), u8"union of sets of indices included in self and other")
                .def("__and__", &BoundarySets__prod__, py::arg("other"), u8"intersection of sets of indices included in self and other")
                .def("__mul__", &BoundarySets__prod__, py::arg("other"), u8"intersection of sets of indices included in self and other")
                .def("__sub__", &BoundarySets__diff__, py::arg("other"), u8"difference of sets of indices included in self and other")
            ;
            py::delattr(scope, "BoundaryInstance");
        }

        py::class_<Boundary, shared_ptr<Boundary>>("Boundary",
            ("Generic boundary specification for "+name+" mesh").c_str(), py::no_init)
            .def("__call__", &Boundary__call__, (py::arg("mesh"), "geometry"), u8"Get boundary instance for particular mesh",
                 py::with_custodian_and_ward_postcall<0,1, py::with_custodian_and_ward_postcall<0,2>>())
            .def("__or__", &Boundary__sum__, py::with_custodian_and_ward_postcall<0,1, py::with_custodian_and_ward_postcall<0,2>>(),
                 py::arg("other"), u8"boundary which represents union of boundaries (union of produced sets of indices by): self and other")
            .def("__add__", &Boundary__sum__, py::with_custodian_and_ward_postcall<0,1, py::with_custodian_and_ward_postcall<0,2>>(),
                 py::arg("other"), u8"boundary which represents union of boundaries (union of produced sets of indices by): self and other")
            .def("__and__", &Boundary__prod__, py::with_custodian_and_ward_postcall<0,1, py::with_custodian_and_ward_postcall<0,2>>(),
                 py::arg("other"), u8"boundary which represents intersection of boundaries (intersection of produced sets of indices by): self and other")
            .def("__mul__", &Boundary__prod__, py::with_custodian_and_ward_postcall<0,1, py::with_custodian_and_ward_postcall<0,2>>(),
                 py::arg("other"), u8"boundary which represents intersection of boundaries (intersection of produced sets of indices by): self and other")
            .def("__sub__", &BoundarySets__diff__, py::with_custodian_and_ward_postcall<0,1, py::with_custodian_and_ward_postcall<0,2>>(),
                 py::arg("other"), u8"boundary which represents difference of boundaries (difference of produced sets of indices by): self and other")
        ;

        detail::RegisterBoundaryConditions<Boundary, py::object>(false);

        boost::python::converter::registry::push_back(&PythonPredicate::convertible, &PythonPredicate::construct, boost::python::type_id<Boundary>());
    }
};

}} // namespace plask::python

#endif // PLASK__PYTHON_MESH_H
