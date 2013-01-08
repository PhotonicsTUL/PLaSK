#ifndef PLASK__PYTHON_BOUNDARIES_H
#define PLASK__PYTHON_BOUNDARIES_H

#include "python_globals.h"
#include <plask/mesh/boundary_conditions.h>

namespace plask { namespace python {

namespace detail {

template <typename MeshT, typename ValueT>
struct RegisterBoundaryConditions {

    typedef BoundaryConditions<MeshT, ValueT> BoundaryConditionsT;
    typedef BoundaryCondition<MeshT, ValueT> ConditionT;

    struct BoundaryCondition_to_python
    {
        static PyObject* convert(ConditionT const& item) {
            return py::incref(py::make_tuple(item.place, item.value).ptr());
        }
    };

    static ConditionT __getitem__(const BoundaryConditionsT& self, int i) {
        if (i < 0) i = self.size() + i;
        return self[i];
    }

    static void __setitem__1(BoundaryConditionsT& self, int i, py::tuple object) {
        if (i < 0) i = self.size() + i;
        auto iter = self.getIteratorForIndex(i);
        if (iter == self.end()) OutOfBoundsException("BoundaryConditions[]", "index");
        try {
            if (py::len(object) != 2) throw py::error_already_set();
            typename MeshT::Boundary boundary = py::extract<typename MeshT::Boundary>(object[0]);
            ValueT value = py::extract<ValueT>(object[1]);
            *iter = ConditionT(boundary, value);
        } catch (py::error_already_set) {
            throw TypeError("You can only assign a tuple (boundary, value)");
        }
    }

    static void __setitem__2(BoundaryConditionsT& self, const typename MeshT::Boundary& boundary, ValueT value) {
        self.add(ConditionT(boundary, value));
    }

    static void __delitem__(BoundaryConditionsT& self, int i) {
        if (i < 0) i = self.size() + i;
        self.erase(size_t(i));
    }

    static void append(BoundaryConditionsT& self, const typename MeshT::Boundary& boundary, ValueT value) {
        self.add(ConditionT(boundary, value));
    }

    static void prepend(BoundaryConditionsT& self, const typename MeshT::Boundary& boundary, ValueT value) {
        self.insert(0, ConditionT(boundary, value));
    }

    static void insert(BoundaryConditionsT& self, int i, const typename MeshT::Boundary& boundary, ValueT value) {
        if (i < 0) i = self.size() + i;
        if (i < 0 || i >= (int)self.size()) OutOfBoundsException("BoundaryConditions[]", "index");
        self.insert(i, ConditionT(boundary, value));
    }

    RegisterBoundaryConditions()
    {
        if (py::converter::registry::lookup(py::type_id<BoundaryConditionsT>()).m_class_object == nullptr) {

            py::to_python_converter<ConditionT, BoundaryCondition_to_python>();

            py::class_<BoundaryConditionsT>("BoundaryConditions")
                .def("__getitem__", &__getitem__)
                .def("__setitem__", &__setitem__1)
                .def("__setitem__", &__setitem__2)
                .def("__delitem__", &__delitem__)
                .def("__len__", &BoundaryConditionsT::size)
                .def("append", &append, "Append new boundary condition to the list", (py::arg("place"), "value"))
                .def("prepend", &prepend, "Prepend new boundary condition to the list", (py::arg("place"), "value"))
                .def("insert", &insert, "Insert new boundary condition to the list at specified position", (py::arg("index"), "place", "value"))
                .def("clear", &BoundaryConditionsT::clear, "Clear all boundary conditions")
                .def("__iter__", py::range<py::return_value_policy<py::return_by_value>>(
                    (typename BoundaryConditionsT::const_iterator(BoundaryConditionsT::*)()const)&BoundaryConditionsT::begin,
                    (typename BoundaryConditionsT::const_iterator(BoundaryConditionsT::*)()const)&BoundaryConditionsT::end))
            ;
            py::delattr(py::scope(), "BoundaryConditions");
        }
    }

};


}

}} // namespace plask::python

#endif // PLASK__PYTHON_BOUNDARIES_H

