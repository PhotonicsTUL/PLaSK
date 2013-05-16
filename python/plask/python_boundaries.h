#ifndef PLASK__PYTHON_BOUNDARIES_H
#define PLASK__PYTHON_BOUNDARIES_H

#include "python_globals.h"
#include <plask/utils/format.h>
#include <plask/mesh/boundary_conditions.h>

#if PY_VERSION_HEX >= 0x03000000
#   define NEXT "__next__"
#else
#   define NEXT "next"
#endif

namespace plask { namespace python {

namespace detail {

struct Dupa {};

template <typename MeshT, typename ValueT>
struct RegisterBoundaryConditions {

    typedef BoundaryConditions<MeshT, ValueT> BoundaryConditionsT;
    typedef BoundaryCondition<MeshT, ValueT> ConditionT;

    static ConditionT& __getitem__(BoundaryConditionsT& self, int i) {
        //TODO special proxy class is needed to ensure safe memory management (if user gets an item and removes the original)
        if (i < 0) i = self.size() + i;
        if (i < 0 || i >= self.size()) throw IndexError("boundary conditions index out of range");
        return self[i];
    }

    static void __setitem__1(BoundaryConditionsT& self, int i, py::tuple object) {
        if (i < 0) i = self.size() + i;
        if (i < 0 || i >= self.size()) throw IndexError("boundary conditions index out of range");
        auto iter = self.getIteratorForIndex(i);
        try {
            if (py::len(object) != 2) throw py::error_already_set();
            typename MeshT::Boundary boundary = py::extract<typename MeshT::Boundary>(object[0]);
            ValueT value = py::extract<ValueT>(object[1]);
            *iter = ConditionT(boundary, value);
        } catch (py::error_already_set) {
            throw TypeError("You can only assign a tuple (boundary, value)");
        }
    }

    static void __setitem__2(BoundaryConditionsT& self, int i, const ConditionT& value) {
        if (i < 0) i = self.size() + i;
        if (i < 0 || i >= self.size()) throw IndexError("boundary conditions index out of range");
        auto iter = self.getIteratorForIndex(i);
        *iter = value;
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

    struct Iter {
        BoundaryConditionsT& obj;
        ptrdiff_t i;
        Iter(BoundaryConditionsT& obj): obj(obj), i(-1) {}
        ConditionT& next() {
            ++i;
            if (i == obj.size()) throw StopIteration("");
            return obj[i];
        }
    };

    static Iter __iter__(BoundaryConditionsT& self) {
        return Iter(self);
    }

    struct ConditionIter {
        const ConditionT& obj;
        unsigned i;
        ConditionIter(const ConditionT& obj): obj(obj), i(0) {}
        py::object next() {
            ++i;
            switch (i) {
                case 1: return py::object(obj.place);
                case 2: return py::object(obj.value);
                default: throw StopIteration("");
            }
        }
    };

    static ConditionIter Condition__iter__(const ConditionT& self) {
        return ConditionIter(self);
    }

    static std::string Condition__repr__(const ConditionT& self) {
        return "(" + std::string(py::extract<std::string>(py::str(py::object(self.place)))) +
               ", " + std::string(py::extract<std::string>(py::str(py::object(self.value)))) + ")";
    }

    RegisterBoundaryConditions()
    {
        if (py::converter::registry::lookup(py::type_id<BoundaryConditionsT>()).m_class_object == nullptr) {

            py::class_<BoundaryConditionsT, boost::noncopyable> bc("BoundaryConditions"); bc
                .def("__getitem__", &__getitem__, py::return_value_policy<py::reference_existing_object>())
                .def("__setitem__", &__setitem__1)
                .def("__setitem__", &__setitem__2)
                .def("__delitem__", &__delitem__)
                .def("__len__", &BoundaryConditionsT::size)
                .def("append", &append, "Append new boundary condition to the list", (py::arg("place"), "value"))
                .def("prepend", &prepend, "Prepend new boundary condition to the list", (py::arg("place"), "value"))
                .def("insert", &insert, "Insert new boundary condition to the list at specified position", (py::arg("index"), "place", "value"))
                .def("clear", &BoundaryConditionsT::clear, "Clear all boundary conditions")
                .def("__iter__", &__iter__)
            ;
            py::delattr(py::scope(), "BoundaryConditions");
            py::scope scope1 = bc;

            py::class_<Iter>("Iterator", py::no_init)
                .def(NEXT, &Iter::next, py::return_value_policy<py::reference_existing_object>())
                .def("__iter__", pass_through)
            ;

            py::class_<ConditionT> cd("BoundaryCondition", py::no_init); cd
                .def_readwrite("place", &ConditionT::place, "Location of the boundary condition")
                .def_readwrite("value", &ConditionT::value, "Value of the boundary condition")
                .def("__iter__", &Condition__iter__/*, py::return_value_policy<py::manage_new_object>()*/)
                .def("__repr__", &Condition__repr__)
            ;

            py::scope scope2 = cd;
            py::class_<ConditionIter>("Iterator", py::no_init)
                .def(NEXT, &ConditionIter::next)
                .def("__iter__", pass_through)
            ;
        }
    }

};


}

}} // namespace plask::python

#endif // PLASK__PYTHON_BOUNDARIES_H

