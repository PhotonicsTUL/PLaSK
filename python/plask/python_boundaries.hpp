#ifndef PLASK__PYTHON_BOUNDARIES_H
#define PLASK__PYTHON_BOUNDARIES_H

#include "python_globals.hpp"
#include "plask/utils/format.hpp"
#include "plask/mesh/boundary_conditions.hpp"
#include "plask/manager.hpp"

namespace plask {

template <> inline boost::python::object parseBoundaryValue<boost::python::object>(const XMLReader& tag_with_value)
{
    std::map<std::string, std::string> attributes = tag_with_value.getAttributes();
    attributes.erase("place");
    attributes.erase("placename");
    attributes.erase("placeref");
    if (attributes.size() == 1) {
        auto found = attributes.find("value");
        if (found != attributes.end())
            return python::eval_common_type(found->second);
    }
    boost::python::dict result;
    for (const auto& item: attributes) {
        result[item.first] = python::eval_common_type(item.second);
    }
    return std::move(result);
}


namespace python {

namespace detail {

template <typename Boundary, typename ValueT>
struct RegisterBoundaryConditions {

    typedef BoundaryConditions<Boundary, ValueT> BoundaryConditionsT;
    typedef BoundaryCondition<Boundary, ValueT> ConditionT;

    static ConditionT& __getitem__(BoundaryConditionsT& self, int i) {
        //TODO special proxy class is needed to ensure safe memory management (if user gets an item and removes the original)
        if (i < 0) i += int(self.size());
        if (i < 0 || std::size_t(i) >= self.size()) throw IndexError(u8"boundary conditions index out of range");
        return self[i];
    }

    static void __setitem__1(BoundaryConditionsT& self, int i, py::tuple object) {
        if (i < 0) i += int(self.size());
        if (i < 0 || std::size_t(i) >= self.size()) throw IndexError(u8"boundary conditions index out of range");
        auto iter = self.getIteratorForIndex(i);
        try {
            if (py::len(object) != 2) throw py::error_already_set();
            Boundary boundary = py::extract<Boundary>(object[0]);
            ValueT value = py::extract<ValueT>(object[1]);
            *iter = ConditionT(boundary, value);
        } catch (py::error_already_set&) {
            throw TypeError(u8"You can only assign a tuple (boundary, value)");
        }
    }

    static void __setitem__2(BoundaryConditionsT& self, int i, const ConditionT& value) {
        if (i < 0) i += int(self.size());
        if (i < 0 || std::size_t(i) >= self.size()) throw IndexError(u8"boundary conditions index out of range");
        auto iter = self.getIteratorForIndex(i);
        *iter = value;
    }

    static void __delitem__(BoundaryConditionsT& self, int i) {
        if (i < 0) i += int(self.size());
        self.erase(size_t(i));
    }

    static void append(BoundaryConditionsT& self, const Boundary& boundary, ValueT value) {
        self.add(ConditionT(boundary, value));
    }

    static void prepend(BoundaryConditionsT& self, const Boundary& boundary, ValueT value) {
        self.insert(0, ConditionT(boundary, value));
    }

    static void insert(BoundaryConditionsT& self, int i, const Boundary& boundary, ValueT value) {
        if (i < 0) i += int(self.size());
        if (i < 0 || i >= (int)self.size()) OutOfBoundsException("BoundaryConditions[]", "index");
        self.insert(i, ConditionT(boundary, value));
    }

    static void read_from_xml(BoundaryConditionsT& self, XMLReader& reader, Manager& manager) {
        manager.readBoundaryConditions<Boundary, ValueT>(reader, self);
    }

    struct Iter {
        BoundaryConditionsT& obj;
        std::ptrdiff_t i;
        Iter(BoundaryConditionsT& obj): obj(obj), i(-1) {}
        ConditionT& next() {
            ++i;
            if (i == std::ptrdiff_t(obj.size())) throw StopIteration("");
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

    RegisterBoundaryConditions(bool delattr=true)
    {
        if (py::converter::registry::lookup(py::type_id<BoundaryConditionsT>()).m_class_object == nullptr) {

            py::class_<BoundaryConditionsT, boost::noncopyable> bc("BoundaryConditions", u8"Set of boundary conditions."); bc
                .def("__getitem__", &__getitem__, py::return_value_policy<py::reference_existing_object>())
                .def("__setitem__", &__setitem__1)
                .def("__setitem__", &__setitem__2)
                .def("__delitem__", &__delitem__)
                .def("__len__", &BoundaryConditionsT::size)
                .def("append", &append, u8"Append new boundary condition to the list.", (py::arg("place"), "value"))
                .def("prepend", &prepend, u8"Prepend new boundary condition to the list.", (py::arg("place"), "value"))
                .def("insert", &insert, u8"Insert new boundary condition to the list at specified position.", (py::arg("index"), "place", "value"))
                .def("clear", &BoundaryConditionsT::clear, u8"Clear all boundary conditions.")
                .def("__iter__", &__iter__)
                .def("read_from_xpl", &read_from_xml, (py::arg("xml"), "manager"),
                     u8"Read boundary conditions from active XPL reader.\n\n"
                     u8"This should only be used in the overloaded :meth:`plask.Solver.load_xpl` method.\n")
            ;
            if (delattr) py::delattr(py::scope(), "BoundaryConditions");
            py::scope scope1 = bc;
            (void) scope1;   // don't warn about unused variable scope1

            py::class_<Iter>("_Iterator", py::no_init)
                .def("__next__", &Iter::next, py::return_value_policy<py::reference_existing_object>())
                .def("__iter__", pass_through)
            ;

            py::class_<ConditionT> cd("BoundaryCondition", py::no_init); cd
                .def_readwrite("place", &ConditionT::place, u8"Location of the boundary condition.")
                .def_readwrite("value", &ConditionT::value, u8"Value of the boundary condition.")
                .def("__iter__", &Condition__iter__/*, py::return_value_policy<py::manage_new_object>()*/)
                .def("__repr__", &Condition__repr__)
            ;

            py::scope scope2 = cd;
            (void) scope2;   // don't warn about unused variable scope2
            py::class_<ConditionIter>("_Iterator", py::no_init)
                .def("__next__", &ConditionIter::next)
                .def("__iter__", pass_through)
            ;
        }
    }

};


}

}} // namespace plask::python

#endif // PLASK__PYTHON_BOUNDARIES_H
