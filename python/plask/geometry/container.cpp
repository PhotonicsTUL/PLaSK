#include <boost/python/suite/indexing/map_indexing_suite.hpp>

#include <plask/geometry/container.h>

#include "geometry.h"

namespace plask { namespace python {

//TODO: make PathHints better manage memory or replace it with normal Python dict

/// Wrapper for PathHints::getChild.
/// Throws exception if there is no such element
GeometryElement* PathHints_getChild(const PathHints& self, GeometryElement* container) {
    GeometryElement* value = self.getChild(container);
    if (value == nullptr) {
        PyErr_SetString(PyExc_KeyError, "No such container in hints");
        throw py::error_already_set();
    }
    return value;
}

// Some other wrappers:

size_t PathHints__len__(const PathHints& self) { return self.hintFor.size(); }

void PathHints__delitem__(PathHints& self, const GeometryElement* key) { self.hintFor.erase(const_cast< GeometryElement*>(key)); }

bool PathHints__contains__(const PathHints& self, const GeometryElement* key) { return self.hintFor.find(const_cast< GeometryElement*>(key)) != self.hintFor.end(); }




void register_geometry_container()
{
    py::class_<PathHints>("PathHints", "Hints are used to to find unique path for all GeometryElement pairs, "
                                            "even if one of the pair element is inserted to geometry graph in more than one place.")
        .def("__len__", &PathHints__len__)

        .def("__getitem__", &PathHints_getChild, py::return_internal_reference<1>())

        .def("__setitem__", (void (PathHints::*)(GeometryElement*,GeometryElement*)) &PathHints::addHint,
                            py::with_custodian_and_ward<1,2, py::with_custodian_and_ward<1,3>>())

        .def("__delitem__", &PathHints__delitem__)

        .def("__contains__", &PathHints__contains__)

        .def("__iter__", py::iterator<PathHints::HintMap, py::return_internal_reference<>>())

        .def("addHint", (void (PathHints::*)(const PathHints::Hint&)) &PathHints::addHint,
             "Add hint to hints map. Overwrite if hint for given container already exists.", py::with_custodian_and_ward<1,2>())

        .def("addHint", (void (PathHints::*)(GeometryElement*,GeometryElement*)) &PathHints::addHint,
             "Add hint to hints map. Overwrite if hint for given container already exists.", py::with_custodian_and_ward<1,2, py::with_custodian_and_ward<1,3>>())

        .def("getChild", &PathHints_getChild, "Get child for given container.", py::return_internal_reference<1>())
    ;
}



}} // namespace plask::python
