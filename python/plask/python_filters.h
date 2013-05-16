#ifndef PLASK__PYTHON_FILTERS_H
#define PLASK__PYTHON_FILTERS_H

#include <plask/filters/filter.h>

#include "python_globals.h"
#include "python_provider.h"


namespace plask { namespace python {

namespace detail {

    static inline void FilterInParseKey(const py::object& key, shared_ptr<GeometryObject>& geom, PathHints*& path) {
        py::object object;
        path = nullptr;
        if (PyTuple_Check(key.ptr())) {
            if (py::len(key) != 2) throw KeyError(py::extract<std::string>(py::str(key)));
            object = key[0];
            path = py::extract<PathHints*>(key[1]);
        } else {
            object = key;
        }
        geom = py::extract<shared_ptr<GeometryObject>>(object);
    }

    template <typename ReceiverT>
    static inline PyObject* FilterInParseResult(const py::object& self, ReceiverT& receiver) {
        py::tuple args = py::make_tuple(self);
        PyObject* result;
        {
            py::object obj(py::ptr(&receiver));
            result = py::incref(obj.ptr());
        }
        return py::with_custodian_and_ward_postcall<0,1>::postcall(args.ptr(), result);
    }

    template <typename PropertyT, typename GeometryT> struct FilterIn
    {
        Filter<PropertyT,GeometryT>& filter;
        FilterIn(Filter<PropertyT,GeometryT>& filter): filter(filter) {}

        static PyObject* __getitem__(const py::object& pyself, const py::object& key) {
            FilterIn<PropertyT,GeometryT>* self = py::extract<FilterIn<PropertyT,GeometryT>*>(pyself);
            shared_ptr<GeometryObject> geom;
            PathHints* path;
            FilterInParseKey(key, geom, path);
            if (geom->hasInSubtree(*self->filter.getGeometry()->getChild())) { // geom is the outer object
                if (auto geomd = dynamic_pointer_cast<GeometryObjectD<2>>(geom))
                    return FilterInParseResult(pyself, self->filter.setOuter(geomd, path));
                if (auto geomd = dynamic_pointer_cast<GeometryObjectD<3>>(geom))
                    return FilterInParseResult(pyself, self->filter.setOuter(geomd, path));
                else
                    throw TypeError("Expected 2D or 3D geometry object, got %1% instead", std::string(py::extract<std::string>(py::str(key))));
            } else if (self->filter.getGeometry()->getChild()->hasInSubtree(*geom)) {
                if (auto geomd = dynamic_pointer_cast<GeometryObjectD<2>>(geom))
                    return FilterInParseResult(pyself, self->filter.appendInner(geomd, path));
                else
                    throw TypeError("Expected 2D geometry object, got %1% instead", std::string(py::extract<std::string>(py::str(key))));
            } else
                throw ValueError("Filter geometry and selected object are not related to each other");
            return nullptr;
        }

        static void __setitem__(const py::object& oself, const py::object& key, const py::object& value) {
        }
    };

    template <typename PropertyT> struct FilterIn<PropertyT,Geometry3D>
    {
        Filter<PropertyT,Geometry3D>& filter;
        FilterIn(Filter<PropertyT,Geometry3D>& filter): filter(filter) {}

        static PyObject* __getitem__(const py::object& pyself, const py::object& key) {
            FilterIn<PropertyT,Geometry3D>* self = py::extract<FilterIn<PropertyT,Geometry3D>*>(pyself);
            shared_ptr<GeometryObject> geom;
            PathHints* path;
            FilterInParseKey(key, geom, path);
            if (geom->hasInSubtree(*self->filter.getGeometry()->getChild())) { // geom is the outer object
                if (auto geomd = dynamic_pointer_cast<GeometryObjectD<3>>(geom))
                    return FilterInParseResult(pyself, self->filter.setOuter(geomd, path));
                else
                    throw TypeError("Expected 3D geometry object", std::string(py::extract<std::string>(py::str(key))));
            } else if (self->filter.getGeometry()->getChild()->hasInSubtree(*geom)) {
                if (auto geomd = dynamic_pointer_cast<GeometryObjectD<3>>(geom))
                    return FilterInParseResult(pyself, self->filter.appendInner(geomd, path));
                else
                    throw TypeError("Expected 3D geometry object or 2D geometry, got %1% instead", std::string(py::extract<std::string>(py::str(key))));
            } else {
                if (auto geomd = dynamic_pointer_cast<Geometry2DCartesian>(geom)) {
                    if (self->filter.getGeometry()->getChild()->hasInSubtree(*geomd->getExtrusion()))
                        return FilterInParseResult(pyself, self->filter.appendInner(geomd, path));
                    else
                        throw ValueError("Filter geometry and selected object are not related to each other");
//                 } else if (auto geomd = dynamic_pointer_cast<Geometry2DCylindrical>(geom)) {
//                     if (self->filter.getGeometry()->getChild()->hasInSubtree(*geomd->getRevolution()))
//                         return FilterInParseResult(pyself, self->filter.appendInner(geomd, path));
//                     else
//                         throw ValueError("Filter geometry and selected object are not related to each other");
                } else
                    throw TypeError("Expected 3D geometry object or 2D geometry, got %1% instead", std::string(py::extract<std::string>(py::str(key))));
            }
            return nullptr;
        }

        static void __setitem__(const py::object& oself, const py::object& key, const py::object& value) {
        }
    };

    template <typename PropertyT, typename GeometryT>
    FilterIn<PropertyT,GeometryT>* getFilterIn(Filter<PropertyT,GeometryT>& filter) {
        return new FilterIn<PropertyT,GeometryT>(filter);
    }

    template <typename PropertyT, typename GeometryT>
    py::class_<Filter<PropertyT,GeometryT>, shared_ptr<Filter<PropertyT,GeometryT>>, py::bases<Solver>, boost::noncopyable>
    registerFilterImpl(const char* suffix)
    {
        py::class_<Filter<PropertyT,GeometryT>, shared_ptr<Filter<PropertyT,GeometryT>>, py::bases<Solver>, boost::noncopyable>
        filter_class(("FilterFor"+type_name<PropertyT>()+suffix).c_str(),
                     ("Data filter for "+std::string(PropertyT::NAME)+" for use in 3D solvers.").c_str(),
                     py::init<shared_ptr<GeometryT>>()
                    );
        filter_class
            .def_readonly("out", &Filter<PropertyT,GeometryT>::out, "Filter output provider")
            .add_property("ins", py::make_function(&getFilterIn<PropertyT,GeometryT>,
                                                   py::with_custodian_and_ward_postcall<0,1,py::return_value_policy<py::manage_new_object>>()),
                          "Filter input receivers collection")
        ;

        py::scope scope = filter_class;

        py::class_<FilterIn<PropertyT,GeometryT>, boost::noncopyable>("Inputs", py::no_init)
            .def("__getitem__", &FilterIn<PropertyT,GeometryT>::__getitem__)
            .def("__setitem__", &FilterIn<PropertyT,GeometryT>::__setitem__)
        ;

        return filter_class;
    }

}


template <typename PropertyT>
void registerFilters() {

    detail::registerFilterImpl<PropertyT,Geometry3D>("3D");
    detail::registerFilterImpl<PropertyT,Geometry2DCartesian>("2D");

}


}} // namespace plask::python

#endif // PLASK__PYTHON_FILTERS_H
