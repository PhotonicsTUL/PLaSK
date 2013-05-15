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
        return py::with_custodian_and_ward_postcall<0,1>::postcall(result, args.ptr());
    }

    template <typename PropertyT, typename GeometryT>
    struct FilterInProxy {
        Filter<PropertyT,GeometryT>& filter;
        FilterInProxy(Filter<PropertyT,GeometryT>& filter): filter(filter) {}
    };

    template <typename PropertyT, typename GeometryT> struct FilterIn
    {
        static PyObject* __getitem__(const py::object& pyself, const py::object& key) {
            FilterInProxy<PropertyT,GeometryT>* self = py::extract<FilterInProxy<PropertyT,GeometryT>*>(pyself);
            shared_ptr<GeometryObject> geom;
            PathHints* path;
            FilterInParseKey(key, geom, path);
            if (geom->hasInSubtree(*self->filter.getGeometry()->getChild())) { // geom is the outer object
                if (auto geomd = dynamic_pointer_cast<GeometryObjectD<2>>(geom))
                    return FilterInParseResult(pyself, self->filter.setOuter(geomd, path));
                if (auto geomd = dynamic_pointer_cast<GeometryObjectD<3>>(geom))
                    return FilterInParseResult(pyself, self->filter.setOuter(geomd, path));
                throw KeyError(py::extract<std::string>(py::str(key)));
            } else if (self->filter.getGeometry()->getChild()->hasInSubtree(*geom)) {
                if (auto geomd = dynamic_pointer_cast<GeometryObjectD<2>>(geom))
                    return FilterInParseResult(pyself, self->filter.appendInner(geomd, path));
            } else
                throw ValueError("Filter geometry and selected object are not related to each other");
            return nullptr;
        }

        static PyObject* __setitem__(const py::object& oself, const py::object& key, const py::object& value) {
            return nullptr;
        }
    };

    template <typename PropertyT> struct FilterIn<PropertyT,Geometry3D>
    {
        static PyObject* __getitem__(const py::object& pyself, const py::object& key) {
            FilterInProxy<PropertyT,Geometry3D>* self = py::extract<FilterInProxy<PropertyT,Geometry3D>*>(pyself);
            shared_ptr<GeometryObject> geom;
            PathHints* path;
            FilterInParseKey(key, geom, path);
            if (geom->hasInSubtree(*self->filter.getGeometry()->getChild())) { // geom is the outer object
                if (auto geomd = dynamic_pointer_cast<GeometryObjectD<3>>(geom))
                    return FilterInParseResult(pyself, self->filter.setOuter(geomd, path));
                throw KeyError(py::extract<std::string>(py::str(key)));
            } else if (self->filter.getGeometry()->getChild()->hasInSubtree(*geom)) {
                if (auto geomd = dynamic_pointer_cast<GeometryObjectD<3>>(geom))
                    return FilterInParseResult(pyself, self->filter.appendInner(geomd, path));
                if (auto geomd = dynamic_pointer_cast<Geometry2DCartesian>(geom))
                    return FilterInParseResult(pyself, self->filter.appendInner(geomd, path));
//                 if (auto geomd = dynamic_pointer_cast<Geometry2DCylindrical>(geom))
//                     return FilterInParseResult(pyself, self->filter.appendInner(geomd, path));
                else throw KeyError(py::extract<std::string>(py::str(key)));
            } else
                throw ValueError("Filter geometry and selected object are not related to each other");
            return nullptr;
        }

        static PyObject* __setitem__(const py::object& oself, const py::object& key, const py::object& value) {
            return nullptr;
        }
    };

    template <typename PropertyT, typename GeometryT>
    FilterInProxy<PropertyT,GeometryT> getFilterIn(Filter<PropertyT,GeometryT>& filter) {
        return FilterInProxy<PropertyT,GeometryT>(filter);
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
            .add_property("ins",
                          py::make_function(&getFilterIn<PropertyT,GeometryT>/*,
                                            py::return_value_policy<py::with_custodian_and_ward_postcall<0,1>>()*/),
                          "Filter input receivers collection")
        ;

        py::scope scope = filter_class;

        py::class_<FilterInProxy<PropertyT,GeometryT>, boost::noncopyable>("Inputs", py::no_init)
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
