#ifndef PLASK__PYTHON_FILTERS_H
#define PLASK__PYTHON_FILTERS_H

#include <plask/filters/filter.h>

#include "python_globals.h"
#include "python_provider.h"


namespace plask { namespace python {

namespace detail {

    static inline void filterin_parse_key(const py::object& key, shared_ptr<GeometryObject>& geom, PathHints*& path) {
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

    struct FilterinGetitemResult {
        template <typename ReceiverT>
        static inline PyObject* call(const py::object& self, ReceiverT& receiver) {
            py::tuple args = py::make_tuple(self);
            PyObject* result;
            {
                py::object obj(py::ptr(&receiver));
                result = py::incref(obj.ptr());
            }
            result = py::with_custodian_and_ward_postcall<0,1>::postcall(args.ptr(), result);
            if (!result) throw py::error_already_set();
            return result;
        }
    };

    struct FilterinSetitemResult {
        template <typename ReceiverT>
        static inline PyObject* call(const py::object& self, ReceiverT& receiver, const py::object& value) {
            typedef detail::RegisterReceiverImpl<ReceiverT, ReceiverT::PropertyTag::propertyType, typename  ReceiverT::PropertyTag::ExtraParams> RegisterT;
            RegisterT::assign(receiver, value);
            return py::incref(Py_None);
        }
    };

    template <typename PropertyT, typename GeometryT> struct FilterIn
    {
        Filter<PropertyT,GeometryT>& filter;
        FilterIn(Filter<PropertyT,GeometryT>& filter): filter(filter) {}

        template <typename Fun, typename... Args>
        static PyObject* __getsetitem__(const py::object& pyself, const py::object& key, Args... value) {
            FilterIn<PropertyT,GeometryT>* self = py::extract<FilterIn<PropertyT,GeometryT>*>(pyself);
            shared_ptr<GeometryObject> geom;
            PathHints* path;
            filterin_parse_key(key, geom, path);
            if (geom->hasInSubtree(*self->filter.getGeometry()->getChild())) { // geom is the outer object
                if (auto geomd = dynamic_pointer_cast<GeometryObjectD<2>>(geom))
                    return Fun::call(pyself, self->filter.setOuter(geomd, path), value...);
                if (auto geomd = dynamic_pointer_cast<GeometryObjectD<3>>(geom))
                    return Fun::call(pyself, self->filter.setOuter(geomd, path), value...);
                else
                    throw TypeError("Expected 2D or 3D geometry object, got %1% instead", std::string(py::extract<std::string>(py::str(key))));
            } else if (self->filter.getGeometry()->getChild()->hasInSubtree(*geom)) {
                if (auto geomd = dynamic_pointer_cast<GeometryObjectD<2>>(geom))
                    return Fun::call(pyself, self->filter.appendInner(geomd, path), value...);
                else
                    throw TypeError("Expected 2D geometry object, got %1% instead", std::string(py::extract<std::string>(py::str(key))));
            } else
                throw ValueError("Filter geometry and selected object are not related to each other");
            return nullptr;
        }
    };

    template <typename PropertyT> struct FilterIn<PropertyT,Geometry3D>
    {
        Filter<PropertyT,Geometry3D>& filter;
        FilterIn(Filter<PropertyT,Geometry3D>& filter): filter(filter) {}

        template <typename Fun, typename... Args>
        static PyObject* __getsetitem__(const py::object& pyself, const py::object& key, Args... value) {
            FilterIn<PropertyT,Geometry3D>* self = py::extract<FilterIn<PropertyT,Geometry3D>*>(pyself);
            shared_ptr<GeometryObject> geom;
            PathHints* path;
            filterin_parse_key(key, geom, path);
            if (geom->hasInSubtree(*self->filter.getGeometry()->getChild())) { // geom is the outer object
                if (auto geomd = dynamic_pointer_cast<GeometryObjectD<3>>(geom))
                    return Fun::call(pyself, self->filter.setOuter(geomd, path), value...);
                else
                    throw TypeError("Expected 3D geometry object", std::string(py::extract<std::string>(py::str(key))));
            } else if (self->filter.getGeometry()->getChild()->hasInSubtree(*geom)) {
                if (auto geomd = dynamic_pointer_cast<GeometryObjectD<3>>(geom))
                    return Fun::call(pyself, self->filter.appendInner(geomd, path), value...);
                else
                    throw TypeError("Expected 3D geometry object or 2D geometry, got %1% instead", std::string(py::extract<std::string>(py::str(key))));
            } else {
                if (auto geomd = dynamic_pointer_cast<Geometry2DCartesian>(geom)) {
                    if (self->filter.getGeometry()->getChild()->hasInSubtree(*geomd->getExtrusion()))
                        return Fun::call(pyself, self->filter.appendInner(geomd, path), value...);
                    else
                        throw ValueError("Filter geometry and selected object are not related to each other");
//                 } else if (auto geomd = dynamic_pointer_cast<Geometry2DCylindrical>(geom)) {
//                     if (self->filter.getGeometry()->getChild()->hasInSubtree(*geomd->getRevolution()))
//                         return Fun::call(pyself, self->filter.appendInner(geomd, path), value...);
//                     else
//                         throw ValueError("Filter geometry and selected object are not related to each other");
                } else
                    throw TypeError("Expected 3D geometry object or 2D geometry, got %1% instead", std::string(py::extract<std::string>(py::str(key))));
            }
            return nullptr;
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
            .def("__getitem__", &FilterIn<PropertyT,GeometryT>::template __getsetitem__<FilterinGetitemResult>)
            .def("__setitem__", &FilterIn<PropertyT,GeometryT>::template __getsetitem__<FilterinSetitemResult, py::object>)
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
