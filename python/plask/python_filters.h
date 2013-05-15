#ifndef PLASK__PYTHON_FILTERS_H
#define PLASK__PYTHON_FILTERS_H

#include <plask/filters/filter.h>

#include "python_globals.h"
#include "python_provider.h"


namespace plask { namespace python {

namespace detail {

    template <typename PropertyT, typename GeometryT>
    struct PythonFilterInProxy {

//         typedef detail::RegisterReceiverImpl<ReceiverFor<PropertyT,GeometryT>, PropertyT::propertyType, typename PropertyT::ExtraParams> ReceiverRegisterT;

        Filter<PropertyT,GeometryT>& filter;

        PythonFilterInProxy(Filter<PropertyT,GeometryT>& filter): filter(filter) {}


        static PyObject* __getitem__(const py::object& pyself, const py::object& key) {
            PythonFilterInProxy<PropertyT,GeometryT>* self = py::extract<PythonFilterInProxy<PropertyT,GeometryT>*>(pyself);
            shared_ptr<GeometryObject> geom;
            PathHints* path;
            parseKey(key, geom, path);
            if (geom->hasInSubtree(self->filter.getGeometry()->getChild())) { // geom is the outer object
                auto geomd = dynamic_pointer_cast<GeometryObjectD<GeometryT::DIM>>(geom);
                if (geomd) return parseResult(self->filter.setOuter(geomd, path));
                else {
                    auto geom3d = dynamic_pointer_cast<GeometryObjectD<3>>(geom);
                    if (geom3d) return parseResult(self->filter.setOuter(geom3d, path));
                    else throw KeyError(py::extract<std::string>(py::str(key)));
                }
            } else {
                auto geomd = dynamic_pointer_cast<GeometryObjectD<GeometryT::DIM>>(geom);
                if (geomd) return parseResult(self->filter.appendInner(geomd, path));
                else {
                    auto geom2d = dynamic_pointer_cast<GeometryObjectD<2>>(geom);
                    if (geom2d) return parseResult(self->filter.appendInner(geom2d, path));
                    else throw KeyError(py::extract<std::string>(py::str(key)));
                }
            }
        }

        static PyObject* __setitem__(const py::object& oself, const py::object& key, const py::object& value) {
        }

      protected:

        static void parseKey(const py::object& key, shared_ptr<GeometryObject>& geom, PathHints*& path) {
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
        static PyObject* parseResult(ReceiverT& receiver, const py::object& self) {
            py::tuple args = py::make_tuple(self);
            PyObject* result;
            {
                py::object obj(py::ptr(&receiver));
                result = py::incref(obj.ptr());
            }
            return py::with_custodian_and_ward_postcall<0,1>::postcall(result, args.ptr());
        }

    };

    template <typename PropertyT, typename GeometryT>
    PythonFilterInProxy<PropertyT,GeometryT> getFilterIn(Filter<PropertyT,GeometryT>& filter) {
        return PythonFilterInProxy<PropertyT,GeometryT>(filter);
    }

    template <typename PropertyT, typename GeometryT>
    py::class_<Filter<PropertyT,GeometryT>, shared_ptr<Filter<PropertyT,GeometryT>>, py::bases<Solver>, boost::noncopyable>
    registerFilterImpl()
    {
        py::class_<Filter<PropertyT,Geometry3D>, shared_ptr<Filter<PropertyT,Geometry3D>>, py::bases<Solver>, boost::noncopyable>(
                ("FilterFor"+type_name<PropertyT>()+"3D").c_str(),
                ("Data filter for "+std::string(PropertyT::NAME)+" for use in 3D solvers.").c_str(),
                py::init<shared_ptr<Geometry3D>>()
            )
            .def_readonly("out", &Filter<PropertyT,GeometryT>::out, "Filter output provider")
            .add_property("in", &getFilterIn<PropertyT,GeometryT>, "Filter input receivers collection",
                          py::return_value_policy<py::with_custodian_and_ward_postcall<0,1>>())
        ;

        py::class_<PythonFilterInProxy<PropertyT,GeometryT>, boost::noncopyable>("FilterInProxy", py::no_init)
            .def("__getitem__", &PythonFilterInProxy<PropertyT,GeometryT>::__getitem__)
            .def("__setitem__", &PythonFilterInProxy<PropertyT,GeometryT>::__setitem__)
        ;

    }

}


template <typename PropertyT>
void registerFilters() {

    // Filter 3D

}


}} // namespace plask::python

#endif // PLASK__PYTHON_FILTERS_H
