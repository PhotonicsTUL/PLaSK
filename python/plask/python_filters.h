#ifndef PLASK__PYTHON_FILTERS_H
#define PLASK__PYTHON_FILTERS_H

#include <plask/filters/filter.h>

#include "python_globals.h"
#include "python_provider.h"


namespace plask { namespace python {

extern py::object flow_module;

namespace detail {

    static inline void filterin_parse_key(const py::object& key, shared_ptr<GeometryObject>& geom, PathHints*& path, int& points) {
        py::object object;
        path = nullptr;
        points = 10;
        if (PyTuple_Check(key.ptr())) {
            if (py::len(key) < 2 || py::len(key) > 3) throw KeyError(py::extract<std::string>(py::str(key)));
            object = key[0];
            if (py::len(key) == 3) {
                path = py::extract<PathHints*>(key[1]);
                points = py::extract<int>(key[2]);
            } else {
                try {
                    path = py::extract<PathHints*>(key[1]);
                } catch (py::error_already_set) {
                    PyErr_Clear();
                    try {
                        points = py::extract<int>(key[1]);
                    } catch (py::error_already_set) {
                        throw KeyError(py::extract<std::string>(py::str(key)));
                    }
                }
            }
            if (points < 0) throw KeyError(py::extract<std::string>(py::str(key)));
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

    template <typename PropertyT, typename GeometryT> struct FilterIn;

    template <typename PropertyT> struct FilterIn<PropertyT,Geometry2DCartesian> {
        template <typename Fun, typename... Args>
        static PyObject* __getsetitem__(const py::object& self, const py::object& key, Args... value) {
            Filter<PropertyT,Geometry2DCartesian>* filter = py::extract<Filter<PropertyT,Geometry2DCartesian>*>(self);
            shared_ptr<GeometryObject> geom;
            PathHints* path;
            int points;
            filterin_parse_key(key, geom, path, points);

            if (auto geomd = dynamic_pointer_cast<GeometryObjectD<2>>(geom))
                return Fun::call(self, filter->input(*geomd, path), value...);
            if (auto geomd = dynamic_pointer_cast<Geometry2DCartesian>(geom))
                return Fun::call(self, filter->input(*geomd, path), value...);

            if (auto geomd = dynamic_pointer_cast<GeometryObjectD<3>>(geom))
                return Fun::call(self, filter->setOuter(*geomd, path, points), value...);
            if (auto geomd = dynamic_pointer_cast<Geometry3D>(geom))
                return Fun::call(self, filter->setOuter(*geomd->getChild(), path, points), value...);

            throw TypeError("Wrong geometry type '%1%'", std::string(py::extract<std::string>(py::str(key[0].attr("__class__")))));
            return nullptr;
        }
    };

    template <typename PropertyT> struct FilterIn<PropertyT,Geometry2DCylindrical> {
        template <typename Fun, typename... Args>
        static PyObject* __getsetitem__(const py::object& self, const py::object& key, Args... value) {
            Filter<PropertyT,Geometry2DCylindrical>* filter = py::extract<Filter<PropertyT,Geometry2DCylindrical>*>(self);
            shared_ptr<GeometryObject> geom;
            PathHints* path;
            int points;
            filterin_parse_key(key, geom, path, points);

            if (auto geomd = dynamic_pointer_cast<GeometryObjectD<2>>(geom))
                 return Fun::call(self, filter->input(*geomd, path), value...);
            if (auto geomd = dynamic_pointer_cast<Geometry2DCylindrical>(geom))
                 return Fun::call(self, filter->input(*geomd, path), value...);

            if (auto geomd = dynamic_pointer_cast<GeometryObjectD<3>>(geom))
                return Fun::call(self, filter->setOuter(*geomd, path, points), value...);
            if (auto geomd = dynamic_pointer_cast<Geometry3D>(geom))
                return Fun::call(self, filter->setOuter(*geomd->getChild(), path, points), value...);

            throw TypeError("Wrong geometry type '%1%'", std::string(py::extract<std::string>(py::str(key[0].attr("__class__")))));
            return nullptr;
        }
    };

    template <typename PropertyT> struct FilterIn<PropertyT,Geometry3D>
    {
        template <typename Fun, typename... Args>
        static PyObject* __getsetitem__(const py::object& self, const py::object& key, Args... value) {
            Filter<PropertyT,Geometry3D>* filter = py::extract<Filter<PropertyT,Geometry3D>*>(self);
            shared_ptr<GeometryObject> geom;
            PathHints* path;
            int points;
            filterin_parse_key(key, geom, path, points);

            if (auto geomd = dynamic_pointer_cast<Extrusion>(geom))
                return Fun::call(self, filter->appendInner2D(*geomd, path), value...);
            if (auto geomd = dynamic_pointer_cast<Geometry2DCartesian>(geom))
                return Fun::call(self, filter->appendInner(*geomd, path), value...);

            if (auto geomd = dynamic_pointer_cast<Revolution>(geom))
                return Fun::call(self, filter->appendInner2D(*geomd, path), value...);
            if (auto geomd = dynamic_pointer_cast<Geometry2DCylindrical>(geom))
                return Fun::call(self, filter->appendInner(*geomd, path), value...);

            if (auto geomd = dynamic_pointer_cast<GeometryObjectD<3>>(geom))
                return Fun::call(self, filter->input(*geomd, path), value...);
            if (auto geomd = dynamic_pointer_cast<Geometry3D>(geom))
                return Fun::call(self, filter->input(*geomd->getChild(), path), value...);

            throw TypeError("Wrong geometry type '%1%'", std::string(py::extract<std::string>(py::str(key[0].attr("__class__")))));
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
        py::scope scope = flow_module;

        py::class_<Filter<PropertyT,GeometryT>, shared_ptr<Filter<PropertyT,GeometryT>>, py::bases<Solver>, boost::noncopyable>
        filter_class((type_name<PropertyT>()+"Filter"+suffix).c_str(),
                     format(
                         "%1%Filter%3%(geometry)     \n\n"
                         "Data filter for %1% into %2% geometry.\n\n"
                         "Args:\n"
                         "    geometry (geometry.%2%): Target geometry.\n\n"
                         "See also:\n    :mod:`plask.filter` for details on filter usage.",
                         std::string(PropertyT::NAME), spaceName<GeometryT>(), suffix).c_str(),
                     py::init<shared_ptr<GeometryT>>((py::arg("geometry")))
                    );
        filter_class
            .def_readonly("out",
                          reinterpret_cast<ProviderFor<PropertyT, GeometryT> Filter<PropertyT,GeometryT>::*>(&Filter<PropertyT,GeometryT>::out),
                          format("Filter output provider.\n\nExample:\n    >>> some_solver.in%1% = my_filter.out\n", type_name<PropertyT>()).c_str())
            .def("__getitem__", &FilterIn<PropertyT,GeometryT>::template __getsetitem__<FilterinGetitemResult>)
            .def("__setitem__", &FilterIn<PropertyT,GeometryT>::template __getsetitem__<FilterinSetitemResult, py::object>)
        ;

        return filter_class;
    }

}


template <typename PropertyT>
void registerFilters() {

    detail::registerFilterImpl<PropertyT,Geometry2DCartesian>("2D");
    detail::registerFilterImpl<PropertyT,Geometry2DCylindrical>("Cyl");
    detail::registerFilterImpl<PropertyT,Geometry3D>("3D");

}


}} // namespace plask::python

#endif // PLASK__PYTHON_FILTERS_H
