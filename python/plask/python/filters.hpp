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
#ifndef PLASK__PYTHON_FILTERS_H
#define PLASK__PYTHON_FILTERS_H

#include <plask/filters/filter.hpp>

#include "globals.hpp"
#include "provider.hpp"


namespace plask { namespace python {

namespace detail {

    void PLASK_PYTHON_API filterin_parse_key(const py::object& key, shared_ptr<GeometryObject>& geom, PathHints*& path, size_t& points);

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
        static inline PyObject* call(const py::object& PLASK_UNUSED(self), ReceiverT& receiver, const py::object& value) {
            typedef detail::RegisterReceiverImpl<ReceiverT, ReceiverT::PropertyTag::propertyType, typename  ReceiverT::PropertyTag::ExtraParams> RegisterT;
            RegisterT::setter(receiver, value);
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
            size_t points;
            filterin_parse_key(key, geom, path, points);

            if (auto geomd = dynamic_pointer_cast<GeometryObjectD<2>>(geom))
                return Fun::call(self, filter->input(*geomd, path), value...);
            if (auto geomd = dynamic_pointer_cast<Geometry2DCartesian>(geom))
                return Fun::call(self, filter->input(*geomd, path), value...);

            if (auto geomd = dynamic_pointer_cast<GeometryObjectD<3>>(geom))
                return Fun::call(self, filter->setOuter(*geomd, path, points), value...);
            if (auto geomd = dynamic_pointer_cast<Geometry3D>(geom))
                return Fun::call(self, filter->setOuter(*geomd->getChild(), path, points), value...);

            throw TypeError(u8"wrong geometry type '{0}'", std::string(py::extract<std::string>(py::str(key[0].attr("__class__")))));
            return nullptr;
        }
    };

    template <typename PropertyT> struct FilterIn<PropertyT,Geometry2DCylindrical> {
        template <typename Fun, typename... Args>
        static PyObject* __getsetitem__(const py::object& self, const py::object& key, Args... value) {
            Filter<PropertyT,Geometry2DCylindrical>* filter = py::extract<Filter<PropertyT,Geometry2DCylindrical>*>(self);
            shared_ptr<GeometryObject> geom;
            PathHints* path;
            size_t points;
            filterin_parse_key(key, geom, path, points);

            if (auto geomd = dynamic_pointer_cast<GeometryObjectD<2>>(geom))
                 return Fun::call(self, filter->input(*geomd, path), value...);
            if (auto geomd = dynamic_pointer_cast<Geometry2DCylindrical>(geom))
                 return Fun::call(self, filter->input(*geomd, path), value...);

            if (auto geomd = dynamic_pointer_cast<GeometryObjectD<3>>(geom))
                return Fun::call(self, filter->setOuter(*geomd, path, points), value...);
            if (auto geomd = dynamic_pointer_cast<Geometry3D>(geom))
                return Fun::call(self, filter->setOuter(*geomd->getChild(), path, points), value...);

            throw TypeError(u8"wrong geometry type '{0}'", std::string(py::extract<std::string>(py::str(key[0].attr("__class__")))));
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
            size_t points;
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

            throw TypeError(u8"wrong geometry type '{0}'", std::string(py::extract<std::string>(py::str(key[0].attr("__class__")))));
            return nullptr;
        }
    };

    template <typename PropertyT, typename GeometryT>
    FilterIn<PropertyT,GeometryT>* getFilterIn(Filter<PropertyT,GeometryT>& filter) {
        return new FilterIn<PropertyT,GeometryT>(filter);
    }

    template <typename PropertyT, typename GeometryT>
    py::class_<Filter<PropertyT,GeometryT>, shared_ptr<Filter<PropertyT,GeometryT>>, py::bases<Solver>, boost::noncopyable>
    registerFilterImpl(const char* suffix, const py::object& flow_module)
    {
        py::scope scope = flow_module;
        (void) scope;   // don't warn about unused variable scope

        std::string out_name = "out" + type_name<PropertyT>();

        py::class_<Filter<PropertyT,GeometryT>, shared_ptr<Filter<PropertyT,GeometryT>>, py::bases<Solver>, boost::noncopyable>
        filter_class((type_name<PropertyT>()+"Filter"+suffix).c_str(),
                     format(
                         u8"{0}Filter{3}(geometry)\n\n"
                         u8"Data filter for {1} into {2} geometry.\n\n"
                         u8"Args:\n"
                         u8"    geometry (geometry.{2}): Target geometry.\n\n"
                         u8"See also:\n    :mod:`plask.filter` for details on filter usage.",
                         type_name<PropertyT>(), std::string(PropertyT::NAME), spaceName<GeometryT>(), suffix).c_str(),
                     py::init<shared_ptr<GeometryT>>((py::arg("geometry")))
                    );
        filter_class
            .def_readonly(out_name.c_str(),
                          reinterpret_cast<ProviderFor<PropertyT, GeometryT> Filter<PropertyT,GeometryT>::*>(&Filter<PropertyT,GeometryT>::out),
                          format(u8"Filter output provider.\n\nExample:\n    >>> some_solver.in{0} = my_filter.out{0}\n",
                                 type_name<PropertyT>()).c_str())
            .def("__getitem__", &FilterIn<PropertyT,GeometryT>::template __getsetitem__<FilterinGetitemResult>)
            .def("__setitem__", &FilterIn<PropertyT,GeometryT>::template __getsetitem__<FilterinSetitemResult, py::object>)
        ;
        filter_class.attr("out") = filter_class.attr(out_name.c_str());

        return filter_class;
    }

}


template <typename PropertyT>
void registerFilters(const py::object& flow_module) {

    detail::registerFilterImpl<PropertyT,Geometry2DCartesian>("2D", flow_module);
    detail::registerFilterImpl<PropertyT,Geometry2DCylindrical>("Cyl", flow_module);
    detail::registerFilterImpl<PropertyT,Geometry3D>("3D", flow_module);

}


}} // namespace plask::python

#endif // PLASK__PYTHON_FILTERS_H
