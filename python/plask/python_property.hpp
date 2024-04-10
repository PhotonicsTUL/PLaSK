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
#ifndef PLASK__PYTHON_PROPERTYTAGS_H
#define PLASK__PYTHON_PROPERTYTAGS_H

#include "python_globals.hpp"
#include "python_provider.hpp"
#include "python_filters.hpp"

namespace plask { namespace python {

namespace detail {

    template <typename PropertyT, PropertyType propertyType, bool filters>
    struct RegisterPropertyImpl;

    template <typename PropertyT, bool filters>
    struct RegisterPropertyImpl<PropertyT,SINGLE_VALUE_PROPERTY,filters> {
        inline static void call(const py::object& flow_module) {
            registerProvider<ProviderFor<PropertyT,void>>(flow_module);
            registerReceiver<ReceiverFor<PropertyT,void>>(flow_module);
        }
    };

    template <typename PropertyT, bool filters>
    struct RegisterPropertyImpl<PropertyT,MULTI_VALUE_PROPERTY,filters> {
        inline static void call(const py::object& flow_module) {
            registerProvider<ProviderFor<PropertyT,void>>(flow_module);
            registerReceiver<ReceiverFor<PropertyT,void>>(flow_module);
        }
    };


    template <typename PropertyT>
    struct RegisterPropertyImpl<PropertyT,FIELD_PROPERTY,false> {
        inline static void call(const py::object& flow_module) {
            registerProvider<ProviderFor<PropertyT,Geometry2DCartesian>>(flow_module);
            registerProvider<ProviderFor<PropertyT,Geometry2DCylindrical>>(flow_module);
            registerProvider<ProviderFor<PropertyT,Geometry3D>>(flow_module);

            registerReceiver<ReceiverFor<PropertyT,Geometry2DCartesian>>(flow_module);
            registerReceiver<ReceiverFor<PropertyT,Geometry2DCylindrical>>(flow_module);
            registerReceiver<ReceiverFor<PropertyT,Geometry3D>>(flow_module);
        }
    };

    template <typename PropertyT>
    struct RegisterPropertyImpl<PropertyT,MULTI_FIELD_PROPERTY,false> {
        inline static void call(const py::object& flow_module) {
            registerProvider<ProviderFor<PropertyT,Geometry2DCartesian>>(flow_module);
            registerProvider<ProviderFor<PropertyT,Geometry2DCylindrical>>(flow_module);
            registerProvider<ProviderFor<PropertyT,Geometry3D>>(flow_module);

            registerReceiver<ReceiverFor<PropertyT,Geometry2DCartesian>>(flow_module);
            registerReceiver<ReceiverFor<PropertyT,Geometry2DCylindrical>>(flow_module);
            registerReceiver<ReceiverFor<PropertyT,Geometry3D>>(flow_module);
        }
    };
    template <typename PropertyT>
    struct RegisterPropertyImpl<PropertyT,FIELD_PROPERTY,true> {
        inline static void call(const py::object& flow_module) {
            RegisterPropertyImpl<PropertyT,FIELD_PROPERTY,false>::call(flow_module);
            registerFilters<PropertyT>(flow_module);
        }
    };

    template <typename PropertyT>
    struct RegisterPropertyImpl<PropertyT,MULTI_FIELD_PROPERTY,true> {
        inline static void call(const py::object& flow_module) {
            RegisterPropertyImpl<PropertyT,MULTI_FIELD_PROPERTY,false>::call(flow_module);
            registerFilters<PropertyT>(flow_module);
        }
    };
}

template <typename PropertyT, bool filters=true>
inline void registerProperty(const py::object& flow_module) {
    detail::RegisterPropertyImpl<PropertyT, PropertyT::propertyType, filters>::call(flow_module);
}

}} // namespace plask

#endif // PLASK__PYTHON_PROPERTYTAGS_H
