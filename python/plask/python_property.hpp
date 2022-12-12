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
        inline static void call() {
            registerProvider<ProviderFor<PropertyT,void>>();
            registerReceiver<ReceiverFor<PropertyT,void>>();
        }
    };

    template <typename PropertyT, bool filters>
    struct RegisterPropertyImpl<PropertyT,MULTI_VALUE_PROPERTY,filters> {
        inline static void call() {
            registerProvider<ProviderFor<PropertyT,void>>();
            registerReceiver<ReceiverFor<PropertyT,void>>();
        }
    };


    template <typename PropertyT>
    struct RegisterPropertyImpl<PropertyT,FIELD_PROPERTY,false> {
        inline static void call() {
            registerProvider<ProviderFor<PropertyT,Geometry2DCartesian>>();
            registerProvider<ProviderFor<PropertyT,Geometry2DCylindrical>>();
            registerProvider<ProviderFor<PropertyT,Geometry3D>>();

            registerReceiver<ReceiverFor<PropertyT,Geometry2DCartesian>>();
            registerReceiver<ReceiverFor<PropertyT,Geometry2DCylindrical>>();
            registerReceiver<ReceiverFor<PropertyT,Geometry3D>>();
        }
    };

    template <typename PropertyT>
    struct RegisterPropertyImpl<PropertyT,MULTI_FIELD_PROPERTY,false> {
        inline static void call() {
            registerProvider<ProviderFor<PropertyT,Geometry2DCartesian>>();
            registerProvider<ProviderFor<PropertyT,Geometry2DCylindrical>>();
            registerProvider<ProviderFor<PropertyT,Geometry3D>>();

            registerReceiver<ReceiverFor<PropertyT,Geometry2DCartesian>>();
            registerReceiver<ReceiverFor<PropertyT,Geometry2DCylindrical>>();
            registerReceiver<ReceiverFor<PropertyT,Geometry3D>>();
        }
    };
    template <typename PropertyT>
    struct RegisterPropertyImpl<PropertyT,FIELD_PROPERTY,true> {
        inline static void call() {
            RegisterPropertyImpl<PropertyT,FIELD_PROPERTY,false>::call();
            registerFilters<PropertyT>();
        }
    };

    template <typename PropertyT>
    struct RegisterPropertyImpl<PropertyT,MULTI_FIELD_PROPERTY,true> {
        inline static void call() {
            RegisterPropertyImpl<PropertyT,MULTI_FIELD_PROPERTY,false>::call();
            registerFilters<PropertyT>();
        }
    };
}

template <typename PropertyT, bool filters=true>
inline void registerProperty() {
    detail::RegisterPropertyImpl<PropertyT, PropertyT::propertyType, filters>::call();
}

}} // namespace plask

#endif // PLASK__PYTHON_PROPERTYTAGS_H
