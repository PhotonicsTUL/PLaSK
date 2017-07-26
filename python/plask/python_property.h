#ifndef PLASK__PYTHON_PROPERTYTAGS_H
#define PLASK__PYTHON_PROPERTYTAGS_H

#include "python_globals.h"
#include "python_provider.h"
#include "python_filters.h"

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
