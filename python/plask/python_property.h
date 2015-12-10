#ifndef PLASK__PYTHON_PROPERTYTAGS_H
#define PLASK__PYTHON_PROPERTYTAGS_H

#include "python_globals.h"
#include "python_provider.h"
#include "python_filters.h"

#include <plask/properties/gain.h>

namespace plask { namespace python {

namespace detail {

    template <typename PropertyT, PropertyType propertyType>
    struct RegisterPropertyImpl;

    template <typename PropertyT>
    struct RegisterPropertyImpl<PropertyT,SINGLE_VALUE_PROPERTY> {
        static void call() {
            registerProvider<ProviderFor<PropertyT,void>>();
            registerReceiver<ReceiverFor<PropertyT,void>>();
        }
    };

    template <typename PropertyT>
    struct RegisterPropertyImpl<PropertyT,MULTI_VALUE_PROPERTY> {
        static void call() {
            registerProvider<ProviderFor<PropertyT,void>>();
            registerReceiver<ReceiverFor<PropertyT,void>>();
        }
    };

    template <typename PropertyT>
    struct RegisterPropertyImpl<PropertyT,FIELD_PROPERTY> {
        static void call() {
            registerProvider<ProviderFor<PropertyT,Geometry2DCartesian>>();
            registerProvider<ProviderFor<PropertyT,Geometry2DCylindrical>>();
            registerProvider<ProviderFor<PropertyT,Geometry3D>>();

            registerReceiver<ReceiverFor<PropertyT,Geometry2DCartesian>>();
            registerReceiver<ReceiverFor<PropertyT,Geometry2DCylindrical>>();
            registerReceiver<ReceiverFor<PropertyT,Geometry3D>>();

            registerFilters<PropertyT>();
        }
    };

    template <typename PropertyT>
    struct RegisterPropertyImpl<PropertyT,MULTI_FIELD_PROPERTY> {
        static void call() {
            registerProvider<ProviderFor<PropertyT,Geometry2DCartesian>>();
            registerProvider<ProviderFor<PropertyT,Geometry2DCylindrical>>();
            registerProvider<ProviderFor<PropertyT,Geometry3D>>();

            registerReceiver<ReceiverFor<PropertyT,Geometry2DCartesian>>();
            registerReceiver<ReceiverFor<PropertyT,Geometry2DCylindrical>>();
            registerReceiver<ReceiverFor<PropertyT,Geometry3D>>();

            registerFilters<PropertyT>();
        }
    };
}

template <typename PropertyT>
inline void registerProperty() {
    detail::RegisterPropertyImpl<PropertyT, PropertyT::propertyType>::call();
}

}} // namespace plask

#endif // PLASK__PYTHON_PROPERTYTAGS_H
