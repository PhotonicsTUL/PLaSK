#ifndef PLASK__PYTHON_PROPERTYTAGS_H
#define PLASK__PYTHON_PROPERTYTAGS_H

#include "python_globals.h"
#include "python_provider.h"
#include "python_filters.h"

namespace plask { namespace python {

namespace detail {

    template <typename PropertyT, PropertyType propertyType>
    struct RegisterPropertyImpl;

    template <typename PropertyT>
    struct RegisterPropertyImpl<PropertyT,SINGLE_VALUE_PROPERTY> {
        static void call() {
            RegisterProvider<ProviderFor<PropertyT,void>>();
            RegisterReceiver<ReceiverFor<PropertyT,void>>();
        }
    };

    template <typename PropertyT>
    struct RegisterPropertyImpl<PropertyT,FIELD_PROPERTY> {
        static void call() {
            RegisterProvider<ProviderFor<PropertyT,Geometry2DCartesian>>();
            RegisterProvider<ProviderFor<PropertyT,Geometry2DCylindrical>>();
            RegisterProvider<ProviderFor<PropertyT,Geometry3D>>();

            RegisterReceiver<ReceiverFor<PropertyT,Geometry2DCartesian>>();
            RegisterReceiver<ReceiverFor<PropertyT,Geometry2DCylindrical>>();
            RegisterReceiver<ReceiverFor<PropertyT,Geometry3D>>();
        }
    };
}

template <typename PropertyT>
inline void registerProperty() {
    detail::RegisterPropertyImpl<PropertyT, PropertyT::propertyType>::call();
}

}} // namespace plask

#endif // PLASK__PYTHON_PROPERTYTAGS_H
