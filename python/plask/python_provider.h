#ifndef PLASK__PYTHON_PROVIDER_H
#define PLASK__PYTHON_PROVIDER_H

#include "python_globals.h"
#include <plask/provider/provider.h>

namespace plask { namespace python {

// TODO: Sprawdź czy klasy są już zarejestrowane w Pythonie i jeśli tak to wyjdź



namespace detail {

    template <typename PropertyTag, typename ValueType, typename SpaceType>
    struct RegisterProviderBase
    {
        typedef ProviderFor<PropertyTag,SpaceType> ProviderType;
        typedef ReceiverFor<PropertyTag,SpaceType> ReceiverType;

        const std::string property_name;

        py::class_<ProviderType, boost::noncopyable> provider_class;
        py::class_<ReceiverType, boost::noncopyable> receiver_class;

        static void connect(ReceiverType& receiver, ProviderType* provider) { receiver.setProvider(provider); }
        static void disconnect(ReceiverType& receiver) { receiver.setProvider(nullptr); }

        RegisterProviderBase() :
            property_name ([](const std::string& s){size_t n=s.find_last_of(':'); return (n!=s.npos)?s.substr(n+1):s; }(py::type_id<PropertyTag>().name())),
            provider_class(("ProviderFor" + property_name).c_str(), py::no_init),
            receiver_class(("ReceiverFor" + property_name).c_str(), py::no_init)
        {
            receiver_class.def("__lshift__", &connect, "Connect provider to receiver");
            receiver_class.def("__rrshift__", &connect, "Connect provider to receiver");
            receiver_class.def("connect", &connect, "Connect provider to receiver");
            receiver_class.def("disconnect", &disconnect, "Disconnect any provider from receiver");
        }
    };

    template <typename PropertyTag, typename ValueType, PropertyType propertyType, typename SpaceType>
    struct RegisterProviderImpl: public RegisterProviderBase<PropertyTag, ValueType, SpaceType> {};

    template <typename PropertyTag, typename ValueType, typename SpaceType>
    struct RegisterProviderImpl<PropertyTag, ValueType, SINGLE_VALUE_PROPERTY, SpaceType>:
    public RegisterProviderBase<PropertyTag, ValueType, SpaceType>
    {
        typedef ProviderFor<PropertyTag,SpaceType> ProviderType;
        typedef ReceiverFor<PropertyTag,SpaceType> ReceiverType;

        template <typename What> static ValueType __call__(What& self) { return self(); }

        RegisterProviderImpl() {
            this->provider_class.def("__call__", &__call__<ProviderType>, "Get value from the provider");
            this->receiver_class.def("__call__", &__call__<ReceiverType>, "Get value from the connected provider");
            py::class_<typename ProviderType::WithDefaultValue, py::bases<ProviderType>, boost::noncopyable>(("ProviderFor" + this->property_name).c_str());
            py::class_<typename ProviderType::WithValue, py::bases<ProviderType>, boost::noncopyable>(("ProviderFor" + this->property_name).c_str());
            py::class_<typename ProviderType::Delegate, py::bases<ProviderType>, boost::noncopyable>(("ProviderFor" + this->property_name).c_str());
        }
    };

    template <typename PropertyTag, typename ValueType, typename SpaceType>
    struct RegisterProviderImpl<PropertyTag, ValueType, FIELD_PROPERTY, SpaceType>:
    public RegisterProviderBase<PropertyTag, ValueType, SpaceType>
    {
        typedef ProviderFor<PropertyTag,SpaceType> ProviderType;
        typedef ReceiverFor<PropertyTag,SpaceType> ReceiverType;

        template <typename What> static ValueType __call__(What& self, const Mesh<SpaceType::DIMS>& mesh) { return self(mesh); }

        RegisterProviderImpl() {
            this->provider_class.def("__call__", &__call__<ProviderType>, "Get value from the provider", (py::arg("mesh")));
            this->receiver_class.def("__call__", &__call__<ReceiverType>, "Get value from the connected provider", (py::arg("mesh")));
            py::class_<typename ProviderType::Delegate, py::bases<ProviderType>, boost::noncopyable>(("ProviderFor" + this->property_name).c_str());
        }
    };

    template <typename PropertyTag, typename ValueType, typename SpaceType>
    struct RegisterProviderImpl<PropertyTag, ValueType, INTERPOLATED_FIELD_PROPERTY, SpaceType>:
    public RegisterProviderBase<PropertyTag, ValueType, SpaceType> {
        typedef ProviderFor<PropertyTag,SpaceType> ProviderType;
        typedef ReceiverFor<PropertyTag,SpaceType> ReceiverType;

        template <typename What> static ValueType __call__(What& self, const Mesh<SpaceType::DIMS>& mesh, InterpolationMethod method) { return self(mesh, method); }

        RegisterProviderImpl() {
            this->provider_class.def("__call__", &__call__<ProviderType>, "Get value from the provider", (py::arg("mesh"), py::arg("interpolation")=DEFAULT_INTERPOLATION));
            this->receiver_class.def("__call__", &__call__<ReceiverType>, "Get value from the connected provider", (py::arg("mesh"), py::arg("interpolation")=DEFAULT_INTERPOLATION));
            py::class_<typename ProviderType::WithValue, py::bases<ProviderType>, boost::noncopyable>(("ProviderFor" + this->property_name).c_str());
            py::class_<typename ProviderType::Delegate, py::bases<ProviderType>, boost::noncopyable>(("ProviderFor" + this->property_name).c_str());
        }
    };


} // namespace detail


template <typename PropertyTag, typename SpaceType=void>
struct RegisterProvider: public detail::RegisterProviderImpl<PropertyTag, typename PropertyTag::ValueType, PropertyTag::propertyType, SpaceType>  {
    /// Delegate all constructors to parent class.
    template<typename ...Args>
    RegisterProvider(Args&&... params)
    : detail::RegisterProviderImpl<PropertyTag, typename PropertyTag::ValueType, PropertyTag::propertyType, SpaceType>(std::forward<Args>(params)...) {
    }
};

}} // namespace plask::python

#endif // PLASK__PYTHON_PROVIDER_H

