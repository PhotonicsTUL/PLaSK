#ifndef PLASK__PYTHON_PROVIDER_H
#define PLASK__PYTHON_PROVIDER_H

#include "python_globals.h"
#include <plask/provider/provider.h>

namespace plask { namespace python {

template <typename T, int dim>
struct DataVectorWrap : public DataVector<T> {
    shared_ptr<Mesh<dim>> mesh;
    DataVectorWrap(const DataVector<T>& src, const shared_ptr<Mesh<dim>>& mesh) : DataVector<T>(src), mesh(mesh) {}
    DataVectorWrap(const DataVector<T>& src) : DataVector<T>(src) {}
    DataVectorWrap() = default;
    DataVectorWrap(const DataVectorWrap<T,dim>& src) = default;
};


namespace detail {

    template <typename PropertyTag, typename ValueType, typename SpaceType>
    struct RegisterProviderReceiverBase
    {
        typedef ProviderFor<PropertyTag,SpaceType> ProviderType;
        typedef ReceiverFor<PropertyTag,SpaceType> ReceiverType;

        const std::string property_name;

        py::class_<ProviderType, boost::noncopyable> provider_class;
        py::class_<ReceiverType, boost::noncopyable> receiver_class;

        static void connect(ReceiverType& receiver, ProviderType* provider) { receiver.setProvider(provider); }
        static void disconnect(ReceiverType& receiver) { receiver.setProvider(nullptr); }

        RegisterProviderReceiverBase() :
            property_name ([](const std::string& s){size_t n=s.find_last_of(':'); return (n!=s.npos)?s.substr(n+1):s; }(py::type_id<PropertyTag>().name())),
            provider_class(("ProviderFor" + property_name + "Base").c_str(), py::no_init),
            receiver_class(("ReceiverFor" + property_name).c_str(), py::no_init)
        {
            receiver_class.def("__lshift__", &connect, "Connect provider to receiver");
            receiver_class.def("__rrshift__", &connect, "Connect provider to receiver");
            receiver_class.def("connect", &connect, "Connect provider to receiver");
            receiver_class.def("disconnect", &disconnect, "Disconnect any provider from receiver");
        }
    };

    template <typename PropertyTag, typename ValueType, PropertyType propertyType, typename SpaceType>
    struct RegisterProviderReceiverImpl: public RegisterProviderReceiverBase<PropertyTag, ValueType, SpaceType> {};

    template <typename PropertyTag, typename ValueType, typename SpaceType>
    struct RegisterProviderReceiverImpl<PropertyTag, ValueType, SINGLE_VALUE_PROPERTY, SpaceType>:
    public RegisterProviderReceiverBase<PropertyTag, ValueType, SpaceType>
    {
        typedef ProviderFor<PropertyTag,SpaceType> ProviderType;
        typedef ReceiverFor<PropertyTag,SpaceType> ReceiverType;

        template <typename What> static ValueType __call__(What& self) { return self(); }

        RegisterProviderReceiverImpl() {
            this->provider_class.def("__call__", &__call__<ProviderType>, "Get value from the provider");
            this->receiver_class.def("__call__", &__call__<ReceiverType>, "Get value from the connected provider");
            py::class_<typename ProviderType::WithDefaultValue, py::bases<ProviderType>, boost::noncopyable>(("ProviderFor" + this->property_name).c_str());
            py::class_<typename ProviderType::WithValue, py::bases<ProviderType>, boost::noncopyable>(("ProviderFor" + this->property_name).c_str());
            py::class_<typename ProviderType::Delegate, py::bases<ProviderType>, boost::noncopyable>(("ProviderFor" + this->property_name).c_str());
        }
    };

    template <typename PropertyTag, typename ValueType, typename SpaceType>
    struct RegisterProviderReceiverImpl<PropertyTag, ValueType, FIELD_PROPERTY, SpaceType>:
    public RegisterProviderReceiverBase<PropertyTag, ValueType, SpaceType>
    {
        typedef ProviderFor<PropertyTag,SpaceType> ProviderType;
        typedef ReceiverFor<PropertyTag,SpaceType> ReceiverType;

        template <typename What, int dim> static DataVectorWrap<ValueType,SpaceType::DIMS> __call__(What& self, const shared_ptr<Mesh<SpaceType::DIMS>>& mesh) {
            return DataVectorWrap<ValueType,SpaceType::DIMS>(self(*mesh), mesh);
        }

        RegisterProviderReceiverImpl() {
            this->provider_class.def("__call__", &__call__<ProviderType>, "Get value from the provider", (py::arg("mesh")));
            this->receiver_class.def("__call__", &__call__<ReceiverType>, "Get value from the connected provider", (py::arg("mesh")));
            py::class_<typename ProviderType::Delegate, py::bases<ProviderType>, boost::noncopyable>(("ProviderFor" + this->property_name).c_str());
        }
    };

    template <typename PropertyTag, typename ValueType, typename SpaceType>
    struct RegisterProviderReceiverImpl<PropertyTag, ValueType, INTERPOLATED_FIELD_PROPERTY, SpaceType>:
    public RegisterProviderReceiverBase<PropertyTag, ValueType, SpaceType> {
        typedef ProviderFor<PropertyTag,SpaceType> ProviderType;
        typedef ReceiverFor<PropertyTag,SpaceType> ReceiverType;

        template <typename What> static DataVectorWrap<ValueType,SpaceType::DIMS> __call__(What& self, const shared_ptr<Mesh<SpaceType::DIMS>>& mesh) {
            return DataVectorWrap<ValueType,SpaceType::DIMS>(self(*mesh), mesh);
        }

        RegisterProviderReceiverImpl() {
            this->provider_class.def("__call__", &__call__<ProviderType>, "Get value from the provider", (py::arg("mesh"), py::arg("interpolation")=DEFAULT_INTERPOLATION));
            this->receiver_class.def("__call__", &__call__<ReceiverType>, "Get value from the connected provider", (py::arg("mesh"), py::arg("interpolation")=DEFAULT_INTERPOLATION));
            py::class_<typename ProviderType::Delegate, py::bases<ProviderType>, boost::noncopyable>(("ProviderFor" + this->property_name).c_str());
        }

        template <typename MeshP>
        RegisterProviderReceiverImpl<PropertyTag, ValueType, INTERPOLATED_FIELD_PROPERTY, SpaceType>& WithValue() {
            py::class_<typename ProviderType::template WithValue<MeshP>, py::bases<ProviderType>, boost::noncopyable>(("ProviderFor" + this->property_name).c_str());
            return *this;
        }
    };


} // namespace detail


template <typename PropertyTag, typename SpaceType>
struct RegisterProviderReceiver: public detail::RegisterProviderReceiverImpl<PropertyTag, typename PropertyTag::ValueType, PropertyTag::propertyType, SpaceType>  {
    /// Delegate all constructors to parent class.
    template<typename ...Args>
    RegisterProviderReceiver(Args&&... params)
    : detail::RegisterProviderReceiverImpl<PropertyTag, typename PropertyTag::ValueType, PropertyTag::propertyType, SpaceType>(std::forward<Args>(params)...) {
    }
};

}} // namespace plask::python

#endif // PLASK__PYTHON_PROVIDER_H

