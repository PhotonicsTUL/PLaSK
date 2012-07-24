#ifndef PLASK__PYTHON_PROVIDER_H
#define PLASK__PYTHON_PROVIDER_H

#include <type_traits>  // std::is_same

#include "python_globals.h"
#include <plask/provider/provider.h>
#include <plask/mesh/rectilinear.h>
#include <plask/mesh/regular.h>

namespace plask { namespace python {

template <typename T, int dim>
struct DataVectorWrap : public DataVector<T> {
    shared_ptr<MeshD<dim>> mesh;
    bool mesh_changed;

    DataVectorWrap(const DataVector<T>& src, const shared_ptr<MeshD<dim>>& mesh) : DataVector<T>(src), mesh(mesh), mesh_changed(false) {
        mesh->changedConnectMethod(this, &DataVectorWrap<T,dim>::onMeshChanged);
    }

    DataVectorWrap(const DataVector<T>& src) : DataVector<T>(src) {}

    DataVectorWrap() = default;
    DataVectorWrap(const DataVectorWrap<T,dim>& src) : DataVector<T>(src), mesh(src.mesh), mesh_changed(src.mesh_changed) {
        if (mesh) mesh->changedConnectMethod(this, &DataVectorWrap<T,dim>::onMeshChanged);
    }

    ~DataVectorWrap() {
        mesh->changedDisconnectMethod(this, &DataVectorWrap<T,dim>::onMeshChanged);
    };

    void onMeshChanged(const typename MeshD<dim>::Event& event) { mesh_changed = true; }
};


namespace detail {


    template <typename ReceiverT>
    struct RegisterReceiverBase {
        typedef ProviderFor<typename ReceiverT::PropertyTag,typename ReceiverT::SpaceType> ProviderT;
        const std::string property_name;
        py::class_<ReceiverT, boost::noncopyable> receiver_class;
        static void connect(ReceiverT& receiver, ProviderT* provider) { receiver.setProvider(provider); }
        static shared_ptr<ProviderT> rconnect(ReceiverT& receiver, const shared_ptr<ProviderT>& provider) {
            receiver.setProvider(provider.get()); return provider;
        }
        static void disconnect(ReceiverT& receiver) { receiver.setProvider(nullptr); }
        RegisterReceiverBase() :
            property_name([](const std::string& s)->std::string{size_t n=s.find_last_of(':'); return (n!=s.npos)?s.substr(n+1):s; }(py::type_id<typename ReceiverT::PropertyTag>().name())),
            receiver_class(("ReceiverFor" + property_name).c_str(), py::no_init) {
            receiver_class.def("__lshift__", &connect, "Connect provider to receiver");
            receiver_class.def("__rrshift__", &rconnect, "Connect provider to receiver");
            receiver_class.def("connect", &connect, "Connect provider to receiver");
            receiver_class.def("disconnect", &disconnect, "Disconnect any provider from receiver");
            py::delattr(py::scope(), ("ReceiverFor" + property_name).c_str());
        }
    };

    template <typename ReceiverT, PropertyType propertyType>
    struct RegisterReceiverImpl: public RegisterReceiverBase<ReceiverT> {};

    template <typename ReceiverT>
    struct RegisterReceiverImpl<ReceiverT, SINGLE_VALUE_PROPERTY> : public RegisterReceiverBase<ReceiverT>
    {
        typedef typename ReceiverT::PropertyTag::ValueType ValueT;
        static ValueT __call__(ReceiverT& self) { return self(); }
        static void setValue(ReceiverT& self, const py::object& obj) { throw TypeError("Operation not allowed for single value receiver"); }
        RegisterReceiverImpl() {
            this->receiver_class.def("__call__", &__call__, "Get value from the connected provider");
        }
    };

    template <typename ReceiverT>
    struct RegisterReceiverImpl<ReceiverT, FIELD_PROPERTY> : public RegisterReceiverBase<ReceiverT>
    {
        typedef typename ReceiverT::PropertyTag::ValueType ValueT;
        static const int dim = ReceiverT::SpaceType::DIMS;
        static DataVectorWrap<ValueT,dim> __call__(ReceiverT& self, const shared_ptr<MeshD<dim>>& mesh) {
            return DataVectorWrap<ValueT,dim>(self(*mesh), mesh);
        }
        static void setValue(ReceiverT& self, const py::object& obj) { throw TypeError("Operation not allowed for non-interpolated field receiver"); }
        RegisterReceiverImpl() {
            this->receiver_class.def("__call__", &__call__, "Get value from the connected provider", (py::arg("mesh")));
        }
    };

    template <typename ReceiverT>
    struct RegisterReceiverImpl<ReceiverT, INTERPOLATED_FIELD_PROPERTY> : public RegisterReceiverBase<ReceiverT> {
        typedef typename ReceiverT::PropertyTag::ValueType ValueT;
        static const int dim = ReceiverT::SpaceType::DIMS;
        static DataVectorWrap<ValueT,dim> __call__(ReceiverT& self, const shared_ptr<MeshD<dim>>& mesh, InterpolationMethod method) {
            return DataVectorWrap<ValueT,dim>(self(*mesh, method), mesh);
        }

        static void setValue(ReceiverT& self, const py::object& obj) {
            DataVectorWrap<ValueT,dim> data = py::extract<DataVectorWrap<ValueT,dim>>(obj);
            if (dim == 2) {
                auto rectilinear_mesh = dynamic_pointer_cast<RectilinearMesh2D>(data.mesh);
                if (rectilinear_mesh) { self.setValue(data, rectilinear_mesh); return; }
            }
        }

        RegisterReceiverImpl() {
            this->receiver_class.def("__call__", &__call__, "Get value from the connected provider", (py::arg("mesh"), py::arg("interpolation")=DEFAULT_INTERPOLATION));
            this->receiver_class.def("setValue", &setValue, "Set previously obtained value", (py::arg("data")));
        }
    };


    template <typename ProviderT>
    struct RegisterProviderBase
    {
        const std::string property_name;
        typedef ProviderFor<typename ProviderT::PropertyTag, typename ProviderT::SpaceType> ProviderBase;
        py::class_<ProviderBase, boost::noncopyable> provider_base_class;
        py::class_<ProviderT, py::bases<ProviderBase>, boost::noncopyable> provider_class;
        RegisterProviderBase() :
            property_name ([](const std::string& s)->std::string{size_t n=s.find_last_of(':'); return (n!=s.npos)?s.substr(n+1):s; }(py::type_id<typename ProviderT::PropertyTag>().name())),
            provider_base_class(("ProviderFor" + property_name + "Base").c_str(), py::no_init),
            provider_class(("ProviderFor" + property_name).c_str(), py::no_init) {
            py::delattr(py::scope(), ("ProviderFor" + property_name + "Base").c_str());
            py::delattr(py::scope(), ("ProviderFor" + property_name).c_str());
        }
    };

    template <typename ProviderT, PropertyType propertyType>
    struct RegisterProviderImpl : public RegisterProviderBase<ProviderT> {};

    template <typename ProviderT>
    struct RegisterProviderImpl<ProviderT, SINGLE_VALUE_PROPERTY> : public RegisterProviderBase<ProviderT>
    {
        typedef typename ProviderT::PropertyTag::ValueType ValueT;
        static ValueT __call__(ProviderT& self) { return self(); }
        RegisterProviderImpl() {
            this->provider_class.def("__call__", &__call__, "Get value from the provider");
        }
    };

    template <typename ProviderT>
    struct RegisterProviderImpl<ProviderT, FIELD_PROPERTY> : public RegisterProviderBase<ProviderT>
    {
        typedef typename ProviderT::PropertyTag::ValueType ValueT;
        static const int dim = ProviderT::SpaceType::DIMS;
        static DataVectorWrap<ValueT,dim> __call__(ProviderT& self, const shared_ptr<MeshD<dim>>& mesh) {
            return DataVectorWrap<ValueT,dim>(self(*mesh), mesh);
        }
        RegisterProviderImpl() {
            this->provider_class.def("__call__", &__call__, "Get value from the provider", (py::arg("mesh")));
        }
    };

    template <typename ProviderT>
    struct RegisterProviderImpl<ProviderT, INTERPOLATED_FIELD_PROPERTY> : public RegisterProviderBase<ProviderT>
    {
        typedef typename ProviderT::PropertyTag::ValueType ValueT;
        static const int dim = ProviderT::SpaceType::DIMS;
        static DataVectorWrap<ValueT,dim> __call__(ProviderT& self, const shared_ptr<MeshD<dim>>& mesh, InterpolationMethod method) {
            return DataVectorWrap<ValueT,dim>(self(*mesh, method), mesh);
        }
        RegisterProviderImpl() {
            this->provider_class.def("__call__", &__call__, "Get value from the provider", (py::arg("mesh"), py::arg("interpolation")=DEFAULT_INTERPOLATION));
        }
    };

} // namespace detail


template <typename ReceiverT>
struct RegisterReceiver : public detail::RegisterReceiverImpl<ReceiverT, ReceiverT::PropertyTag::propertyType> {};

template <typename ProviderT>
struct RegisterProvider : public detail::RegisterProviderImpl<ProviderT, ProviderT::PropertyTag::propertyType>  {};


}} // namespace plask::python

#endif // PLASK__PYTHON_PROVIDER_H

