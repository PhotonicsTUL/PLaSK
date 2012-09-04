#ifndef PLASK__PYTHON_PROVIDER_H
#define PLASK__PYTHON_PROVIDER_H

#include <type_traits>  // std::is_same

#include "python_globals.h"
#include <plask/provider/providerfor.h>
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

    DataVectorWrap(DataVector<T>&& src, const shared_ptr<MeshD<dim>>& mesh) : DataVector<T>(std::forward<DataVector<T>>(src)), mesh(mesh), mesh_changed(false) {
        mesh->changedConnectMethod(this, &DataVectorWrap<T,dim>::onMeshChanged);
    }

    DataVectorWrap(const DataVector<T>& src) : DataVector<T>(src) {}

    DataVectorWrap(DataVector<T>&& src) : DataVector<T>(std::forward<DataVector<T>>(src)) {}

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
    struct RegisterReceiverImpl<ReceiverT, ON_MESH_PROPERTY> : public RegisterReceiverBase<ReceiverT>
    {
        typedef typename ReceiverT::PropertyTag::ValueType ValueT;
        static const int DIMS = ReceiverT::SpaceType::DIMS;
        static DataVectorWrap<ValueT,DIMS> __call__(ReceiverT& self, const shared_ptr<MeshD<DIMS>>& mesh) {
            return DataVectorWrap<ValueT,DIMS>(self(*mesh), mesh);
        }
        static void setValue(ReceiverT& self, const py::object& obj) { throw TypeError("Operation not allowed for non-interpolated field receiver"); }
        RegisterReceiverImpl() {
            this->receiver_class.def("__call__", &__call__, "Get value from the connected provider", (py::arg("mesh")));
        }
    };

    template <int DIMS, typename ReceiverT> struct ReceiverSetValueForMeshes {};

    template <typename ReceiverT>
    struct RegisterReceiverImpl<ReceiverT, FIELD_PROPERTY> : public RegisterReceiverBase<ReceiverT> {
        typedef typename ReceiverT::PropertyTag::ValueType ValueT;
        static const int DIMS = ReceiverT::SpaceType::DIMS;
        typedef DataVectorWrap<typename ReceiverT::PropertyTag::ValueType, DIMS> DataT;
        static void setValue(ReceiverT& self, const py::object& obj) {
            DataT data = py::extract<DataT>(obj);
            ReceiverSetValueForMeshes<DIMS, ReceiverT>::call(self, data);
        }
        static DataT __call__(ReceiverT& self, const shared_ptr<MeshD<DIMS>>& mesh, InterpolationMethod method) {
            return DataT(self(*mesh, method), mesh);
        }
        RegisterReceiverImpl() {
            this->receiver_class.def("__call__", &__call__, "Get value from the connected provider", (py::arg("mesh"), py::arg("interpolation")=DEFAULT_INTERPOLATION));
            this->receiver_class.def("setValue", &setValue, "Set previously obtained value", (py::arg("data")));
        }
      private:
        template <typename MeshT>
        static inline bool setValueForMesh(ReceiverT& self, const DataT& data) {
            shared_ptr<MeshT> mesh = dynamic_pointer_cast<MeshT>(data.mesh);
            if (mesh) { self.setValue(data, mesh); return true; }
            return false;
        }
        friend struct ReceiverSetValueForMeshes<DIMS, ReceiverT>;
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
    struct RegisterProviderImpl<ProviderT, ON_MESH_PROPERTY> : public RegisterProviderBase<ProviderT>
    {
        typedef typename ProviderT::PropertyTag::ValueType ValueT;
        static const int DIMS = ProviderT::SpaceType::DIMS;
        static DataVectorWrap<ValueT,DIMS> __call__(ProviderT& self, const shared_ptr<MeshD<DIMS>>& mesh) {
            return DataVectorWrap<ValueT,DIMS>(self(*mesh), mesh);
        }
        RegisterProviderImpl() {
            this->provider_class.def("__call__", &__call__, "Get value from the provider", (py::arg("mesh")));
        }
    };

    template <typename ProviderT>
    struct RegisterProviderImpl<ProviderT, FIELD_PROPERTY> : public RegisterProviderBase<ProviderT>
    {
        typedef typename ProviderT::PropertyTag::ValueType ValueT;
        static const int DIMS = ProviderT::SpaceType::DIMS;
        static DataVectorWrap<ValueT,DIMS> __call__(ProviderT& self, const shared_ptr<MeshD<DIMS>>& mesh, InterpolationMethod method) {
            return DataVectorWrap<ValueT,DIMS>(self(*mesh, method), mesh);
        }
        RegisterProviderImpl() {
            this->provider_class.def("__call__", &__call__, "Get value from the provider", (py::arg("mesh"), py::arg("interpolation")=DEFAULT_INTERPOLATION));
        }
    };

    // Here add new mesh types that should be able to be provided in DataVector to receivers:

    // 2D meshes:
    template <typename ReceiverT> struct ReceiverSetValueForMeshes<2, ReceiverT> {
        typedef RegisterReceiverImpl<ReceiverT, FIELD_PROPERTY> RegisterT;
        static void call(ReceiverT& self, const typename RegisterT::DataT& data) {

            if (RegisterT::template setValueForMesh< RectilinearMesh2D >(self, data)) return;
            if (RegisterT::template setValueForMesh< RegularMesh2D >(self, data)) return;

            throw TypeError("Data on wrong mesh type for this operation");
        }
    };

    // 3D meshes:
    template <typename ReceiverT> struct ReceiverSetValueForMeshes<3, ReceiverT> {
        typedef RegisterReceiverImpl<ReceiverT, FIELD_PROPERTY> RegisterT;
        static void call(ReceiverT& self, const typename RegisterT::DataT& data) {

            if (RegisterT::template setValueForMesh< RectilinearMesh3D >(self, data)) return;
            if (RegisterT::template setValueForMesh< RegularMesh3D >(self, data)) return;

            throw TypeError("Data on wrong mesh type for this operation");
        }
    };

} // namespace detail


template <typename ReceiverT>
struct RegisterReceiver : public detail::RegisterReceiverImpl<ReceiverT, ReceiverT::PropertyTag::propertyType> {};

template <typename ProviderT>
struct RegisterProvider : public detail::RegisterProviderImpl<ProviderT, ProviderT::PropertyTag::propertyType>  {};


}} // namespace plask::python

#endif // PLASK__PYTHON_PROVIDER_H

