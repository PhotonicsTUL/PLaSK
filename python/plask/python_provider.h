#ifndef PLASK__PYTHON_PROVIDER_H
#define PLASK__PYTHON_PROVIDER_H

#include <type_traits>  // std::is_same

#include "python_globals.h"
#include <plask/utils/stl.h>
#include <plask/provider/providerfor.h>
#include <plask/mesh/rectilinear.h>
#include <plask/mesh/regular.h>

namespace plask { namespace python {

template <typename T, int dim>
struct DataVectorWrap : public DataVector<T> {
    shared_ptr<MeshD<dim>> mesh;
    bool mesh_changed;

    DataVectorWrap(const DataVector<T>& src, const shared_ptr<MeshD<dim>>& mesh):
        DataVector<T>(src), mesh(mesh), mesh_changed(false) {
        mesh->changedConnectMethod(this, &DataVectorWrap<T,dim>::onMeshChanged);
    }

    DataVectorWrap(DataVector<T>&& src, const shared_ptr<MeshD<dim>>& mesh):
        DataVector<T>(std::forward<DataVector<T>>(src)), mesh(mesh), mesh_changed(false) {
        mesh->changedConnectMethod(this, &DataVectorWrap<T,dim>::onMeshChanged);
    }

    DataVectorWrap(const DataVector<T>& src) : DataVector<T>(src) {}

    DataVectorWrap(DataVector<T>&& src) : DataVector<T>(std::forward<DataVector<T>>(src)) {}

    DataVectorWrap() = default;
    DataVectorWrap(const DataVectorWrap<T,dim>& src):
    DataVector<T>(src), mesh(src.mesh), mesh_changed(src.mesh_changed) {
        if (mesh) mesh->changedConnectMethod(this, &DataVectorWrap<T,dim>::onMeshChanged);
    }

    ~DataVectorWrap() {
        mesh->changedDisconnectMethod(this, &DataVectorWrap<T,dim>::onMeshChanged);
    };

    void onMeshChanged(const typename MeshD<dim>::Event& event) { mesh_changed = true; }
};

// ---------- Receiver ------------

namespace detail {

    template <typename ReceiverT>
    struct RegisterReceiverBase
    {
        typedef ProviderFor<typename ReceiverT::PropertyTag,typename ReceiverT::SpaceType> ProviderT;

        const std::string property_name;
        py::class_<ReceiverT, boost::noncopyable> receiver_class;

        static void connect(ReceiverT& receiver, ProviderT* provider) {
            receiver.setProvider(provider);
        }

        static void disconnect(ReceiverT& receiver) { receiver.setProvider(nullptr); }

        RegisterReceiverBase(const std::string& suffix="") :
            property_name(type_name<typename ReceiverT::PropertyTag>()),
            receiver_class(("ReceiverFor" + property_name + suffix).c_str()) {
            receiver_class.def("connect", &connect, "Connect provider to the receiver", py::arg("provider"));
            receiver_class.def("disconnect", &disconnect, "Disconnect any provider from receiver");
            receiver_class.def("assign", &ReceiverT::template setConstValue<const typename ReceiverT::PropertyTag::ValueType&>, "Assign constant value to the receiver", py::arg("value"));
            receiver_class.add_property("changed", (bool (ReceiverT::*)() const)&ReceiverT::changed, "Indicates whether the receiver value has changed since last retrieval");
        }
    };

    template <typename ReceiverT>
    static bool assignProvider(ReceiverT& receiver, const py::object& obj) {
        typedef ProviderFor<typename ReceiverT::PropertyTag, typename ReceiverT::SpaceType> ProviderT;
        try {
            ProviderT* provider = py::extract<ProviderT*>(obj);
            receiver.setProvider(provider);
            return true;
        } catch (py::error_already_set) { PyErr_Clear(); }
        return false;
    }

    template <typename ReceiverT>
    static bool assignValue(ReceiverT& receiver, const py::object& obj) {
        typedef typename ReceiverT::PropertyValueType ValueT;
        try {
            ValueT value = py::extract<ValueT>(obj);
            receiver = value;
            return true;
        } catch (py::error_already_set) { PyErr_Clear(); }
        return false;
    }

    template <typename ReceiverT, PropertyType propertyType, typename VariadicTemplateTypesHolder> struct RegisterReceiverImpl;

    template <typename ReceiverT, typename... ExtraParams>
    struct RegisterReceiverImpl<ReceiverT, SINGLE_VALUE_PROPERTY, VariadicTemplateTypesHolder<ExtraParams...> > :
    public RegisterReceiverBase<ReceiverT>
    {
        typedef typename ReceiverT::PropertyTag::ValueType ValueT;

        static void assign(ReceiverT& self, const py::object& obj) {
            if (obj == py::object()) { self.setProvider(nullptr); return; }
            if (assignProvider(self, obj)) return;
            if (assignValue(self, obj)) return;
            throw TypeError("You can only assign %1% provider, or value of type '%2%'",
                            type_name<typename ReceiverT::PropertyTag>(),
                            std::string(py::extract<std::string>(py::object(dtype<ValueT>()).attr("__name__"))));
        }

        static ValueT __call__(ReceiverT& self, const ExtraParams&... params) { return self(params...); }

        RegisterReceiverImpl() {
            this->receiver_class.def("__call__", &__call__, "Get value from the connected provider");
        }
    };

    template <int DIMS, typename ReceiverT, typename... ExtraParams> struct ReceiverSetValueForMeshes;

    template <typename ReceiverT, typename... ExtraParams>
    struct RegisterReceiverImpl<ReceiverT, FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraParams...> > :
    public RegisterReceiverBase<ReceiverT>
    {
        typedef typename ReceiverT::PropertyValueType ValueT;
        static const int DIMS = ReceiverT::SpaceType::DIM;
        typedef DataVectorWrap<const ValueT, DIMS> DataT;

        static void assign(ReceiverT& self, const py::object& obj) {
            if (obj == py::object()) { self.setProvider(nullptr); return; }
            if (assignProvider(self, obj)) return;
            if (assignValue(self, obj)) return;
            try {
                DataT data = py::extract<DataT>(obj);
                ReceiverSetValueForMeshes<DIMS,ReceiverT,ExtraParams...>::call(self, data);
            } catch (py::error_already_set) {
                throw TypeError("You can only assign %1% provider, data, or constant of type '%2%'",
                                type_name<typename ReceiverT::PropertyTag>(),
                                std::string(py::extract<std::string>(py::object(dtype<ValueT>()).attr("__name__"))));
            }
        }

        static DataT __call__(ReceiverT& self, const shared_ptr<MeshD<DIMS>>& mesh, const ExtraParams&... params, InterpolationMethod method) {
            return DataT(self(*mesh, params..., method), mesh);
        }

        RegisterReceiverImpl(): RegisterReceiverBase<ReceiverT>(spaceSuffix<typename ReceiverT::SpaceType>()) {
            this->receiver_class.def("__call__", &__call__, "Get value from the connected provider", py::arg("interpolation")=DEFAULT_INTERPOLATION);
        }

      private:

        template <typename MeshT>
        static inline bool setValueForMesh(ReceiverT& self, const DataT& data) {
            shared_ptr<MeshT> mesh = dynamic_pointer_cast<MeshT>(data.mesh);
            if (mesh) { self.setValue(data, mesh); return true; }
            return false;
        }
        friend struct ReceiverSetValueForMeshes<DIMS, ReceiverT, ExtraParams...>;
    };


    template <typename Class, typename ReceiverT>
    struct ReceiverSetter
    {
        typedef typename ReceiverT::PropertyTag PropertyT;
        typedef detail::RegisterReceiverImpl<ReceiverT, PropertyT::propertyType, typename PropertyT::ExtraParams> RegisterT;

        ReceiverSetter(ReceiverT Class::* field) : field(field) {}

        void operator()(Class& self, py::object obj) {
            RegisterT::assign(self.*field, obj);
        }

      private:
        ReceiverT Class::* field;
    };

}

// ---------- Provider ------------

template <typename ProviderT, PropertyType propertyType, typename ParamsT>
struct PythonProviderFor;

template <typename ProviderT, typename... _ExtraParams>
struct PythonProviderFor<ProviderT, SINGLE_VALUE_PROPERTY, VariadicTemplateTypesHolder<_ExtraParams...>>:
public ProviderFor<typename ProviderT::PropertyTag>::Delegate {

    typedef typename ProviderFor<typename ProviderT::PropertyTag>::ProvidedType ProvidedType;

    PythonProviderFor(const py::object& function):  ProviderFor<typename ProviderT::PropertyTag>::Delegate(
        [function](_ExtraParams... params) -> ProvidedType {
            return py::extract<ProvidedType>(function(params...));
        }
    ) {}

};


template <typename ProviderT, typename... _ExtraParams>
struct PythonProviderFor<ProviderT, FIELD_PROPERTY, VariadicTemplateTypesHolder<_ExtraParams...>>:
public ProviderFor<typename ProviderT::PropertyTag, typename ProviderT::SpaceType>::Delegate {

    typedef typename ProviderFor<typename ProviderT::PropertyTag, typename ProviderT::SpaceType>::ProvidedType ProvidedType;

    PythonProviderFor(const py::object& function): ProviderFor<typename ProviderT::PropertyTag, typename ProviderT::SpaceType>::Delegate(
        [function](const MeshD<ProviderT::SpaceType::DIM>& dst_mesh, _ExtraParams... params, InterpolationMethod method) -> ProvidedType
        {
            typedef DataVectorWrap<const typename ProviderT::PropertyValueType, ProviderT::SpaceType::DIM> ReturnedType;
            ReturnedType result = py::extract<ReturnedType>(function(boost::ref(dst_mesh), params..., method));
            return ProvidedType(result);
        }
    ) {}
};

template <typename ProviderT>
shared_ptr<PythonProviderFor<ProviderT, ProviderT::PropertyTag::propertyType, typename ProviderT::PropertyTag::ExtraParams>>
PythonProviderFor__init__(const py::object& function) {
    return make_shared<PythonProviderFor<ProviderT, ProviderT::PropertyTag::propertyType, typename ProviderT::PropertyTag::ExtraParams>>
        (function);
}


// ---------- Combined Provider ------------
template <typename CombinedProviderT>
struct RegisterCombinedProvider {

    typedef py::class_<CombinedProviderT, py::bases<ProviderFor<typename CombinedProviderT::PropertyTag, typename CombinedProviderT::SpaceType>>, boost::noncopyable> Class;

    static py::object __add__(py::object pyself, typename CombinedProviderT::BaseProviderClass* provider) {
        CombinedProviderT* self = py::extract<CombinedProviderT*>(pyself);
        self->add(provider);
        return pyself;
    }

    static void __iadd__(py::object pyself, typename CombinedProviderT::BaseProviderClass* provider) {
        __add__(pyself, provider);
    }

    static CombinedProviderT* add(typename CombinedProviderT::BaseProviderClass* provider1, typename CombinedProviderT::BaseProviderClass* provider2) {
        auto self = new CombinedProviderT;
        self->add(provider1);
        self->add(provider2);
        return self;
    }

    RegisterCombinedProvider(const std::string& name)  {
        Class pyclass(name.c_str(), (std::string("Combined provider for ") + CombinedProviderT::NAME).c_str());
        pyclass.def("__iadd__", &__iadd__, py::with_custodian_and_ward<1,2>())
               .def("__len__", &CombinedProviderT::size)
               .def("add", &__iadd__, "Add another provider to the combination", py::with_custodian_and_ward_postcall<0,2>())
               .def("remove", &CombinedProviderT::remove, "Remove provider from the combination")
               .def("clear", &CombinedProviderT::clear, "Clear all elements of combined provider")
               .def("__add__", &__add__, py::with_custodian_and_ward_postcall<0,2>())
        ;

        py::scope scope;
        boost::optional<py::object> oldadd;
        try { oldadd.reset(scope.attr("__add__")); }
        catch (py::error_already_set) { PyErr_Clear(); }
        py::def("__add__", &add, py::with_custodian_and_ward_postcall<0,1,
                              py::with_custodian_and_ward_postcall<0,2,
                              py::return_value_policy<py::manage_new_object>>>());
        py::handle<> cls = py::handle<>(py::borrowed(reinterpret_cast<PyObject*>(
            py::converter::registry::lookup(py::type_id<typename CombinedProviderT::BaseProviderClass>()).m_class_object
        )));
        if (cls) py::object(cls).attr("__add__") = scope.attr("__add__");
        if (oldadd)
            scope.attr("__add__") = *oldadd;
        else
            py::delattr(scope, "__add__");
    }

};


namespace detail {

    template <typename ProviderT>
    struct RegisterProviderBase {
        const std::string property_name;
        py::class_<ProviderT, shared_ptr<ProviderT>, boost::noncopyable> provider_class;
        RegisterProviderBase(const std::string& suffix="") :
            property_name (type_name<typename ProviderT::PropertyTag>()),
            provider_class(("ProviderFor" + property_name + suffix).c_str(), py::no_init) {
            py::class_<PythonProviderFor<ProviderT, ProviderT::PropertyTag::propertyType, typename ProviderT::PropertyTag::ExtraParams>,
                       py::bases<ProviderT>, boost::noncopyable>(("ProviderFor" + property_name + suffix).c_str(),
                       ("Provider class for " + property_name + " in Geometry" + suffix).c_str(), // TODO documentation
                       py::no_init)
                       .def("__init__", py::make_constructor(PythonProviderFor__init__<ProviderT>));
        }
    };

    template <typename ProviderT, PropertyType propertyType, typename VariadicTemplateTypesHolder> struct RegisterProviderImpl;

    template <typename ProviderT, typename... ExtraParams>
    struct RegisterProviderImpl<ProviderT, SINGLE_VALUE_PROPERTY, VariadicTemplateTypesHolder<ExtraParams...> > :
    public RegisterProviderBase<ProviderT>
    {
        typedef typename ProviderT::PropertyTag::ValueType ValueT;
        static ValueT __call__(ProviderT& self, const ExtraParams&... params) { return self(params...); }
        RegisterProviderImpl() {
            this->provider_class.def("__call__", &__call__, "Get value from the provider");
        }
    };

    template <typename ProviderT, typename... ExtraParams>
    struct RegisterProviderImpl<ProviderT, FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraParams...> > :
    public RegisterProviderBase<ProviderT>
    {
        static const int DIMS = ProviderT::SpaceType::DIM;
        typedef typename ProviderT::PropertyValueType ValueT;

        static DataVectorWrap<const ValueT,DIMS> __call__(ProviderT& self, const shared_ptr<MeshD<DIMS>>& mesh, const ExtraParams&... params, InterpolationMethod method) {
            return DataVectorWrap<const ValueT,DIMS>(self(*mesh, params..., method), mesh);
        }
        RegisterProviderImpl(): RegisterProviderBase<ProviderT>(spaceSuffix<typename ProviderT::SpaceType>()) {
            this->provider_class.def("__call__", &__call__, "Get value from the provider", py::arg("interpolation")=DEFAULT_INTERPOLATION);
        }
    };

    // Here add new mesh types that should be able to be provided in DataVector to receivers:

    // 2D meshes:
    template <typename ReceiverT, typename... ExtraParams>
    struct ReceiverSetValueForMeshes<2, ReceiverT, ExtraParams...> {
        typedef RegisterReceiverImpl<ReceiverT, FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraParams...> > RegisterT;
        static void call(ReceiverT& self, const typename RegisterT::DataT& data) {

            if (RegisterT::template setValueForMesh< RectilinearMesh2D >(self, data)) return;
            if (RegisterT::template setValueForMesh< RegularMesh2D >(self, data)) return;

            throw TypeError("Data on wrong mesh type for this operation");
        }
    };

    // 3D meshes:
    template <typename ReceiverT, typename... ExtraParams>
    struct ReceiverSetValueForMeshes<3, ReceiverT, ExtraParams...> {
        typedef RegisterReceiverImpl<ReceiverT, FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraParams...> > RegisterT;
        static void call(ReceiverT& self, const typename RegisterT::DataT& data) {

            if (RegisterT::template setValueForMesh<RectilinearMesh3D>(self, data)) return;
            if (RegisterT::template setValueForMesh<RegularMesh3D>(self, data)) return;

            throw TypeError("Data on wrong mesh type for this operation");
        }
    };

} // namespace detail

template <typename ReceiverT>
inline void registerReceiver() {
    if (py::converter::registry::lookup(py::type_id<ReceiverT>()).m_class_object == nullptr) {
        detail::RegisterReceiverImpl<ReceiverT, ReceiverT::PropertyTag::propertyType, typename ReceiverT::PropertyTag::ExtraParams>();
    }
}

template <typename ProviderT>
void registerProvider() {
    if (py::converter::registry::lookup(py::type_id<ProviderT>()).m_class_object == nullptr) {
        detail::RegisterProviderImpl<ProviderT, ProviderT::PropertyTag::propertyType, typename ProviderT::PropertyTag::ExtraParams>();
    }
}

}} // namespace plask::python

#endif // PLASK__PYTHON_PROVIDER_H

