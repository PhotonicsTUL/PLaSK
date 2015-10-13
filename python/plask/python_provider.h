#ifndef PLASK__PYTHON_PROVIDER_H
#define PLASK__PYTHON_PROVIDER_H

#include <type_traits>  // std::is_same

#include "python_globals.h"

#include <boost/python/args.hpp>

#include <plask/utils/stl.h>
#include <plask/provider/providerfor.h>
#include <plask/mesh/rectangular.h>

namespace plask { namespace python {

    template <typename PropertyT> struct PropertyArgsSingleValue {
        static py::detail::keywords<1> value() {
            return py::arg("self");
        }
    };

    template <typename PropertyT> struct PropertyArgsMultiValue {
        static py::detail::keywords<2> value() {
            return py::arg("self"), py::arg("n");
        }
    };

    template <typename PropertyT> struct PropertyArgsField {
        static py::detail::keywords<3> value() {
            return py::arg("self"), py::arg("mesh"), py::arg("interpolation")=INTERPOLATION_DEFAULT;
        }
    };

    template <typename PropertyT> struct PropertyArgsMultiField {
        static py::detail::keywords<4> value() {
            return py::arg("self"), py::arg("n"), py::arg("mesh"), py::arg("interpolation")=INTERPOLATION_DEFAULT;
        }
    };

}} // namespace plask::python

#include "python_property_desc.h"

namespace plask { namespace python {

extern py::object flow_module;

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

    DataVectorWrap(const DataVectorWrap<T,dim>& src)
        : DataVector<T>(src), mesh(src.mesh), mesh_changed(src.mesh_changed)
    {
        if (mesh) mesh->changedConnectMethod(this, &DataVectorWrap<T,dim>::onMeshChanged);
    }

    ~DataVectorWrap() {
        if (mesh) mesh->changedDisconnectMethod(this, &DataVectorWrap<T,dim>::onMeshChanged);
    }

    void onMeshChanged(const typename MeshD<dim>::Event& event) { mesh_changed = true; }
};

// ---------- Receiver ------------

extern const char* docstring_receiver;
extern const char* docstring_receiver_connect;
extern const char* docstring_receiver_assign;

template <typename ProviderT, PropertyType propertyType, typename ParamsT>
struct PythonProviderFor;

namespace detail {

    template <typename ReceiverT, PropertyType propertyType, typename VariadicTemplateTypesHolder> struct RegisterReceiverImpl;

    template <typename ReceiverT>
    struct RegisterReceiverBase
    {
        typedef typename ReceiverT::PropertyTag PropertyT;
        typedef ProviderFor<PropertyT, typename ReceiverT::SpaceType> ProviderT;
        typedef RegisterReceiverImpl<ReceiverT, PropertyT::propertyType, typename PropertyT::ExtraParams> RegisterT;

        const std::string property_name;
        py::class_<ReceiverT, boost::noncopyable> receiver_class;

        static void connect(ReceiverT& receiver, py::object oprovider) {
            ProviderT* provider = py::extract<ProviderT*>(oprovider);
            receiver.setProvider(provider);
            // Make sure that provider stays alive as long as it is connected
            PyObject* obj = oprovider.ptr();
            py::incref(obj);
            receiver.providerValueChanged.connect_extended(
                [obj](const boost::signals2::connection& conn, ReceiverBase&, ReceiverBase::ChangeReason reason) -> void {
                    if (reason == ReceiverT::ChangeReason::REASON_PROVIDER || reason == ReceiverT::ChangeReason::REASON_DELETE) {
                        conn.disconnect();
                        py::decref(obj);
                    }
                }
            );
        }

        static void disconnect(ReceiverT& receiver) { receiver.setProvider(nullptr); }

        static py::object __get__(const py::object& self, const py::object&, const py::object&) { return self; }

        static void __set__(ReceiverT& self, const py::object&, const py::object& value) {
            RegisterT::assign(self, value);
        }

        RegisterReceiverBase(const std::string& suffix="", const std::string& space="") :
            property_name(type_name<PropertyT>()),
            receiver_class((property_name + "Receiver" + suffix).c_str(),
                format(docstring_receiver, property_name, suffix, ReceiverT::ProviderType::PropertyTag::NAME,
                (space!="")? " in "+space+" geometry" : "", ReceiverT::ProviderType::PropertyTag::UNIT).c_str()
            ) {
            receiver_class.def("connect", &connect,
                               format(docstring_receiver_connect, property_name).c_str(),
                               py::arg("provider"));
            receiver_class.def("disconnect", &disconnect, "Disconnect any provider from the receiver.");
            receiver_class.def("assign", &ReceiverT::template setConstValue<const typename ReceiverT::ValueType&>,
                               format(docstring_receiver_assign, property_name).c_str(),
                               py::arg("value"));
            receiver_class.add_property("changed", (bool (ReceiverT::*)() const)&ReceiverT::changed,
                                        "Indicates whether the receiver value has changed since the last retrieval.");
            receiver_class.def("__get__", &__get__);
            receiver_class.def("__set__", &__set__);
        }
    };

    template <typename ReceiverT>
    static bool assignProvider(ReceiverT& receiver, const py::object& obj) {
        try {
            RegisterReceiverBase<ReceiverT>::connect(receiver, obj);
            return true;
        } catch (py::error_already_set) { PyErr_Clear(); }
        return false;
    }

    template <typename ReceiverT>
    static bool assignValue(ReceiverT& receiver, const py::object& obj) {
        typedef typename ReceiverT::ValueType ValueT;
        try {
            ValueT value = py::extract<ValueT>(obj);
            receiver = value;
            return true;
        } catch (py::error_already_set) { PyErr_Clear(); }
        return false;
    }

    template <typename ReceiverT>
    static bool assignMultipleValues(ReceiverT& receiver, const py::object& obj) {
        if (!PySequence_Check(obj.ptr())) return false;
        typedef typename ReceiverT::ValueType ValueT;
        try {
            py::stl_input_iterator<ValueT> begin(obj), end;
            receiver.setValues(begin, end);
            return true;
        } catch (py::error_already_set) { PyErr_Clear(); }
        return false;
    }

    template <typename ReceiverT, typename... ExtraParams>
    struct RegisterReceiverImpl<ReceiverT, SINGLE_VALUE_PROPERTY, VariadicTemplateTypesHolder<ExtraParams...> > :
    public RegisterReceiverBase<ReceiverT>
    {
        typedef typename ReceiverT::PropertyTag PropertyT;
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
            this->receiver_class.def("__call__", &__call__, PropertyArgsSingleValue<PropertyT>::value(), "Get value from the connected provider");
        }
    };

    template <typename ReceiverT, typename... ExtraParams>
    struct RegisterReceiverImpl<ReceiverT, MULTI_VALUE_PROPERTY, VariadicTemplateTypesHolder<ExtraParams...> > :
    public RegisterReceiverBase<ReceiverT>
    {
        typedef typename ReceiverT::PropertyTag PropertyT;
        typedef typename ReceiverT::PropertyTag::ValueType ValueT;

        static void assign(ReceiverT& self, const py::object& obj) {
            if (obj == py::object()) { self.setProvider(nullptr); return; }
            if (assignProvider(self, obj)) return;
            if (assignMultipleValues(self, obj)) return;
            if (assignValue(self, obj)) return;
            throw TypeError("You can only assign %1% provider, or sequence of values of type '%2%'",
                            type_name<typename ReceiverT::PropertyTag>(),
                            std::string(py::extract<std::string>(py::object(dtype<ValueT>()).attr("__name__"))));
        }

        static ValueT __call__n(ReceiverT& self, size_t n, const ExtraParams&... params) { return self(n, params...); }

        static ValueT __call__0(ReceiverT& self, const ExtraParams&... params) { return self(0, params...); }

        RegisterReceiverImpl() {
            this->receiver_class.def("__call__", &__call__0, PropertyArgsSingleValue<PropertyT>::value(), "Get value from the connected provider");
            this->receiver_class.def("__call__", &__call__n, PropertyArgsMultiValue<PropertyT>::value(), "Get value from the connected provider");
            this->receiver_class.def("__len__", (size_t (ReceiverT::*)()const)&ReceiverT::size, "Get number of values from connected provider");
        }
    };

    template <int DIMS, typename ReceiverT, typename... ExtraParams> struct ReceiverSetValueForMeshes;

    template <typename ReceiverT, typename... ExtraParams>
    struct RegisterReceiverImpl<ReceiverT, FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraParams...> > :
    public RegisterReceiverBase<ReceiverT>
    {
        typedef typename ReceiverT::PropertyTag PropertyT;
        typedef typename ReceiverT::ValueType ValueT;
        static const int DIMS = ReceiverT::SpaceType::DIM;
        typedef DataVectorWrap<const ValueT, DIMS> DataT;

        static void assign(ReceiverT& self, const py::object& obj) {
            if (obj == py::object()) { self.setProvider(nullptr); return; }
            if (assignProvider(self, obj)) return;
            if (assignValue(self, obj)) return;
            auto data = make_shared<PythonProviderFor<typename ReceiverT::ProviderType, ReceiverT::PropertyTag::propertyType, VariadicTemplateTypesHolder<ExtraParams...>>>(obj);
            if (assignProvider(self, py::object(data))) return;
            throw TypeError("You can only assign %1% provider, data, or constant of type '%2%'",
                            type_name<typename ReceiverT::PropertyTag>(),
                            std::string(py::extract<std::string>(py::object(dtype<ValueT>()).attr("__name__"))));
        }

        static DataT __call__(ReceiverT& self, const shared_ptr<MeshD<DIMS>>& mesh, const ExtraParams&... params, InterpolationMethod method) {
            return DataT(self(mesh, params..., method), mesh);
        }

        RegisterReceiverImpl(): RegisterReceiverBase<ReceiverT>(spaceSuffix<typename ReceiverT::SpaceType>(), spaceName<typename ReceiverT::SpaceType>()) {
            this->receiver_class.def("__call__", &__call__, PropertyArgsField<PropertyT>::value(), "Get value from the connected provider");
        }

      private:

        template <typename MeshT>
        static inline bool setValueForMesh(ReceiverT& self, const shared_ptr<Mesh>& msh, const DataT& data) {
            shared_ptr<MeshT> mesh = dynamic_pointer_cast<MeshT>(msh);
            if (mesh) { self.setValue(data, mesh); return true; }
            return false;
        }
    };

    template <typename ReceiverT, typename... ExtraParams>
    struct RegisterReceiverImpl<ReceiverT, MULTI_FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraParams...> > :
    public RegisterReceiverBase<ReceiverT>
    {
        typedef typename ReceiverT::PropertyTag PropertyT;
        typedef typename ReceiverT::ValueType ValueT;
        static const int DIMS = ReceiverT::SpaceType::DIM;
        typedef DataVectorWrap<const ValueT, DIMS> DataT;

        static void assign(ReceiverT& self, const py::object& obj) {
            if (obj == py::object()) { self.setProvider(nullptr); return; }
            if (assignProvider(self, obj)) return;
            if (assignValue(self, obj)) return;
            auto data = make_shared<PythonProviderFor<typename ReceiverT::ProviderType, ReceiverT::PropertyTag::propertyType, VariadicTemplateTypesHolder<ExtraParams...>>>(obj);
            if (assignProvider(self, py::object(data))) return;
            throw TypeError("You can only assign %1% provider, sequence of data, or constant of type '%2%'",
                            type_name<typename ReceiverT::PropertyTag>(),
                            std::string(py::extract<std::string>(py::object(dtype<ValueT>()).attr("__name__"))));
        }

        static DataT __call__n(ReceiverT& self, size_t n, const shared_ptr<MeshD<DIMS>>& mesh, const ExtraParams&... params, InterpolationMethod method) {
            return DataT(self(n, mesh, params..., method), mesh);
        }

        static DataT __call__0(ReceiverT& self, const shared_ptr<MeshD<DIMS>>& mesh, const ExtraParams&... params, InterpolationMethod method) {
            return DataT(self(0, mesh, params..., method), mesh);
        }

        RegisterReceiverImpl(): RegisterReceiverBase<ReceiverT>(spaceSuffix<typename ReceiverT::SpaceType>(), spaceName<typename ReceiverT::SpaceType>()) {
            this->receiver_class.def("__call__", &__call__0, PropertyArgsField<PropertyT>::value(), "Get value from the connected provider");
            this->receiver_class.def("__call__", &__call__n, PropertyArgsMultiField<PropertyT>::value(), "Get value from the connected provider");
            this->receiver_class.def("__len__", (size_t (ReceiverT::*)()const)&ReceiverT::size, "Get number of values from connected provider");
        }

      private:

        template <typename MeshT, typename DataT>
        static inline bool setValueForMesh(ReceiverT& self, const shared_ptr<Mesh>& msh, const DataT& data) {
            shared_ptr<MeshT> mesh = dynamic_pointer_cast<MeshT>(msh);
            if (mesh) { self.setValue(data, mesh); return true; }
            return false;
        }
    };


    template <typename Class, typename ClassT, typename ReceiverT>
    struct ReceiverSetter
    {
        typedef typename ReceiverT::PropertyTag PropertyT;
        typedef detail::RegisterReceiverImpl<ReceiverT, PropertyT::propertyType, typename PropertyT::ExtraParams> RegisterT;

        ReceiverSetter(ReceiverT ClassT::* field) : field(field) {}

        void operator()(Class& self, py::object obj) {
            RegisterT::assign(self.*field, obj);
        }

      private:
        ReceiverT ClassT::* field;
    };
}

// ---------- Provider ------------

template <typename ProviderT, typename... _ExtraParams>
struct PythonProviderFor<ProviderT, SINGLE_VALUE_PROPERTY, VariadicTemplateTypesHolder<_ExtraParams...>>:
public ProviderFor<typename ProviderT::PropertyTag>::Delegate {

    typedef typename ProviderFor<typename ProviderT::PropertyTag>::ProvidedType ProvidedType;

    py::object function;
    OmpLock provider_omp_lock;

    PythonProviderFor(const py::object& function):  ProviderFor<typename ProviderT::PropertyTag>::Delegate(
        [this](_ExtraParams... params) -> ProvidedType {
            OmpLockGuard<OmpLock> lock(this->provider_omp_lock);
            if (PyCallable_Check(this->function.ptr())) 
                return py::extract<ProvidedType>(this->function(params...));
            else
                return py::extract<ProvidedType>(this->function);
        }        
    ), function(function) {}

};

template <typename ProviderT, typename... _ExtraParams>
struct PythonProviderFor<ProviderT, MULTI_VALUE_PROPERTY, VariadicTemplateTypesHolder<_ExtraParams...>>:
public ProviderFor<typename ProviderT::PropertyTag>::Delegate {

    typedef typename ProviderFor<typename ProviderT::PropertyTag>::ProvidedType ProvidedType;

    py::object function;
    OmpLock provider_omp_lock;

    PythonProviderFor(const py::object& function):  ProviderFor<typename ProviderT::PropertyTag>::Delegate(
        [this](size_t n, _ExtraParams... params) -> ProvidedType {
            OmpLockGuard<OmpLock> lock(this->provider_omp_lock);
            if (PyCallable_Check(this->function.ptr()))
                return py::extract<ProvidedType>(this->function(n, params...));
            else
                return py::extract<ProvidedType>(this->function[n]);
        },
        [this]() -> size_t {
            OmpLockGuard<OmpLock> lock(this->provider_omp_lock);
            return py::extract<size_t>(this->function.attr("__len__")());
        }
    ), function(function) {}

};

py::object Data(PyObject* obj, py::object omesh);

template <typename T, int dim>
DataVectorWrap<T,dim> PLASK_PYTHON_API dataInterpolate(
    const DataVectorWrap<T,dim>& src, shared_ptr<MeshD<dim>> dst_mesh, InterpolationMethod method);


template <typename ProviderT, typename... _ExtraParams>
struct PythonProviderFor<ProviderT, FIELD_PROPERTY, VariadicTemplateTypesHolder<_ExtraParams...>>:
public ProviderFor<typename ProviderT::PropertyTag, typename ProviderT::SpaceType>::Delegate {

    typedef typename ProviderFor<typename ProviderT::PropertyTag, typename ProviderT::SpaceType>::ProvidedType ProvidedType;
    typedef DataVectorWrap<const typename ProviderT::ValueType, ProviderT::SpaceType::DIM> ReturnedType;

    py::object function;
    OmpLock provider_omp_lock;

    PythonProviderFor(const py::object& function): ProviderFor<typename ProviderT::PropertyTag, typename ProviderT::SpaceType>::Delegate(
        [this](const shared_ptr<const MeshD<ProviderT::SpaceType::DIM>>& dst_mesh, _ExtraParams... params, InterpolationMethod method) -> ProvidedType
        {
            OmpLockGuard<OmpLock> lock(this->provider_omp_lock);
            if (PyCallable_Check(this->function.ptr())) {
                py::object omesh(const_pointer_cast<MeshD<ProviderT::SpaceType::DIM>>(dst_mesh));
                py::object result = this->function(omesh, params..., method);
                try {
                    return py::extract<ReturnedType>(result)();
                } catch (py::error_already_set) {
                    PyErr_Clear();
                    return py::extract<ReturnedType>(Data(result.ptr(), omesh))();
                }
            } else {
                ReturnedType data = py::extract<ReturnedType>(this->function);
                if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
                return dataInterpolate(data, const_pointer_cast<MeshD<ProviderT::SpaceType::DIM>>(dst_mesh), method);
            }
        }
    ), function(function) {
        if (!PyCallable_Check(function.ptr()) && !py::extract<ReturnedType>(function).check())
            throw TypeError("'data' in custom Python provider must be a callable or a Data object");
    }
};

template <typename ProviderT, typename... _ExtraParams>
struct PythonProviderFor<ProviderT, MULTI_FIELD_PROPERTY, VariadicTemplateTypesHolder<_ExtraParams...>>:
public ProviderFor<typename ProviderT::PropertyTag, typename ProviderT::SpaceType>::Delegate {

    typedef typename ProviderFor<typename ProviderT::PropertyTag, typename ProviderT::SpaceType>::ProvidedType ProvidedType;
    typedef DataVectorWrap<const typename ProviderT::ValueType, ProviderT::SpaceType::DIM> ReturnedType;

    py::object function;
    OmpLock provider_omp_lock;

    PythonProviderFor(const py::object& function): ProviderFor<typename ProviderT::PropertyTag, typename ProviderT::SpaceType>::Delegate (
        [this](size_t n, const shared_ptr<const MeshD<ProviderT::SpaceType::DIM>>& dst_mesh, _ExtraParams... params, InterpolationMethod method) -> ProvidedType
        {
            OmpLockGuard<OmpLock> lock(this->provider_omp_lock);
            if (PyCallable_Check(this->function.ptr())) {
                py::object omesh(const_pointer_cast<MeshD<ProviderT::SpaceType::DIM>>(dst_mesh));
                py::object result = this->function(n, omesh, params..., method);
                try {
                    return py::extract<ReturnedType>(result)();
                } catch (py::error_already_set) {
                    PyErr_Clear();
                    return py::extract<ReturnedType>(Data(result.ptr(), omesh))();
                }
            } else {
                ReturnedType data = py::extract<ReturnedType>(this->function[n]);
                if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
                return dataInterpolate(data, const_pointer_cast<MeshD<ProviderT::SpaceType::DIM>>(dst_mesh), method);
            }
        },
        [this]() -> size_t {
            OmpLockGuard<OmpLock> lock(this->provider_omp_lock);
            return py::extract<size_t>(this->function.attr("__len__")());
        }
    ), function(function) {
        if (PyCallable_Check(function.ptr())) return;
        if (!PySequence_Check(function.ptr())) {
            throw TypeError("'data' in custom Python provider must be a callable or a sequence of Data objects");
        }
        size_t len = py::len(function);
        if (!len) return;
        ReturnedType data0(py::extract<ReturnedType>(this->function[0]));
        for (size_t n = 0; n != len; ++n) {
            py::extract<ReturnedType> data(this->function[n]);
            if (!data.check())
                throw TypeError("'data' in custom Python provider must be a callable or a sequence of Data objects");
            if (ReturnedType(data).mesh != data0.mesh)
                throw ValueError("Mesh in each element of 'data' sequence must be the same");
        }
    }
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

    static py::object __add__(py::object pyself, typename CombinedProviderT::BaseType* provider) {
        CombinedProviderT* self = py::extract<CombinedProviderT*>(pyself);
        self->add(provider);
        return pyself;
    }

    static void __iadd__(py::object pyself, typename CombinedProviderT::BaseType* provider) {
        __add__(pyself, provider);
    }

    static CombinedProviderT* add(typename CombinedProviderT::BaseType* provider1, typename CombinedProviderT::BaseType* provider2) {
        auto self = new CombinedProviderT;
        self->add(provider1);
        self->add(provider2);
        return self;
    }

    RegisterCombinedProvider(const std::string& name)  {
        py::scope scope = flow_module;
        Class pyclass(name.c_str(), (std::string(
            "Combined provider for ") + CombinedProviderT::NAME + ".\n\n"
            "This provider holds a sum of the other providers, so the provided field\n"
            "is the sum of its sources.\n"
        ).c_str());
        pyclass.def("__iadd__", &__iadd__, py::with_custodian_and_ward<1,2>())
               .def("__len__", &CombinedProviderT::size)
               .def("add", &__iadd__, py::arg("provider"),
                    "Add another provider to the combination.\n"
                    "Using this function is equal to calling ``self += provider``.\n\n"
                    "Args:\n"
                    "    provider: Provider to add.\n",
                    py::with_custodian_and_ward_postcall<0,2>()
                   )
               .def("remove", &CombinedProviderT::remove, py::arg("provider"),
                    "Remove provider from the combination.\n\n"
                    "Args:\n"
                    "    provider: Provider to remove.\n"
                   )
               .def("clear", &CombinedProviderT::clear, "Clear all elements of the combined provider.")
               .def("__add__", &__add__, py::with_custodian_and_ward_postcall<0,2>())
        ;

        py::handle<> cls = py::handle<>(py::borrowed(reinterpret_cast<PyObject*>(
            py::converter::registry::lookup(py::type_id<typename CombinedProviderT::BaseType>()).m_class_object
        )));
        if (!cls) throw CriticalException("No registered provider for %1%", py::type_id<typename CombinedProviderT::BaseType>().name());
        py::scope cls_scope = py::object(cls);
        py::def("__add__", &add, py::with_custodian_and_ward_postcall<0,1,
                                 py::with_custodian_and_ward_postcall<0,2,
                                 py::return_value_policy<py::manage_new_object>>>());
    }

};


// ---------- Scaled Provider ------------
template <typename ScaledProviderT>
struct RegisterScaledProvider {

    py::handle<> cls;

    typedef py::class_<ScaledProviderT, py::bases<ProviderFor<typename ScaledProviderT::PropertyTag, typename ScaledProviderT::SpaceType>>, boost::noncopyable> Class;

    static void __imul__(ScaledProviderT* self, typename ScaledProviderT::ScaleType factor) {
        self->scale *= factor;
    }

    static void __idiv__(ScaledProviderT* self, typename ScaledProviderT::ScaleType factor) {
        self->scale /= factor;
    }

    static ScaledProviderT* mul(typename ScaledProviderT::SourceType* source, typename ScaledProviderT::ScaleType scale) {
        auto self = new ScaledProviderT;
        self->set(source);
        self->scale = scale;
        return self;
    }

    static ScaledProviderT* div(typename ScaledProviderT::SourceType* source, typename ScaledProviderT::ScaleType scale) {
        auto self = new ScaledProviderT;
        self->set(source);
        self->scale = 1./scale;
        return self;
    }

    RegisterScaledProvider(const std::string& name)  {
        Class pyclass(name.c_str(), (std::string("Scaled provider for ") + ScaledProviderT::NAME).c_str(),
                      py::init<typename ScaledProviderT::ScaleType>(py::arg("scale")=1.)
        );
        pyclass.def("__imul__", &__imul__)
               .def("__idiv__", &__idiv__)
               .def("__itruediv__", &__idiv__)
               .def_readwrite("scale", &ScaledProviderT::scale)
        ;

        cls = py::handle<>(py::borrowed(reinterpret_cast<PyObject*>(
            py::converter::registry::lookup(py::type_id<typename ScaledProviderT::SourceType>()).m_class_object
        )));

        registerOperator("__mul__", &mul);
        registerOperator("__rmul__", &mul);
        registerOperator("__div__", &div);
        registerOperator("__truediv__", &div);
    }

  private:

    template <typename F>
    void registerOperator(const char* name, F func) {
        py::scope scope;
        boost::optional<py::object> old;
        try { old.reset(scope.attr(name)); }
        catch (py::error_already_set) { PyErr_Clear(); }
        py::def(name, func, py::with_custodian_and_ward_postcall<0,1,
                            py::return_value_policy<py::manage_new_object>>());
        if (cls) py::object(cls).attr(name) = scope.attr(name);
        if (old) scope.attr(name) = *old;
        else py::delattr(scope, name);
    }
};


template <PropertyType propertyType> const char* docstring_provider();

namespace detail {

    template <typename ProviderT>
    struct RegisterProviderBase {
        typedef PythonProviderFor<ProviderT, ProviderT::PropertyTag::propertyType, typename ProviderT::PropertyTag::ExtraParams> PythonProviderType;
        const std::string property_name;
        py::class_<ProviderT, shared_ptr<ProviderT>, boost::noncopyable> provider_class;
        RegisterProviderBase(const std::string& suffix="", const std::string& space="") :
            property_name(type_name<typename ProviderT::PropertyTag>()),
            provider_class((property_name + "Provider" + suffix).c_str(), py::no_init) {
            py::class_<PythonProviderType, shared_ptr<PythonProviderType>, py::bases<ProviderT>, boost::noncopyable>((
                property_name + "Provider" + suffix).c_str(),
                format(docstring_provider<ProviderT::PropertyTag::propertyType>(),
                        property_name, suffix, ProviderT::PropertyTag::NAME,
                        (space!="")? " in "+space+" geometry" : "",
                        docstrig_property_optional_args<typename ProviderT::PropertyTag>(),
                        docstrig_property_optional_args_desc<typename ProviderT::PropertyTag>(),
                        ProviderT::PropertyTag::UNIT
                        ).c_str(),
                py::no_init)
                .def("__init__", py::make_constructor(PythonProviderFor__init__<ProviderT>, py::default_call_policies(), py::args("data")))
                .def("__get__", &RegisterProviderBase<ProviderT>::__get__)
                .def("set_changed", &ProviderT::fireChanged,
                    "Inform all connected receivers that the provided value has changed.\n\n"
                    "The receivers will have its `changed` attribute set to True and solvers will\n"
                    "call the provider again if they need its value (otherwise they might take it\n"
                    "from the cache.\n");
        }
        static shared_ptr<PythonProviderType> __get__(const shared_ptr<PythonProviderType>& self, PyObject* instance, PyObject* owner) {
            PyObject* func = self->function.ptr();
            if (!PyCallable_Check(func) || (PyMethod_Check(func) && PyMethod_Self(func))) return self;
#           if PY_VERSION_HEX >= 0x03000000
                PyObject* bound_method = PyMethod_New(func, instance);
#           else
                PyObject* bound_method = PyMethod_New(func, instance, owner);
#           endif
            return PythonProviderFor__init__<ProviderT>(py::object(py::handle<>(bound_method)));
        }
    };

    template <typename ProviderT, PropertyType propertyType, typename VariadicTemplateTypesHolder> struct RegisterProviderImpl;

    template <typename ProviderT, typename... ExtraParams>
    struct RegisterProviderImpl<ProviderT, SINGLE_VALUE_PROPERTY, VariadicTemplateTypesHolder<ExtraParams...> > :
    public RegisterProviderBase<ProviderT>
    {
        typedef typename ProviderT::PropertyTag PropertyT;
        typedef typename ProviderT::PropertyTag::ValueType ValueT;
        static ValueT __call__(ProviderT& self, const ExtraParams&... params) { return self(params...); }
        RegisterProviderImpl() {
            this->provider_class.def("__call__", &__call__, PropertyArgsSingleValue<PropertyT>::value(), "Get value from the provider.");
        }
    };

    template <typename ProviderT, typename... ExtraParams>
    struct RegisterProviderImpl<ProviderT, MULTI_VALUE_PROPERTY, VariadicTemplateTypesHolder<ExtraParams...> > :
    public RegisterProviderBase<ProviderT>
    {
        typedef typename ProviderT::PropertyTag PropertyT;
        typedef typename ProviderT::PropertyTag::ValueType ValueT;
        static ValueT __call__n(ProviderT& self, int n, const ExtraParams&... params) {
            if (n < 0) n = self.size() + n;
            if (n < 0 || n >= self.size())
                throw NoValue(format("%1% [%2%]", self.name(), n).c_str());
            return self(n, params...);
        }
        static ValueT __call__0(ProviderT& self, const ExtraParams&... params) { return self(0, params...); }
        RegisterProviderImpl() {
            this->provider_class.def("__call__", &__call__0, PropertyArgsSingleValue<PropertyT>::value(), "Get value from the provider.");
            this->provider_class.def("__call__", &__call__n, PropertyArgsMultiValue<PropertyT>::value(), "Get value from the provider.");
            this->provider_class.def("__len__", &ProviderT::size, "Get number of provided values.");
        }
    };

    template <typename ProviderT, typename... ExtraParams>
    struct RegisterProviderImpl<ProviderT, FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraParams...> > :
    public RegisterProviderBase<ProviderT>
    {
        static const int DIMS = ProviderT::SpaceType::DIM;
        typedef typename ProviderT::PropertyTag PropertyT;
        typedef typename ProviderT::ValueType ValueT;

        static DataVectorWrap<const ValueT,DIMS> __call__(ProviderT& self, const shared_ptr<MeshD<DIMS>>& mesh, const ExtraParams&... params, InterpolationMethod method) {
            if (!mesh) throw TypeError("You must provide proper mesh to %1% provider", self.name());
            return DataVectorWrap<const ValueT,DIMS>(self(mesh, params..., method), mesh);
        }
        RegisterProviderImpl(): RegisterProviderBase<ProviderT>(spaceSuffix<typename ProviderT::SpaceType>(), spaceName<typename ProviderT::SpaceType>()) {
            this->provider_class.def("__call__", &__call__, PropertyArgsField<PropertyT>::value(), "Get value from the provider.");
        }
    };

    template <typename ProviderT, typename... ExtraParams>
    struct RegisterProviderImpl<ProviderT, MULTI_FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraParams...> > :
    public RegisterProviderBase<ProviderT>
    {
        static const int DIMS = ProviderT::SpaceType::DIM;
        typedef typename ProviderT::PropertyTag PropertyT;
        typedef typename ProviderT::ValueType ValueT;

        static DataVectorWrap<const ValueT,DIMS> __call__n(ProviderT& self, int n, const shared_ptr<MeshD<DIMS>>& mesh, const ExtraParams&... params, InterpolationMethod method) {
            if (!mesh) throw TypeError("You must provide proper mesh to %1% provider", self.name());
            if (n < 0) n = self.size() + n;
            if (n < 0 || n >= self.size())
                throw NoValue(format("%1% [%2%]", self.name(), n).c_str());
            return DataVectorWrap<const ValueT,DIMS>(self(n, mesh, params..., method), mesh);
        }
        static DataVectorWrap<const ValueT,DIMS> __call__0(ProviderT& self, const shared_ptr<MeshD<DIMS>>& mesh, const ExtraParams&... params, InterpolationMethod method) {
            if (!mesh) throw TypeError("You must provide proper mesh to %1% provider", self.name());
            return DataVectorWrap<const ValueT,DIMS>(self(0, mesh, params..., method), mesh);
        }
        RegisterProviderImpl(): RegisterProviderBase<ProviderT>(spaceSuffix<typename ProviderT::SpaceType>(), spaceName<typename ProviderT::SpaceType>()) {
            this->provider_class.def("__call__", &__call__0, PropertyArgsField<PropertyT>::value(), "Get value from the provider.");
            this->provider_class.def("__call__", &__call__n, PropertyArgsMultiField<PropertyT>::value(), "Get value from the provider.");
            this->provider_class.def("__len__", &ProviderT::size, "Get number of provided values.");
        }
    };

} // namespace detail

template <typename ReceiverT>
inline void registerReceiver() {
    if (py::converter::registry::lookup(py::type_id<ReceiverT>()).m_class_object == nullptr) {
        py::scope scope = flow_module;
        detail::RegisterReceiverImpl<ReceiverT, ReceiverT::PropertyTag::propertyType, typename ReceiverT::PropertyTag::ExtraParams>();
    }
}

template <typename ProviderT>
void registerProvider() {
    if (py::converter::registry::lookup(py::type_id<ProviderT>()).m_class_object == nullptr) {
        py::scope scope = flow_module;
        detail::RegisterProviderImpl<ProviderT, ProviderT::PropertyTag::propertyType, typename ProviderT::PropertyTag::ExtraParams>();
    }
}

}} // namespace plask::python

#endif // PLASK__PYTHON_PROVIDER_H

