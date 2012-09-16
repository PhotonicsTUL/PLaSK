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

// ---------- Step Profile ------------

struct PythonProfile: public Provider {

    struct Place
    {
        /// Object for which we specify the value
        weak_ptr<GeometryObject> object;

        /// Hints specifying pointed object
        PathHints hints;

        /**
         * Create place
         * \param object geometry object of the place
         * \param hints path hints further specifying the place
         */
        Place(GeometryObject& object, const PathHints& hints=PathHints())
            : object(object.shared_from_this()), hints(hints) {}

        /**
         * Create place
         * \param src python tuple holding object and hints
         */
        Place(py::object src);

        /// Comparison operator for std::find
        inline bool operator==(const Place& other) const {
            return !(object < other.object || other.object < object ||
                     hints  < other.hints  || other.hints  < hints);
        }
    };

    /// Object for which coordinates we specify the values
    weak_ptr<const Geometry> root_geometry;

    /// Values for places.
    std::deque<Place> places;
    std::deque<py::object> values;

    /// Default value, provided for places where there is no other value
    py::object custom_default_value;

    /**
     * Create step profile
     * \param root root geometry
     * \param default_value default value
     */
    PythonProfile(const Geometry& geometry, py::object default_value=py::object()):
        root_geometry(dynamic_pointer_cast<const Geometry>(geometry.shared_from_this())), custom_default_value(default_value) {}

    /// Get value for place
    py::object __getitem__(py::object key);

    /// Set value for place
    void __setitem__(py::object key, py::object value);

    /// Delete place
    void __delitem__(py::object key);

    /// Clear all the values
    void clear();

    /// Return number of defined places
    size_t size() const { return places.size(); }

    /// Return list of all places
    py::list keys() const;

    /// Return list of all values
    py::list pyvalues() const;

    /// Return values for specified mesh
    template <typename ValueT, int DIMS>
    DataVector<ValueT> get(const plask::MeshD<DIMS>& dst_mesh, ValueT default_value) const {
        if (custom_default_value != py::object()) default_value = py::extract<ValueT>(custom_default_value);

        auto geometry = dynamic_pointer_cast<const GeometryD<DIMS>>(root_geometry.lock());
        if (!geometry) return DataVector<ValueT>(dst_mesh.size(), default_value);
        auto root = geometry->getChild();
        if (!root) throw DataVector<ValueT>(dst_mesh.size(), default_value);

        std::vector<ValueT> vals; vals.reserve(values.size());
        for (auto val: values) vals.push_back(py::extract<ValueT>(val));

        DataVector<ValueT> result(dst_mesh.size());

        size_t i = 0;
        for (Vec<DIMS, double> point: dst_mesh) {
            bool assigned = false;
            for (auto place = places.begin(); place != places.end(); ++place) {
                auto object = dynamic_pointer_cast<GeometryObjectD<DIMS>>(place->object.lock());
                if (!object) continue;
                auto regions = root->getObjectInThisCoordinates(object, place->hints);
                for (const auto& region: regions) {
                    if (region && region->includes(point)) {
                        result[i] = vals[place-places.begin()];
                        assigned = true;
                        break;
                    }
                }
                if (assigned) break;
            }
            if (!assigned) result[i] = default_value;
            ++i;
        }

        return result;
    }

};


namespace detail {

// ---------- Profile ------------

    template <typename, PropertyType, typename> struct ProfileProvider;

    template <typename ReceiverT, typename... ExtraParams>
    struct ProfileProvider<ReceiverT,ON_MESH_PROPERTY,VariadicTemplateTypesHolder<ExtraParams...>>:
    public ProviderFor<typename ReceiverT::PropertyTag, typename ReceiverT::SpaceType>, public Provider::Listener
    {
        shared_ptr<PythonProfile> profile;
        ProfileProvider(const shared_ptr<PythonProfile>& parent): profile(parent) {
            parent->add(this);
        }
        virtual ~ProfileProvider() { profile->remove(this); }
        virtual DataVector<typename ReceiverT::PropertyTag::ValueType> operator()(const MeshD<ReceiverT::SpaceType::DIMS>& mesh, ExtraParams...) const {
            return profile->get<typename ReceiverT::PropertyTag::ValueType>(mesh, ReceiverT::PropertyTag::getDefaultValue());
        }
        virtual void onChange() { this->fireChanged(); }
    };

    template <typename ReceiverT, typename... ExtraParams>
    struct ProfileProvider<ReceiverT,FIELD_PROPERTY,VariadicTemplateTypesHolder<ExtraParams...>>:
    public ProviderFor<typename ReceiverT::PropertyTag, typename ReceiverT::SpaceType>, public Provider::Listener
    {
        shared_ptr<PythonProfile> profile;
        ProfileProvider(const shared_ptr<PythonProfile>& parent): profile(parent) {
                parent->add(this);
        }
        virtual ~ProfileProvider() { profile->remove(this); }
        virtual DataVector<typename ReceiverT::PropertyTag::ValueType> operator()(const MeshD<ReceiverT::SpaceType::DIMS>& mesh, ExtraParams..., InterpolationMethod) const {
            return profile->get<typename ReceiverT::PropertyTag::ValueType>(mesh, ReceiverT::PropertyTag::getDefaultValue());
        }
        virtual void onChange() { this->fireChanged(); }
    };

    template <typename ReceiverT>
    void connectProfileProvider(ReceiverT& receiver, shared_ptr<PythonProfile> profile) {
        typedef ProfileProvider<ReceiverT, ReceiverT::PropertyTag::propertyType, typename ReceiverT::PropertyTag::ExtraParams> ProviderT;
        receiver.setProvider(new ProviderT(profile), true);
    }

// ---------- Receiver ------------

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
            receiver_class(("ReceiverFor" + property_name + suffix).c_str(), py::no_init) {
            receiver_class.def("__lshift__", &connect, "(DEPRECIATED) Connect provider to receiver");
            receiver_class.def("connect", &connect, "Connect provider to receiver");
            receiver_class.def("disconnect", &disconnect, "Disconnect any provider from receiver");
            receiver_class.def_readonly("changed", &ReceiverT::changed, "Indicates whether the receiver value has changed since last retrieval");
            py::delattr(py::scope(), ("ReceiverFor" + property_name + suffix).c_str());
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
        typedef typename ReceiverT::PropertyTag::ValueType ValueT;
        try {
            ValueT value = py::extract<ValueT>(obj);
            receiver = value;
            return true;
        } catch (py::error_already_set) { PyErr_Clear(); }
        return false;
    }

    template <typename ReceiverT>
    static bool assignProfile(ReceiverT& receiver, const py::object& obj) {
        typedef ProfileProvider<ReceiverT, ReceiverT::PropertyTag::propertyType, typename ReceiverT::PropertyTag::ExtraParams> StepProviderT;
        try {
            shared_ptr<PythonProfile> profile = py::extract<shared_ptr<PythonProfile>>(obj);
            receiver.setProvider(new StepProviderT(profile), true);
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

    template <typename ReceiverT, typename... ExtraParams>
    struct RegisterReceiverImpl<ReceiverT, ON_MESH_PROPERTY, VariadicTemplateTypesHolder<ExtraParams...> > :
    public RegisterReceiverBase<ReceiverT>
    {
        typedef typename ReceiverT::PropertyTag::ValueType ValueT;
        static const int DIMS = ReceiverT::SpaceType::DIMS;

        static void assign(ReceiverT& self, const py::object& obj) {
            if (obj == py::object()) { self.setProvider(nullptr); return; }
            if (assignProvider(self, obj)) return;
            if (assignProfile(self, obj)) return;
            if (assignValue(self, obj)) return;
            throw TypeError("You can only assign %1% provider, profile, or constant of type '%2%'",
                            type_name<typename ReceiverT::PropertyTag>(),
                            std::string(py::extract<std::string>(py::object(dtype<ValueT>()).attr("__name__"))));
        }

        static DataVectorWrap<ValueT,DIMS> __call__(ReceiverT& self, const shared_ptr<MeshD<DIMS>>& mesh, const ExtraParams&... params) {
            return DataVectorWrap<ValueT,DIMS>(self(*mesh, params...), mesh);
        }

        RegisterReceiverImpl(): RegisterReceiverBase<ReceiverT>(spaceSuffix<typename ReceiverT::SpaceType>()) {
            this->receiver_class.def("__call__", &__call__, "Get value from the connected provider");
            this->receiver_class.def("__lshift__", &connectProfileProvider<ReceiverT>, "(DEPRECIATED) Connect profile to receiver");
            this->receiver_class.def("connect", &connectProfileProvider<ReceiverT>, "Connect profile to receiver");
        }
    };

    template <int DIMS, typename ReceiverT, typename... ExtraParams> struct ReceiverSetValueForMeshes;

    template <typename ReceiverT, typename... ExtraParams>
    struct RegisterReceiverImpl<ReceiverT, FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraParams...> > :
    public RegisterReceiverBase<ReceiverT>
    {
        typedef typename ReceiverT::PropertyTag::ValueType ValueT;
        static const int DIMS = ReceiverT::SpaceType::DIMS;
        typedef DataVectorWrap<ValueT, DIMS> DataT;

        static void assign(ReceiverT& self, const py::object& obj) {
            if (obj == py::object()) { self.setProvider(nullptr); return; }
            if (assignProvider(self, obj)) return;
            if (assignProfile(self, obj)) return;
            if (assignValue(self, obj)) return;
            try {
                DataT data = py::extract<DataT>(obj);
                ReceiverSetValueForMeshes<DIMS,ReceiverT,ExtraParams...>::call(self, data);
            } catch (py::error_already_set) {
                throw TypeError("You can only assign %1% provider, profile, data, or constant of type '%2%'",
                                type_name<typename ReceiverT::PropertyTag>(),
                                std::string(py::extract<std::string>(py::object(dtype<ValueT>()).attr("__name__"))));
            }
        }

        static DataT __call__(ReceiverT& self, const shared_ptr<MeshD<DIMS>>& mesh, const ExtraParams&... params, InterpolationMethod method) {
            return DataT(self(*mesh, params..., method), mesh);
        }

        RegisterReceiverImpl(): RegisterReceiverBase<ReceiverT>(spaceSuffix<typename ReceiverT::SpaceType>()) {
            this->receiver_class.def("__call__", &__call__, "Get value from the connected provider", py::arg("interpolation")=DEFAULT_INTERPOLATION);
            this->receiver_class.def("__lshift__", &connectProfileProvider<ReceiverT>, "(DEPRECIATED) Connect step profile to receiver");
            this->receiver_class.def("connect", &connectProfileProvider<ReceiverT>, "Connect profile to receiver");
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

// ---------- Provider ------------

    template <typename ProviderT>
    struct RegisterProviderBase
    {
        const std::string property_name;
        typedef ProviderFor<typename ProviderT::PropertyTag, typename ProviderT::SpaceType> ProviderBase;
        py::class_<ProviderBase, boost::noncopyable> provider_base_class;
        py::class_<ProviderT, py::bases<ProviderBase>, boost::noncopyable> provider_class;
        RegisterProviderBase(const std::string& suffix="") :
            property_name (type_name<typename ProviderT::PropertyTag>()),
            provider_base_class(("ProviderFor" + property_name + suffix + "Base").c_str(), py::no_init),
            provider_class(("ProviderFor" + property_name + suffix).c_str(), py::no_init) {
            py::delattr(py::scope(), ("ProviderFor" + property_name+ suffix + "Base").c_str());
            py::delattr(py::scope(), ("ProviderFor" + property_name + suffix).c_str());
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
    struct RegisterProviderImpl<ProviderT, ON_MESH_PROPERTY, VariadicTemplateTypesHolder<ExtraParams...> > :
    public RegisterProviderBase<ProviderT>
    {
        typedef typename ProviderT::PropertyTag::ValueType ValueT;
        static const int DIMS = ProviderT::SpaceType::DIMS;
        static DataVectorWrap<ValueT,DIMS> __call__(ProviderT& self, const shared_ptr<MeshD<DIMS>>& mesh, const ExtraParams&... params) {
            return DataVectorWrap<ValueT,DIMS>(self(*mesh, params...), mesh);
        }
        RegisterProviderImpl(): RegisterProviderBase<ProviderT>(spaceSuffix<typename ProviderT::SpaceType>()) {
            this->provider_class.def("__call__", &__call__, "Get value from the provider");
        }
    };

    template <typename ProviderT, typename... ExtraParams>
    struct RegisterProviderImpl<ProviderT, FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraParams...> > :
    public RegisterProviderBase<ProviderT>
    {
        typedef typename ProviderT::PropertyTag::ValueType ValueT;
        static const int DIMS = ProviderT::SpaceType::DIMS;
        static DataVectorWrap<ValueT,DIMS> __call__(ProviderT& self, const shared_ptr<MeshD<DIMS>>& mesh, const ExtraParams&... params, InterpolationMethod method) {
            return DataVectorWrap<ValueT,DIMS>(self(*mesh, params..., method), mesh);
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

            if (RegisterT::template setValueForMesh< RectilinearMesh3D >(self, data)) return;
            if (RegisterT::template setValueForMesh< RegularMesh3D >(self, data)) return;

            throw TypeError("Data on wrong mesh type for this operation");
        }
    };

} // namespace detail


template <typename ReceiverT>
struct RegisterReceiver :
public detail::RegisterReceiverImpl<ReceiverT, ReceiverT::PropertyTag::propertyType, typename ReceiverT::PropertyTag::ExtraParams> {};

template <typename ProviderT>
struct RegisterProvider :
public detail::RegisterProviderImpl<ProviderT, ProviderT::PropertyTag::propertyType, typename ProviderT::PropertyTag::ExtraParams>  {};


}} // namespace plask::python

#endif // PLASK__PYTHON_PROVIDER_H

