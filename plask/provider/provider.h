#ifndef PLASK__PROVIDER_H
#define PLASK__PROVIDER_H

/** @file
This file includes base classes and templates which allow to generate providers and receivers.
@see @ref providers


@page providers Providers and receivers

@section providers_about About provider-receiver mechanism
This page describe providers and receivers mechanism, which allow for data exchange between modules.

Provider is an object which has type derived from plask::Provider and provide some value(s)
(has operator() which return provided value(s)).
It also has set of listeners which are inform about changes of provided data.

Receiver is an object of class derived from plask::Provider::Listener, which is typically connected
with provider and allow for reading value(s) provided by it
(has operator() which return provided value(s)).

Each type of provider has corresponding type of receiver (see plask::Receiver),
and only provider and receiver witch corresponding types can be connected.

@section providers_in_modules Using providers and receivers in modules
Each module should have one provider class field for each physical property which it want to make
available for other modules and reports and it also should have one receiver field for each physical
property which value it wants to know (needs for calculations).
Most providers are classes obtain by using plask::ProviderFor template.

See @ref modules_writing for more details and examples.

An example of using providers and receivers in modules can be found in description of plask::Temperature.

@section providers_writing Writing new providers and receivers types

@subsection providers_writing_easy Easy (half-automatic) way
The easiest way to create new provider and corresponding receiver types is to write physical property
tag class and use it to specialize plask::ProviderFor and plask::ReceiverFor templates.

Physical property tag class is an class which only has static fields and typedefs which describe
physical property. It can be easy obtain by subclass instantiation of one of templates:
- plask::Property — allows to obtain all possible physical properties tags classes, but require many parameters (not recommended);
- plask::SingleValueProperty — allows to obtain tags for properties described by one value (typically one scalar), require only one parameter - type of provided value;
- plask::FieldProperty — allows to obtain tags for properties described by values in points described by mesh (doesn't use interpolation), require only one parameter - type of provided value;
- plask::InterpolatedFieldProperty — allows to obtain tags for properties described by values in points described by mesh and use interpolation, require only one parameter - type of provided value;
- plask::ScalarFieldProperty — equals to plask::InterpolatedFieldProperty\<double\>, doesn't require any parameters.

Both templates plask::ProviderFor and plask::ReceiverFor may take two parameters:
- first is physical property tag and it's required,
- second is type of space (see ) and it's required (and allowed) only for fields properties.

plask::ProviderFor class cannot be used directly, but one must declare it using some specialized class within the plask::ProviderFor namespace.
E.g. \b plask::ProviderFor<MyProperty>::WithValue. The specialized class \b WithValue specifies how the provided values can be obtained.
You can choose from the following options:
- \b WithValue (available only for plask::SingleValueProperty) — the value is stored in the provider itself.
  It can be assigned a value just like any class member field.
- \b WithDefaultValue (available only for plask::SingleValueProperty) — similar to \b WithValue, however it always has some value.
  Use it if there is always some sensible default value for the provided quantity, even before any calculations have been performed.
- \b Delegate (available for all properties) — the module needs to contain the method that computes the provided value (field or scalar) on demand.
  This provider requires the pointer to both the module containing it and the this method as its constructor arguments. See \ref modules_writing_details
  for an example.

Example:
@code
// Physical property tag class for something.
struct MyProperty: public plask::SingleValueProperty<double> {
    static constexpr const char* NAME = "my property"; // use lowercase here
};

// Base type for MyProperty provider.
typedef plask::ProviderFor<MyProperty> MyPropertyProvider;

// Type for MyProperty receiver class.
typedef plask::ReceiverFor<MyProperty> MyPropertyReceiver;

// ...
// Usage example:
MyPropertyProvider::WithValue provider;
MyPropertyReceiver receiver;
receiver << provider;       // connect
provider = 2.0;             // set some value to provider
assert(receiver() == 2.0);  // test the received value
@endcode


@subsection providers_writing_manual Flexible (manual) way

The (described @ref providers_writing_easy "above") method of creating providers and receivers should be sufficient for most cases. However, there is also
a harder but more flexible approach than using plask::ProviderFor and plask::ReceiverFor templates. You can write your own provider class which
inherits from plask::Provider and has operator(), which for some parameters (depending on your choice) returns the provided value.

Receiver class for your provider still may be very easy obtained by plask::Receiver template. This template requires only one parameter: the type of the provider.
You can use it directly or as a base class for your receiver.

Example:
@code
// Provider type which multiple its argument by value
struct ScalerProvider: public plask::Provider {

    double scale;

    ScalerProvider(double scale): scale(scale) {}

    double operator()(double param) const {
        return scale * param;
    }
};

// Receiver corresponding to ScalerProvider
typedef Receiver<ScalerProvider> ScalerReceiver;
// or class ScalerReceiver: public Receiver<ScalerProvider> { ... };

// ...
// Usage example:
ScalerProvider sp(2.0);
ScalerReceiver sr;
sr << sp;               // connect
assert(sr(3.0) == 6.0); // test the received value
@endcode
*/

#include <set>
#include <vector>
#include <functional>   // std::function
#include <type_traits>  // std::is_same
#include <boost/optional.hpp>
#include <boost/signals2.hpp>

#include "../exceptions.h"
#include "../mesh/mesh.h"
#include "../mesh/interpolation.h"

namespace plask {

/**
 * Base class for all Providers.
 *
 * It implements listener (observer) pattern (can be observed by Receiver).
 *
 * Subclasses should only have implemented operator()(...) which return provided value, or throw NoValue exception.
 * Receiver (for given provider type) can be easy implemented by inherit Receiver class template.
 *
 * @see @ref providers
 */
struct Provider {

    static constexpr const char* NAME = "undefined";

    Provider & operator=(const Provider&) = delete;
    Provider(const Provider&) = delete;
    Provider() = default;

    /**
     * Provider listener (observer). Can react to Provider changes.
     */
    struct Listener {
        ///Called when provider value was changed.
        virtual void onChange() = 0;

        /**
         * Called just before disconnecting. By default does nothing.
         * @param from_where provider from which listener is being disconnected
         */
        virtual void onDisconnect(Provider* from_where) {}

        virtual ~Listener() {}
    };

    ///Set of added (registered) listeners. This provider can call methods of listeners included in this set.
    std::set<Listener*> listeners;

    ///Call onDisconnect for all lighteners in listeners set.
    virtual ~Provider() {
        for (typename std::set<Listener*>::iterator i = listeners.begin(); i != listeners.end(); ++i)
            (*i)->onDisconnect(this);
    }

    /**
     * Add listener to listeners set.
     * @param listener listener to add (register)
     */
    void add(Listener* listener) {
        listeners.insert(listener);
    }

    /**
     * Remove (unregister) listener from listeners set.
     * @param listener listener to remove (unregister)
     */
    void remove(Listener* listener) {
        listener->onDisconnect(this);
        listeners.erase(listener);
    }

    /**
     * Call onChange for all listeners.
     * Should be called after change of value represented by this provider.
     */
    void fireChanged() {
        for (typename std::set<Listener*>::iterator i = listeners.begin(); i != listeners.end(); ++i)
            (*i)->onChange();
    }

};

/**
 * Base class for all Receivers.
 *
 * Implement listener (observer) pattern (is listener for providers).
 * Delegate all operator() calling to provider.
 *
 * For most providers types, Receiver type can be defined as: <code>Receiver<ProviderClass>;</code>
 * (where <code>ProviderClass</code> is type of provider class)
 *
 * @tparam ProviderT type of provider, can has defined ProviderT::ConstProviderT to reciver setConst method work.
 *
 * @see @ref providers
 */
template <typename ProviderT>
struct Receiver: public Provider::Listener {

    /// Name of provider.
    static constexpr const char* PROVIDER_NAME = ProviderT::NAME;

    /// Signal called when provider value or provider was changed (called by onChange)
    boost::signals2::signal<void(Receiver& src)> providerValueChanged;

    Receiver & operator=(const Receiver&) = delete;
    Receiver(const Receiver&) = delete;

    /// Pointer to connected provider. Can be nullptr if no provider is connected.
    ProviderT* provider;

    /// Is @c true only if data provides by provider was changed after previous value retrieval.
    bool changed;

    /// Is @c true only if provider is private and will be deleted by this receiver.
    bool hasPrivateProvider;

    /// Construct Receiver without connected provider and with set changed flag.
    Receiver(): provider(0), changed(true), hasPrivateProvider(false) {}

    /// Destructor. Disconnect from provider.
    ~Receiver() {
        setProvider(0);
    }

    /**
     * Change provider. If new provider is different from current one then changed flag is set.
     * @param provider new provider, can be @c nullptr to only disconnect from current provider.
     * @param newProviderIsPrivate @c true only if @p provider is private for this and will be delete by this receiver
     */
    void setProvider(ProviderT* provider, bool newProviderIsPrivate = false) {
        if (this->provider == provider) return;
        if (this->provider) this->provider->listeners.erase(this);
        if (hasPrivateProvider) delete this->provider;
        if (provider) provider->add(this);
        this->provider = provider;
        this->hasPrivateProvider = newProviderIsPrivate;
        onChange();
    }

    /**
     * Change provider. If new provider is different from current one then changed flag is set.
     * @param provider new provider
     */
    void setProvider(ProviderT &provider) {
        setProvider(&provider);
    }

    /**
     * Change provider. If new provider is different from current one then changed flag is set.
     * @param provider new provider, can be @c nullptr to only disconnect from current provider.
     */
    void operator<<(ProviderT *provider) {
        setProvider(provider);
    }

    /**
     * Change provider. If new provider is different from current one then changed flag is set.
     * @param provider new provider
     */
    void operator<<(ProviderT &provider) {
        setProvider(&provider);
    }

    /**
     * Current provider getter.
     * @return current provider or @c nullptr if there is no connected provider
     */
    ProviderT* getProvider() { return provider; }

    /**
     * Current provider getter.
     * @return current provider or @c nullptr if there is no connected provider
     */
    const ProviderT* getProvider() const { return provider; }

    /// React on provider value changes. Set changed flag to true.
    void onChange() {
        changed = true;
        providerValueChanged(*this);
    }

    virtual void onDisconnect(Provider* from_where) {
        if (from_where == provider) {
            provider = 0;
            onChange();
        }
    }

    ///@throw NoProvider when provider is not available
    void ensureHasProvider() {
        if (!provider) throw NoProvider(PROVIDER_NAME);
    }

    /**
     * Get value from provider using its operator().
     * @return value from provider
     * @throw NoProvider when provider is not available
     * @throw NoValue when provider can't give value (is uninitialized, etc.)
     */
    //TODO const version? only const version?
    template<typename ...Args> auto
    operator()(Args&&... params) -> decltype((*provider)(std::forward<Args>(params)...)) {
        beforeGetValue();
        return (*provider)(std::forward<Args>(params)...);
    }

    /**
     * Get value from provider using its operator().
     * If value can't be gotten (there is no provider or provider can't give value) empty optional is returned.
     * @return value from provider or empty optional if value couldn't be got
     */
    template<typename ...Args> auto
    optional(Args&&... params) -> boost::optional<decltype((*provider)(std::forward<Args>(params)...))> {
        try {
            return boost::optional<decltype((*provider)(std::forward<Args>(params)...))>(this->operator()(std::forward<Args>(params)...));
        } catch (std::exception&) {
            return boost::optional<decltype((*provider)(std::forward<Args>(params)...))>();
        }
    }


    /**
     * Set provider for this to provider of constant.
     *
     * Use ProviderT::ConstProviderT as provider of const type.
     * @param constProviderConstructorArgs parameters passed to ProviderT::ConstProviderT constructor
     */
    template <typename ...ConstProviderConstructorArgs>
    void setValue(ConstProviderConstructorArgs&&... constProviderConstructorArgs) {
        setProvider(new typename ProviderT::ConstProviderT(std::forward<ConstProviderConstructorArgs>(constProviderConstructorArgs)...), true);
    }

protected:

    /**
     * Check if value can be read and throw exception if not.
     * Set changed flag to false.
     *
     * Subclass can call this just before reading value.
     *
     * @throw NoProvider when provider is not available
     */
    void beforeGetValue() {
        ensureHasProvider();
        changed = false;
    }

};

/**
 * Instantiation of this template is abstract base class for provider which provide one value (for example one double).
 * @tparam ValueT type of provided value
 */
template <typename ValueT>
struct SingleValueProvider: public Provider {

    static constexpr const char* NAME = "undefined value";

    /// Type of provided value.
    typedef ValueT ProvidedValueType;

    /**
     * Provided value getter.
     * @return provided value
     */
    virtual ProvidedValueType operator()() const = 0;

};

//TODO typedef for SingleValueReceiver (GCC 4.7 needed)

/**
 * Instantiation of this template is abstract base class for provider which provide values in points described by mesh
 * and don't use interpolation.
 */
template <typename ValueT, typename SpaceT>
struct OnMeshProvider: public Provider {

    static constexpr const char* NAME = "undefined field";

    /// Type of value provided by this (returned by operator()).
    typedef DataVector<ValueT> ProvidedValueType;

    /**
     * @param dst_mesh set of requested points
     * @return values in points describe by mesh @a dst_mesh
     */
    virtual ProvidedValueType operator()(const Mesh<SpaceT::DIMS>& dst_mesh) const = 0;

};

//TODO typedef for OnMeshReceiver (GCC 4.7 needed)

/**
 * Instantiation of this template is abstract base class for provider class which provide values in points describe by mesh
 * and use interpolation.
 */
template <typename ValueT, typename SpaceT>
struct OnMeshProviderWithInterpolation: public OnMeshProvider<ValueT, SpaceT> {

    static constexpr const char* NAME = "undefined field";

    /// Type of value provided by this (returned by operator()).
    typedef typename OnMeshProvider<ValueT, SpaceT>::ProvidedValueType ProvidedValueType;

    /**
     * @param dst_mesh set of requested points
     * @param method method which should be use to do interpolation
     * @return values in points describe by mesh @a dst_mesh
     */
    virtual ProvidedValueType operator()(const Mesh<SpaceT::DIMS>& dst_mesh, InterpolationMethod method) const = 0;

    /**
     * Implementation of OnMeshProvider method, call this->operator()(dst_mesh, DEFAULT).
     * @param dst_mesh set of requested points
     * @return values in points describe by mesh @a dst_mesh
     */
    virtual ProvidedValueType operator()(const Mesh<SpaceT::DIMS>& dst_mesh) const {
        return this->operator()(dst_mesh, DEFAULT_INTERPOLATION);
    }

};

//TODO typedef for OnMeshReceiverWithInterpolation (GCC 4.7 needed)

template<typename _Signature> struct DelegateProvider;

/**
 * Template of class which is good base class for providers which delegate calls of operator() to external functor
 * (function or method).
 * @tparam _Res(_ArgTypes...) functor signature (result and arguments types)
 */
template<typename _Res, typename... _ArgTypes>
struct DelegateProvider<_Res(_ArgTypes...)>: public Provider {

    /// Hold external functor.
    std::function<_Res(_ArgTypes...)> valueGetter;

    /**
     * Initialize valueGetter using given params.
     * @param params parameters for valueGetter constructor
     */
    template<typename ...Args>
    DelegateProvider<_Res(_ArgTypes...)>(Args&&... params)
    : valueGetter(std::forward<Args>(params)...) {
    }

    /**
     * Call functor hold by valueGetter.
     * @param params parameters for functor hold by valueGetter
     * @return value returned by functor hold by valueGetter
     */
    virtual _Res operator()(_ArgTypes&&... params) const {
        return valueGetter(std::forward<_ArgTypes>(params)...);
    }
};

template<typename _BaseClass, typename _Signature> struct PolymorphicDelegateProvider;



/**
 * Template of class which is a good base class for providers which delegate calls of operator() to external functor
 * (function or method).
 * @tparam _Res(_ArgTypes...) functor signature (result and arguments types)
 */
template<typename _BaseClass, typename _Res, typename... _ArgTypes>
struct PolymorphicDelegateProvider<_BaseClass, _Res(_ArgTypes...)>: public _BaseClass {

    /// Hold external functor.
    std::function<_Res(_ArgTypes...)> valueGetter;

    template<typename ClassType, typename MemberType>
    PolymorphicDelegateProvider<_BaseClass, _Res(_ArgTypes...)>(ClassType* object, MemberType member)
        : valueGetter(
          [object, member](_ArgTypes&&... params) {
              return (object->*member)(std::forward<_ArgTypes>(params)...);
          })
    {}

    /**
     * Initialize valueGetter using given parameters.
     * @param params parameters for valueGetter constructor
     */
    template<typename ...Args>
    PolymorphicDelegateProvider<_BaseClass, _Res(_ArgTypes...)>(Args&&... params)
    : valueGetter(std::forward<Args>(params)...) {
    }

    /**
     * Call functor hold by valueGetter.
     * @param params parameters for functor hold by valueGetter
     * @return value returned by functor hold by valueGetter
     */
    _Res operator()(_ArgTypes... params) const {
        return valueGetter(std::forward<_ArgTypes>(params)...);
    }
};

/**
 * Type of properties.
 * @see @ref providers
 */
enum PropertyType {
    SINGLE_VALUE_PROPERTY = 0,	        ///< Single value property
    FIELD_PROPERTY = 1,			///< Property for field of values which can't be interpolated
    INTERPOLATED_FIELD_PROPERTY = 2	///< Property for field of values which can be interpolated
};	//TODO change this to empty classes(?)

template <PropertyType prop_type>
struct PropertyTypeToProviderName {
    static constexpr const char* value = "undefined";
};

template <>
struct PropertyTypeToProviderName<SINGLE_VALUE_PROPERTY> {
    static constexpr const char* value = "undefined value";
};

template <>
struct PropertyTypeToProviderName<FIELD_PROPERTY> {
    static constexpr const char* value = "undefined field";
};

template <>
struct PropertyTypeToProviderName<INTERPOLATED_FIELD_PROPERTY> {
    static constexpr const char* value = "undefined field";
};

/**
 * Helper class which makes easiest to define property tags class.
 *
 * Property tags class are used for ProviderFor and ReceiverFor templates instantiations.,
 *
 * Properties tag class can be subclass of this, but never should be typedefs to this
 * (tag class for each property must by separate class - always use different types for different properties).
 */
template <PropertyType _propertyType, typename _ValueType>
struct Property {
    ///Type of property.
    static const PropertyType propertyType = _propertyType;
    ///Type of provided value.
    typedef _ValueType ValueType;

    static constexpr const char* NAME = PropertyTypeToProviderName<_propertyType>::value;
};

/**
 * Helper class which makes easiest to define property tags class for single value (double type by default) properties.
 *
 * Properties tag class can be subclass of this, but never should be typedefs to this
 * (tag class for each property must by separate class - always use different types for different properties).
 */
template<typename ValueT = double>
struct SingleValueProperty: public Property<SINGLE_VALUE_PROPERTY, ValueT> {};

/**
 * Helper class which makes easiest to define property tags class for non-scalar fields.
 *
 * Properties tag class can be subclass of this, but never should be typedefs to this
 * (tag class for each property must by separate class - always use different types for different properties).
 */
template<typename ValueT>
struct FieldProperty: public Property<FIELD_PROPERTY, ValueT> {};

/**
 * Helper class which makes easiest to define property tags class for possible to interpolate fields.
 *
 * Properties tag class can be subclass of this, but never should be typedefs to this
 * (tag class for each property must by separate class - always use different types for different properties).
 */
template<typename ValueT = double>
struct InterpolatedFieldProperty: public Property<INTERPOLATED_FIELD_PROPERTY, ValueT> {};

/**
 * Helper class which makes easiest to define property tags class for scalar fields (fields of doubles).
 *
 * Properties tag class can be subclass of this, but never should be typedefs to this
 * (tag class for each property must by separate class - always use different types for different properties).
 */
typedef InterpolatedFieldProperty<double> ScalarFieldProperty;

/**
 * Specializations of this class are implementations of providers for given property tag class and this tag properties.
 *
 * Don't use this class directly. Use plask::Provider class or plask::ProviderFor template.
 */
template <typename PropertyT, typename ValueT, PropertyType propertyType, typename spaceType>
struct ProviderImpl {};

/**
 * Specializations of this class define implementations of providers for given property tag:
 * - ProviderFor<PropertyT, SpaceT> is abstract, base class which inherited from Provider;
 * - ProviderFor<PropertyT, SpaceT>::Delegate is class inherited from ProviderFor<PropertyT, SpaceT> which delegates all request to functor given as constructor parameter;
 * - ProviderFor<PropertyT, SpaceT>::WithValue is class inherited from ProviderFor<PropertyT, SpaceT> which stores provided value (has value field) and know if it was initialized;
 * - ProviderFor<PropertyT, SpaceT>::WithDefaultValue is class inherited from ProviderFor<PropertyT, SpaceT> which stores provided value (has value field) and doesn't know if it was initialized (should always have reasonable default value).
 * @tparam PropertyT property tag class (describe physical property)
 * @tparam SpaceT type of space, required (and allowed) only for fields properties
 * @see plask::Temperature (includes example); @ref providers
 */
template <typename PropertyT, typename SpaceT = void>
struct ProviderFor: public ProviderImpl<PropertyT, typename PropertyT::ValueType, PropertyT::propertyType, SpaceT> {

    typedef PropertyT PropertyTag;
    typedef SpaceT SpaceType;

    /// Delegate all constructors to parent class.
    template<typename ...Args>
    ProviderFor(Args&&... params)
    : ProviderImpl<PropertyT, typename PropertyT::ValueType, PropertyT::propertyType, SpaceT>(std::forward<Args>(params)...) {
    }

};
//TODO redefine ProviderFor using template aliases (require gcc 4.7), and than fix ReceiverFor

/**
 * Specializations of this class are implementations of Receiver for given property tag.
 * @tparam PropertyT property tag class (describe physical property)
 * @tparam SpaceT type of space, required (and allowed) only for fields properties
 */
template <typename PropertyT, typename SpaceT = void>
struct ReceiverFor: public Receiver< ProviderImpl<PropertyT, typename PropertyT::ValueType, PropertyT::propertyType, SpaceT> > {
    ReceiverFor & operator=(const ReceiverFor&) = delete;
    ReceiverFor(const ReceiverFor&) = delete;
    ReceiverFor() = default;

    typedef PropertyT PropertyTag;
    typedef SpaceT SpaceType;

    /**
     * Set provider for this to provider of constant.
     *
     * Use ProviderT::ConstProviderT as provider of const type.
     * @param v value which should be provided for this receiver
     * @return *this
     */
    ReceiverFor<PropertyT, SpaceT>& operator=(const typename PropertyT::ValueType& v) {
        this->setValue(v);
        return *this;
    }

    static_assert(!(std::is_same<SpaceT, void>::value && (PropertyT::propertyType == FIELD_PROPERTY || PropertyT::propertyType == INTERPOLATED_FIELD_PROPERTY)),
                  "Receivers for fields properties require SpaceT. Use ReceiverFor<propertyTag, SpaceT>, where SpaceT is one of the classes defined in .");
    static_assert(!(!std::is_same<SpaceT, void>::value && (PropertyT::propertyType == SINGLE_VALUE_PROPERTY)),
                  "Receivers for single value properties doesn't need SpaceT. Use ReceiverFor<propertyTag> (without second template parameter).");
};
//struct ReceiverFor: public Receiver< ProviderFor<PropertyT> > {};

/**
 * Partial specialization which implements abstract provider class which provides a single value, typically one double.
 *
 * @tparam PropertyT
 * @tparam ValueT type of provided value
 * @tparam SpaceT ignored
 */
template <typename PropertyT, typename ValueT, typename SpaceT>
struct ProviderImpl<PropertyT, ValueT, SINGLE_VALUE_PROPERTY, SpaceT>: public SingleValueProvider<ValueT> {

    static constexpr const char* NAME = PropertyT::NAME;

    static_assert(std::is_same<SpaceT, void>::value,
                  "Providers for single value properties doesn't need SpaceT. Use ProviderFor<propertyTag> (without second template parameter).");

    /// Type of provided value.
    typedef typename SingleValueProvider<ValueT>::ProvidedValueType ProvidedValueType;

    /**
     * Implementation of one value provider class which holds value inside (in value field) and operator() returns this hold value.
     */
    struct WithDefaultValue: public ProviderFor<PropertyT, SpaceT> {

        /// Type of provided value.
        typedef ValueT ProvidedValueType;

        /// Provided value.
        ProvidedValueType value;

        /// Delegate all constructors to value.
        template<typename ...Args>
        WithDefaultValue(Args&&... params): value(std::forward<Args>(params)...) {}

        /**
         * Set new value.
         * @param v new value
         * @return *this
         */
        WithDefaultValue& operator=(const ValueT& v) {
            value = v;
            return *this;
        }

        /**
         * Get provided value.
         * @return provided value
         */
        ProvidedValueType& operator()() { return value; }

        /**
         * Get provided value.
         * @return provided value
         */
        virtual ProvidedValueType operator()() const { return value; }
    };

    /**
     * Implementation of one value provider class which holds value inside (in value field) and operator() return this hold value.
     */
    struct WithValue: public ProviderFor<PropertyT, SpaceT> {

        /// Type of provided value.
        typedef ValueT ProvidedValueType;

        /// Provided value.
        boost::optional<ProvidedValueType> value;

        /// Reset value to be uninitialized.
        void invalidate() { value.reset(); }

        /**
         * Check if this has value / is initialized.
         * @return @c true only if this is initialized (has value)
         */
        bool hasValue() const { return value; }

        /// Throw NoValue exception if value is not initialized.
        void ensureHasValue() const {
            if (!hasValue()) throw NoValue(NAME);
        }

        /// Delegate all constructors to value.
        template<typename ...Args>
        WithValue(Args&&... params): value(ProvidedValueType(std::forward<Args>(params)...)) {}

        /// Create empty boost::optional value.
        WithValue() {}

        /**
         * Set new value.
         * @param v new value
         * @return *this
         */
        WithValue& operator=(const ValueT& v) {
            value.reset(v);
            return *this;
        }

        /**
         * Get provided value.
         * @return provided value
         * @throw NoValue if value is empty boost::optional
         */
        ProvidedValueType& operator()() {
            ensureHasValue();
            return *value;
        }

        /**
         * Get provided value.
         * @return provided value
         * @throw NoValue if value is empty boost::optional
         */
        virtual ProvidedValueType operator()() const {
            ensureHasValue();
            return *value;
        }
    };

    /**
     * Implementation of one value provider class which delegates all operator() calls to external functor.
     */
    typedef PolymorphicDelegateProvider<ProviderFor<PropertyT, SpaceT>, ProvidedValueType()> Delegate;

    /// Used by receivers as const value provider, see Receiver::setConst
    typedef WithValue ConstProviderT;

};

/**
 * Partial specialization which implements providers classes which provide values in mesh points
 * and don't use interpolation.
 */
template <typename PropertyT, typename ValueT, typename SpaceT>
struct ProviderImpl<PropertyT, ValueT, FIELD_PROPERTY, SpaceT>: public OnMeshProvider<ValueT, SpaceT> {

    static constexpr const char* NAME = PropertyT::NAME;

    static_assert(!std::is_same<SpaceT, void>::value,
                  "Providers for fields properties require SpaceT. Use ProviderFor<propertyTag, SpaceT>, where SpaceT is one of the class defined in .");

    /// Type of provided value.
    typedef typename OnMeshProvider<ValueT, SpaceT>::ProvidedValueType ProvidedValueType;

    /**
     * Implementation of  field provider class which delegates all operator() calls to external functor.
     */
    typedef PolymorphicDelegateProvider<ProviderFor<PropertyT, SpaceT>, ProvidedValueType(const Mesh<SpaceT::DIMS>& dst_mesh)> Delegate;

    /**
     * Return same value in all points.
     *
     * Used by receivers as const value provider, see Receiver::setConst
     */
    struct ConstProviderT: public ProviderFor<PropertyT, SpaceT> {

        typedef ProviderImpl<PropertyT, ValueT, FIELD_PROPERTY, SpaceT>::ProvidedValueType ProvidedValueType;

        ValueT value;

        //ConstProviderT(const ValueT& value): value(value) {}

        template<typename ...Args>
        ConstProviderT(Args&&... params): value(std::forward<Args>(params)...) {}

        virtual ProvidedValueType operator()(const Mesh<SpaceT::DIMS>& dst_mesh) const {
            //return copy of value for each point in dst_mesh
            //return make_shared< const std::vector<ValueT> >(dst_mesh.size(), value);
            return ProvidedValueType(dst_mesh.size(), value);
        }
    };

};

/**
 * Specialization which implements provider class which provides values in mesh points and uses interpolation.
 */
template <typename PropertyT, typename ValueT, typename SpaceT>
struct ProviderImpl<PropertyT, ValueT, INTERPOLATED_FIELD_PROPERTY, SpaceT>: public OnMeshProviderWithInterpolation<ValueT, SpaceT> {

    static constexpr const char* NAME = PropertyT::NAME;

    static_assert(!std::is_same<SpaceT, void>::value,
                  "Providers for fields properties require SpaceT. Use ProviderFor<propertyTag, SpaceT>, where SpaceT is one of the class defined in .");

    ///Type of provided value.
    typedef typename OnMeshProviderWithInterpolation<ValueT, SpaceT>::ProvidedValueType ProvidedValueType;

    /**
     * Template for implementation of field provider class which holds vector of values and mesh inside.
     * operator() call plask::interpolate.
     * @tparam MeshPtrType type of pointer (shared_ptr or unique_ptr) to mesh which is used for calculation and which describe places of data points
     */
    template <typename MeshPtrType>
    struct WithValue: public ProviderFor<PropertyT, SpaceT> {

        /// Type of mesh pointer
        typedef MeshPtrType MeshPointerType;

        /// Type of provided value.
        typedef ProviderImpl<PropertyT, ValueT, INTERPOLATED_FIELD_PROPERTY, SpaceT>::ProvidedValueType ProvidedValueType;

        /// Provided value. Values in points describe by this->mesh.
        ProvidedValueType values;

        /// Mesh which describes in which points there are this->values.
        MeshPtrType meshPtr;

        /**
         * Get mesh.
         * @return @c *meshPtr
         */
        auto mesh() -> decltype(*meshPtr) { return *meshPtr; }

        /**
         * Get mesh (const).
         * @return @c *meshPtr
         */
        auto mesh() const -> decltype(*meshPtr) { return *meshPtr; }

        /// Reset values to uninitilized state (nullptr data).
        void invalidate() { values.reset(); }

        /// Reserve memory in values using mesh size.
        void allocate() { values.reset(mesh().size()); }

        /**
         * Check if this has value / is initialized.
         * @return @c true only if this is initialized (has value)
         */
        bool hasValue() const { return values.data() != nullptr; }

        /// Throw NoValue exception if value is not initialized.
        void ensureHasValue() const {
            if (!hasValue()) throw NoValue(NAME);
        }

        /**
         * @param values provided value, values in points describe by this->mesh.
         * @param meshPtr pointer to mesh which describes in which points there are this->values
         */
        explicit WithValue(ProvidedValueType values, MeshPtrType meshPtr = nullptr)
            : values(values), meshPtr(meshPtr) {}

        /**
         * @param meshPtr pointer to mesh which describes in which points there are this->values
         */
        explicit WithValue(MeshPtrType meshPtr = nullptr)
            : meshPtr(meshPtr) {}

        /**
         * Get provided value in points describe by this->mesh.
         * @return provided value in points describe by this->mesh
         */
        ProvidedValueType& operator()() { return values; }

        /**
         * Get provided value in points describe by this->mesh.
         * @return provided value in points describe by this->mesh
         */
        const ProvidedValueType& operator()() const { return values; }

        /**
         * Calculate interpolated values using plask::interpolate.
         * @param dst_mesh set of requested points
         * @param method method which should be use to do interpolation
         * @return values in points describe by mesh @a dst_mesh
         */
        virtual ProvidedValueType operator()(const Mesh<SpaceT::DIMS>& dst_mesh, InterpolationMethod method) const {
            ensureHasValue();
            return interpolate(mesh(), values, dst_mesh, method);
        }
    };

    /**
     * Implementation of field provider class which delegates all operator() calls to external functor.
     */
    typedef PolymorphicDelegateProvider<ProviderFor<PropertyT, SpaceT>, ProvidedValueType(const Mesh<SpaceT::DIMS>& dst_mesh, InterpolationMethod method)> Delegate;

    /**
     * Return same value in all points.
     *
     * Used by receivers as const value provider, see Receiver::setConst
     */
    struct ConstProviderT: public ProviderFor<PropertyT, SpaceT> {

        typedef ProviderImpl<PropertyT, ValueT, INTERPOLATED_FIELD_PROPERTY, SpaceT>::ProvidedValueType ProvidedValueType;

        /// Provided value
        ValueT value;

        //ConstProviderT(const ValueT& value): value(value) {}

        /**
         * Constructor which delegate all parameters to value constructor.
         * @param params ValueT constructor parameters, forwarded to value
         */
        template<typename ...Args>
        ConstProviderT(Args&&... params): value(std::forward<Args>(params)...) {}

        /**
         * @return copy of value for each point in dst_mesh, ignore interpolation method
         */
        virtual ProvidedValueType operator()(const Mesh<SpaceT::DIMS>& dst_mesh, InterpolationMethod) const {
            return ProvidedValueType(dst_mesh.size(), value);
        }
    };

};

}; //namespace plask

#endif  //PLASK__PROVIDER_H
