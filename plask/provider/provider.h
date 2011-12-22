#ifndef PLASK__PROVIDER_H
#define PLASK__PROVIDER_H

/** @file
This file includes base classes and templates which allow to generate providers and receivers.
@see @ref providers


@page providers Provider and receivers

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

Example:
@code
TODO
@endcode

@section providers_writing Writing new providers and receivers types

@subsection providers_writing_easy Easy (half-automatic) way
The easiest way to create new provider and corresponding receiver types is to write physical property
tag class and use it to specialize plask::ProviderFor and plask::ReceiverFor templates.

Physical property tag class is an class which only has static fields and typedefs which describe
physical property. It can be easy obtain by subclass specialization of one of templates:
- plask::Property - allow to obtain all possible physical properties tags classes, but require many parameters
- plask::SingleValueProperty - allow to obtain tags for properties described by one value (typically one scalar), require only one parameter - type of provided value
- plask::FieldProperty
- plask::InterpolatedFieldProperty
- plask::ScalarFieldProperty

Example:
@code
//Physical property tag class for temperature.
struct Temperature: ScalarFieldProperty {};

//Type for provider class:
typedef ProviderFor<Temperature> TemperatureProvider;

//Type for receiver class:
typedef ReceiverFor<Temperature> TemperatureReceiver;
@endcode

@subsection providers_writing_easy Flexible (manual) way



*/

#include <set>
#include <config.h>
#include <vector>
#include <functional>	//std::function

#include "../exceptions.h"
#include "../mesh/mesh.h"
#include "../mesh/interpolation.h"

namespace plask {

/**
 * Base class for all Providers.
 *
 * It implements listener (observer) pattern (can be observed by Receiver).
 *
 * Subclasses should only have implemented operator()(...) which return provided value.
 * Receiver (for given provider type) can be easy implemented by inherit Receiver class template.
 *
 * @see @ref providers
 */
struct Provider {

    /**
     * Provider listener (observer). Can react to Provider changes.
     */
    struct Listener {
        ///Called when provider value was changed.
        virtual void onChange() = 0;

        /**
         * Called just before disconnect. By default do nothing.
         * @param from_where provider from which listener is disconnected
         */
        virtual void onDisconnect(Provider* from_where) {}
    };

    ///Set of added (registered) listeners. This provider can call methods of listeners included in this set.
    std::set<Listener*> listeners;

    ///Call onDisconnect for all lighteners in listeners set.
    ~Provider() {
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
 * @tparam ProviderT type of provider
 *
 * @see @ref providers
 */
template <typename ProviderT>
struct Receiver: public Provider::Listener {

    ///Pointer to connected provider. Can be nullptr if no provider is connected.
    ProviderT* provider;

    ///Is @c true only if data provides by provider was changed after recent value getting.
    bool changed;

    ///Construct Receiver without connected provider and with set changed flag.
    Receiver(): provider(0), changed(true) {}

    ///Destructor. Disconnect from provider.
    ~Receiver() {
        setProvider(0);
    }

    /**
     * Change provider. If new provider is different from current one then changed flag is set.
     * @param provider new provider, can be @c nullptr to only disconnect from current provider.
     */
    void setProvider(ProviderT* provider) {
        if (this->provider == provider) return;
        if (this->provider) this->provider->listeners.erase(this);
        if (provider) provider->add(this);
        this->provider = provider;
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
    void operator=(ProviderT *provider) {
        setProvider(&provider);
    }

    /**
     * Change provider. If new provider is different from current one then changed flag is set.
     * @param provider new provider
     */
    void operator=(ProviderT &provider) {
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

    ///React on provider value changes. Set changed flag to true.
    void onChange() {
        changed = true;
        //TODO callback?
    }

    virtual void onDisconnect(Provider* from_where) {
        if (from_where == provider) {
            provider = 0;
            onChange();
        }
    }

    ///@throw NoProvider when provider is not available
    void ensureHasProvider() {
        if (!provider) throw NoProvider();	//TODO some name, maybe Provider should have virtual name or name field?
    }

    /**
     * Get value from provider using its operator().
     * @return value from provider
     * @throw NoProvider when provider is not available
     */
    //TODO const version? only const version?
    template<typename ...Args> auto
    operator()(Args&&... params) -> decltype((*provider)(std::forward<Args>(params)...)) {
        beforeGetValue();
        return (*provider)(std::forward<Args>(params)...);
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
 * Specialization of this template is abstract base class for provider which provide one value (for example one double).
 * @tparam ValueT type of provided value
 */
template <typename ValueT>
struct SingleValueProvider: public Provider {

    ///Type of provided value.
    typedef ValueT ProvidedValueType;

    /**
     * Provided value getter.
     * @return provided value
     */
    virtual ProvidedValueType operator()() const = 0;

};

//TODO typedef for SingleValueReceiver (GCC 4.7 needed)

/**
 * Specialization of this template is abstract base class for provider which provide values in points describe by mesh
 * and don't use interpolation.
 */
template <typename ValueT, typename SpaceType>
struct OnMeshProvider: public Provider {

    ///Type of value provided by this (returned by operator()).
    typedef shared_ptr< const std::vector<ValueT> > ProvidedValueType;

    /**
     * @param dst_mesh set of requested points
     * @return values in points describe by mesh @a dst_mesh
     */
    virtual ProvidedValueType operator()(const Mesh<SpaceType>& dst_mesh) const = 0;

};

//TODO typedef for OnMeshReceiver (GCC 4.7 needed)

/**
 * Specialization of this template is abstract base class for provider class which provide values in points describe by mesh
 * and use interpolation.
 */
template <typename ValueT, typename SpaceType>
struct OnMeshProviderWithInterpolation: public OnMeshProvider<ValueT, SpaceType> {

    ///Type of value provided by this (returned by operator()).
    typedef typename OnMeshProvider<ValueT, SpaceType>::ProvidedValueType ProvidedValueType;

    /**
     * @param dst_mesh set of requested points
     * @param method method which should be use to do interpolation
     * @return values in points describe by mesh @a dst_mesh
     */
    virtual ProvidedValueType operator()(const Mesh<SpaceType>& dst_mesh, InterpolationMethod method) const = 0;

    /**
     * Implementation of OnMeshProvider method, call this->operator()(dst_mesh, DEFAULT).
     * @param dst_mesh set of requested points
     * @return values in points describe by mesh @a dst_mesh
     */
    virtual ProvidedValueType operator()(const Mesh<SpaceType>& dst_mesh) const {
        return this->operator()(dst_mesh, DEFAULT);
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
     * Call functor holded by valueGetter.
     * @param params parameters for functor holded by valueGetter
     * @return value returned by functor holded by valueGetter
     */
    _Res operator()(_ArgTypes&&... params) const {
        return valueGetter(std::forward<_ArgTypes>(params)...);
    }
};

template<typename _BaseClass, typename _Signature> struct PolymorphicDelegateProvider;

/**
 * Template of class which is good base class for providers which delegate calls of operator() to external functor
 * (function or method).
 * @tparam _Res(_ArgTypes...) functor signature (result and arguments types)
 */
template<typename _BaseClass, typename _Res, typename... _ArgTypes>
struct PolymorphicDelegateProvider<_BaseClass, _Res(_ArgTypes...)>: public _BaseClass {

    /// Hold external functor.
    std::function<_Res(_ArgTypes...)> valueGetter;

    /**
     * Initialize valueGetter using given params.
     * @param params parameters for valueGetter constructor
     */
    template<typename ...Args>
    PolymorphicDelegateProvider<_BaseClass, _Res(_ArgTypes...)>(Args&&... params)
    : valueGetter(std::forward<Args>(params)...) {
    }

    /**
     * Call functor holded by valueGetter.
     * @param params parameters for functor holded by valueGetter
     * @return value returned by functor holded by valueGetter
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

/**
 * Helper class which makes easiest to define property tags class.
 *
 * Property tags class are used for Provider and Receiver templates specializations.,
 *
 * Properties tag class can be subclass of this, but never should be typedefs to this
 * (tag class for each property must by separate class - always use different types for different properties).
 */
template <PropertyType _propertyType, typename _ValueType, typename _spaceType = void>
struct Property {
    ///Type of property.
    static const PropertyType propertyType = _propertyType;
    ///Type of provided value.
    typedef _ValueType ValueType;
    ///Type of space.
    typedef _spaceType SpaceType;
};

/**
 * Helper class which makes easiest to define property tags class for single value (double type by default) properties.
 *
 * Properties tag class can be subclass of this, but never should be typedefs to this
 * (tag class for each property must by separate class - always use different types for different properties).
 */
template<typename ValueType = double>
struct SingleValueProperty: public Property<SINGLE_VALUE_PROPERTY, ValueType> {};

/**
 * Helper class which makes easiest to define property tags class for non-scalar fields.
 *
 * Properties tag class can be subclass of this, but never should be typedefs to this
 * (tag class for each property must by separate class - always use different types for different properties).
 */
template<typename ValueType>
struct FieldProperty: public Property<FIELD_PROPERTY, ValueType> {};

/**
 * Helper class which makes easiest to define property tags class for possible to interpolate fields.
 *
 * Properties tag class can be subclass of this, but never should be typedefs to this
 * (tag class for each property must by separate class - always use different types for different properties).
 */
template<typename ValueType = double>
struct InterpolatedFieldProperty: public Property<INTERPOLATED_FIELD_PROPERTY, ValueType> {};

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
template <typename PropertyTag, typename ValueType, PropertyType propertyType, typename spaceType>
struct ProviderImpl {};

/**
 * Specializations of this class are implementations of providers for given property tag.
 */
template <typename PropertyTag>
struct ProviderFor: public ProviderImpl<PropertyTag, typename PropertyTag::ValueType, PropertyTag::propertyType, typename PropertyTag::SpaceType> {

    /// Delegate all constructors to parent class.
    template<typename ...Args>
    ProviderFor(Args&&... params)
    : ProviderImpl<PropertyTag, typename PropertyTag::ValueType, PropertyTag::propertyType, typename PropertyTag::SpaceType>(std::forward<Args>(params)...) {
    }

};
//TODO redefine ProviderFor using template aliases (require gcc 4.7), and than fix ReceiverFor

/**
 * Specializations of this class are implementations of Receiver for given property tag.
 */
template <typename PropertyTag>
struct ReceiverFor: public Receiver< ProviderImpl<PropertyTag, typename PropertyTag::ValueType, PropertyTag::propertyType, typename PropertyTag::SpaceType> > {};
//struct ReceiverFor: public Receiver< ProviderFor<PropertyTag> > {};

/**
 * Partial specialization which implement abstract provider class which provide one value, typically one double.
 * 
 * @tparam PropertyTag
 * @tparam ValueT type of provided value
 * @tparam SpaceType ignored
 */
template <typename PropertyTag, typename ValueT, typename SpaceType>
struct ProviderImpl<PropertyTag, ValueT, SINGLE_VALUE_PROPERTY, SpaceType>: public SingleValueProvider<ValueT> {

    ///Type of provided value.
    typedef typename  SingleValueProvider<ValueT>::ProvidedValueType ProvidedValueType;

    /**
     * Implementation of one value provider class which holds value inside (in value field) and operator() return this holded value.
     */
    struct WithValue: public ProviderImpl<PropertyTag, ValueT, SINGLE_VALUE_PROPERTY, SpaceType> {
        
        ///Type of provided value.
        typedef ValueT ProvidedValueType;
        
        ///Provided value.
        ProvidedValueType value;
        
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
     * Implementation of one value provider class which delegates all operator() calls to external functor.
     */
    typedef PolymorphicDelegateProvider< ProviderImpl<PropertyTag, ValueT, SINGLE_VALUE_PROPERTY, SpaceType>, ProvidedValueType() > Delegate;
    
};

/**
 * Partial specialization which implement providers classes which provide values in points describe by mesh,
 * and don't use interpolation.
 */
template <typename PropertyTag, typename ValueT, typename SpaceType>
struct ProviderImpl<PropertyTag, ValueT, FIELD_PROPERTY, SpaceType>: public OnMeshProvider<ValueT, SpaceType> {

    ///Type of provided value.
    typedef typename OnMeshProvider<ValueT, SpaceType>::ProvidedValueType ProvidedValueType;
    
    /*
     * Template for implementation of field provider class which holds vector of values and mesh inside.
     * @tparam MeshType type of mesh which is used for calculation and which describe places of data points
     */
    /*template <typename MeshType>
    struct WithValue: public ProviderImpl<PropertyTag, ValueT, SINGLE_VALUE_PROPERTY, SpaceType> {
        
        typedef ProviderImpl<PropertyTag, ValueT, FIELD_PROPERTY, SpaceType>::ProvidedValueType ProvidedValueType;
        
        ProvidedValueType values;
        
        MeshType mesh;
        
        ProvidedValueType& operator()() { return values; }
        
        const ProvidedValueType& operator()() const { return values; }
        
        virtual ProvidedValueType operator()(const Mesh<SpaceType>& dst_mesh, InterpolationMethod method) {
            return interpolate(mesh, values, dst_mesh, method);
        }
    };*/
    
    /**
     * Implementation of  field provider class which delegates all operator() calls to external functor.
     */
    typedef PolymorphicDelegateProvider< ProviderImpl<PropertyTag, ValueT, FIELD_PROPERTY, SpaceType>, ProvidedValueType(const Mesh<SpaceType>& dst_mesh) > Delegate;

};

/**
 * Specialization which implement provider class which provide values in points describe by mesh and use interpolation.
 */
template <typename PropertyTag, typename ValueT, typename SpaceType>
struct ProviderImpl<PropertyTag, ValueT, INTERPOLATED_FIELD_PROPERTY, SpaceType>: public OnMeshProviderWithInterpolation<ValueT, SpaceType> {

    ///Type of provided value.
    typedef typename OnMeshProviderWithInterpolation<ValueT, SpaceType>::ProvidedValueType ProvidedValueType;
    
    /**
     * Template for implementation of field provider class which holds vector of values and mesh inside.
     * operator() call plask::interpolate.
     * @tparam MeshType type of mesh which is used for calculation and which describe places of data points
     */
    template <typename MeshType>
    struct WithValue: public ProviderImpl<PropertyTag, ValueT, INTERPOLATED_FIELD_PROPERTY, SpaceType> {
        
        ///Type of provided value.
        typedef ProviderImpl<PropertyTag, ValueT, INTERPOLATED_FIELD_PROPERTY, SpaceType>::ProvidedValueType ProvidedValueType;
        
        ///Provided value. Values in points describe by this->mesh.
        ProvidedValueType values;
        
        ///Mesh which describe in which points are this->values.
        MeshType mesh;
        
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
        virtual ProvidedValueType operator()(const Mesh<SpaceType>& dst_mesh, InterpolationMethod method) {
            return interpolate(mesh, values, dst_mesh, method);
        }
    };
    
    /**
     * Implementation of  field provider class which delegates all operator() calls to external functor.
     */
    typedef PolymorphicDelegateProvider< ProviderImpl<PropertyTag, ValueT, INTERPOLATED_FIELD_PROPERTY, SpaceType>, ProvidedValueType(const Mesh<SpaceType>& dst_mesh, InterpolationMethod method) > Delegate;

};

}; //namespace plask

#endif  //PLASK__PROVIDER_H
