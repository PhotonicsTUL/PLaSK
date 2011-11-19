#ifndef PLASK__PROVIDER_H
#define PLASK__PROVIDER_H

#include <set>
#include <memory>
#include <vector>

#include "../exceptions.h"
#include "../mesh/mesh.h"
#include "../mesh/interpolation.h"

namespace plask {

/**
 * Template for base class for all Providers.
 * Implement listener (observer) pattern (can be observed by reciver).
 * 
 * Subclasses should have implemented operator()(...) which return provided value.
 */
struct ProviderBase {
    
    /**
     * ProviderBase listener (observer). Can react to ProviderBase changes.
     */
    struct Listener {
        ///called when value changed
        virtual void onChange() = 0;
        
        /**
         * Called just before disconnect. By default do nothing.
         * @param from_where provider from which listener is disconnected
         */
        virtual void onDisconnect(ProviderBase* from_where) {}
    };
    
    ///Set of added (registered) listeners. This provider can call methods of listeners included in this set.
    std::set<Listener*> listeners;
    
    ///Call onDisconnect for all liteners in listeners set.
    ~ProviderBase() {
        for (typename std::set<Listener*>::iterator i = listeners.begin(); i != listeners.end(); ++i)
            (*i)->onDisconnect(this);
    }
    
    /**
     * Add litener to listeners set.
     * @param listener listener to add (register)
     */
    void add(Listener* litener) {
        listeners.insert(litener);
    }
    
    /**
     * Remove (unregister) listner from listeners set.
     * @param listener listener to remove (unregister)
     */
    void remove(Listener* litener) {
        litener->onDisconnect(this);
        listeners.erase(litener);
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
 * Base class for all recivers.
 *
 * Implement listener (observer) pattern (is listener for providers).
 * Delegeta all operator() calling to provider.
 *
 * For most providers types, reciver type can be defined as: <code>ReceiverBase<ProviderClass>;</code>
 *
 * @tparam ProviderT type of provider
 */
template <typename ProviderT>
struct ReceiverBase: public ProviderBase::Listener {
    
    ProviderT* provider;
    
    ///true only if data provides by provider was changed after recent value getting
    bool changed;

    ReceiverBase(): provider(0), changed(true) {}
    
    ~ReceiverBase() {
        setProvider(0);
    }
    
    void setProvider(ProviderT* provider) {
        if (this->provider == provider) return;
        if (this->provider) this->provider->listeners.erase(this);
        if (provider) provider->add(this);
        this->provider = provider;
        onChange();
    }
    
    void setProvider(ProviderT &provider) {
		setProvider(&provider);
    }
    
    ProviderT* getProvider() { return provider; }
    
    const ProviderT* getProvider() const { return provider; }
    
    ///React on provider value changes. Set changed flag to true.
    void onChange() {
        changed = true;
        //TODO callback?
    }
    
    virtual void onDisconnect(ProviderBase* from_where) {
        if (from_where == provider) {
            provider = 0;
            onChange();
        }
    }
    
    ///@throw NoProvider when provider is not available
    void ensureHasProvider() throw (NoProvider) {
        if (!provider) throw NoProvider();	//TODO some name, maybe Provider should have virtual name or name field?
    }
    
    /**
     * Get value from provider using its operator().
     * @return value from provider
     * @throw NoProvider when provider is not available
     */
    template<typename ...Args>
    auto
    //typename ProviderT::ProvidedValueType
    operator()(Args&&... params) throw (NoProvider) -> decltype((*provider)(std::forward<Args>(params)...)) {
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
    void beforeGetValue() throw (NoProvider) {
        ensureHasProvider();
        changed = false;
    }
    
};

/**
 * Type of properies.
 */
enum PropertyType {
    SINGLE_VALUE_PROPERTY = 0,	///<Single value property.
    FIELD_PROPERTY = 1,			///<Property for field of values which can't be interpolate.
    INTERPOLATED_FIELD_PROPERTY = 2	///<Property for field of values which can be interpolate.
};	//TODO change this to empty classes(?)

/**
 * Helper class which makes easiest to define property tags class.
 *
 * Proporty tags class are used for Provider and Reciver templates specializations.,
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
typedef InterpolatedFieldProperty<double> ScalarField;

/**
 * Specializations of this class are implementations of providers for given property tag class and this tag properties.
 *
 * Don't use this class directly. Use Provider class.
 */
template <typename PropertyTag, typename ValueType, PropertyType propertyType>
struct ProviderImpl {};

/**
 * Specializations of this class are implementations of providers for given property tag.
 */
template <typename PropertyTag>
struct Provider: ProviderImpl<PropertyTag, typename PropertyTag::ValueType, PropertyTag::propertyType> {
    
    ///Delegate all constructors to parent class.
    template<typename ...Args>
    Provider(Args&&... params)
    : ProviderImpl<PropertyTag, typename PropertyTag::ValueType, PropertyTag::propertyType>(std::forward<Args>(params)...) {
    }
    
};

/**
 * Specializations of this class are implementations of reciver for given property tag.
 */
template <typename PropertyTag>
struct Reciver: public ReceiverBase< Provider<PropertyTag> > {};

/**
 * Template for base class for all providers which provide one value, typically one double.
 */
template <typename PropertyTag, typename ValueT>
struct ProviderImpl<PropertyTag, ValueT, SINGLE_VALUE_PROPERTY>: public ProviderBase {
    
    typedef ValueT ProvidedValueType;
    
    ProvidedValueType value;
    
    ProvidedValueType& operator()() { return value; }
    
    const ProvidedValueType& operator()() const { return value; }
    
};

//TODO ProviderImpl for meshes:
/*template <typename PropertyTag, typename ValueT>
struct ProviderImpl<PropertyTag, ValueT, FIELD_PROPERTY>: public ProviderBase {
    
    //typedef std::shared_ptr<std::vector<ValueT> > ProvidedValueType;
    
    ProvidedValueType& operator()() ...
    
};*/

/*struct TestProp {
    typedef double ValueType;
    static const PropertyType propertyType = SINGLE_VALUE_PROPERTY;
};

Provider<TestProp> test;
Reciver<TestProp> testr;
double x = testr();*/

template <typename ValueT> struct OnMeshInterpolatedReceiver;

/**
 * Template for base class for all providers which provide values in points describe by mesh,
 * use interpolation, and has vector of data.
 */
template <typename ModuleType, typename ValueT>
struct OnMeshInterpolatedProvider: public ProviderBase {
    
    typedef ValueT ValueType;
    
    typedef std::shared_ptr< std::vector<ValueT> > ValueVecPtr;
    
    typedef std::shared_ptr< const std::vector<ValueT> > ValueConstVecPtr;
    
    typedef ValueConstVecPtr (ModuleType::*MethodPtr)(Mesh& mesh, InterpolationMethod method);
    
    //TODO use std::function<ValueConstVecPtr(Mesh&, InterpolationMethod)> or maybe each provider should have pointer to module?
    ModuleType* module;
    MethodPtr module_value_get_method;
    
    ValueVecPtr value;
    
    OnMeshInterpolatedProvider(ModuleType* module, MethodPtr module_value_get_method)
    : module(module), module_value_get_method(module_value_get_method) {
    }
    
    ValueConstVecPtr operator()(Mesh& mesh, InterpolationMethod method) {
        return module->*module_value_get_method(mesh, method);
    }
    
};

template <typename OnMeshInterpolatedProviderT>
struct OnMeshInterpolatedReceiver: public ReceiverBase<OnMeshInterpolatedProviderT> {
    
    /**
     * Get value from provider.
     * @return value from provider
     * @throw NoProvider when provider is not available
     */
    typename OnMeshInterpolatedProviderT::ValueConstVecPtr operator()(Mesh& mesh, InterpolationMethod method) const throw (NoProvider) {
        ReceiverBase<OnMeshInterpolatedProviderT>::beforeGetValue();
        return (*ReceiverBase<OnMeshInterpolatedProviderT>::provider)(mesh, method);
    }
    
};



}; //namespace plask

#endif  //PLASK__PROVIDER_H
