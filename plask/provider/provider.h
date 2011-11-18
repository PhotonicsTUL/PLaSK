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
 * Subclasses should have typedef for provided value type:
 * typedef ... ProvidedValueType;
 */
struct ProviderBase {
    
    struct Listener {
        ///called when value changed
        virtual void onChange() = 0;
        
        /**
         * Called just before disconnect.
         * @param from_where provider from which listener is disconnected
         */
        virtual void onDisconnect(ProviderBase* from_where) {}
    };
    
    std::set<Listener*> listeners;
    
    ~ProviderBase() {
        for (typename std::set<Listener*>::iterator i = listeners.begin(); i != listeners.end(); ++i)
            (*i)->onDisconnect(this);
    }
    
    void add(Listener* litener) {
        listeners.insert(litener);
    }
    
    void remove(Listener* litener) {
        litener->onDisconnect(this);
        listeners.erase(litener);
    }
    
    /**
     * Call onChange for all receivers.
     * Should be call recalculation of value represented by provider.
     */
    void fireChanged() {
        for (typename std::set<Listener*>::iterator i = listeners.begin(); i != listeners.end(); ++i)
            (*i)->onChange();
    }
    
};

/**
 * Base class for all recivers.
 * Implement listener (observer) pattern (is listener for providers).
 * @tparam ProviderT type of provider
 */
template <typename ProviderT>
struct ReceiverBase: public ProviderBase::Listener {
    
    ProviderT* provider;
    
    ///true only if data provides by provider was changed after recent value getting
    bool changed;

    ReceiverBase(): changed(true) {}
    
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
     * Get value from provider.
     * @return value from provider
     * @throw NoProvider when provider is not available
     */
    template<typename ...Args>
    decltype((*provider)(std::forward<Args>(params)...))
    //typename ProviderT::ProvidedValueType
    operator()(Args&&... params) throw (NoProvider) {
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

enum PropertyType {
    SINGLE_VALUE_PROPERTY = 0,
    FIELD_PROPERTY = 1,
    INTERPOLATED_FIELD_PROPERTY = 2
};

template <typename PropertyTag, typename ValueType, PropertyType propertyType>
struct ProviderImpl {};

template <typename PropertyTag>
struct Provider: ProviderImpl<PropertyTag, typename PropertyTag::ValueType, PropertyTag::propertyType> {
    
    //delegate all constructors to parent class
    template<typename ...Args>
    Provider(Args&&... params)
    : ProviderImpl<PropertyTag, typename PropertyTag::ValueType, PropertyTag::propertyType>(std::forward<Args>(params)...) {
    };
    
};

template <typename PropertyTag>
struct Reciver: public ReceiverBase< Provider<PropertyTag> > {
    
    //delegate all constructors to parent class
    /*template<typename ...Args>
    Reciver(Args&&... params)
    : ReceiverBase< Provider<PropertyTag> >(std::forward<Args>(params)...) {
    };*/
    
};

/**
 * Template for base class for all providers which provide one value, typically one double.
 */
template <typename PropertyTag, typename ValueT>
struct ProviderImpl<PropertyTag, ValueT, SINGLE_VALUE_PROPERTY>: public ProviderBase {
    
    //typedef ValueT ProvidedValueType;
    
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
