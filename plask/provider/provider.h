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
 * Implement listener pattern.
 * @tparam 
 */
template <typename ReceiverT>
struct Provider {
    
    std::set<ReceiverT*> receivers;
    
    ~Provider() {
        for (typename std::set<ReceiverT*>::iterator i = receivers.begin(); i != receivers.end(); ++i)
            i->provider = 0;
    }
    
    void add(ReceiverT* receiver) {
        receiver.provider = this;
        receivers.insert(receiver);
    }
    
    void remove(ReceiverT* receiver) {
        receiver.provider = 0;
        receivers.remove(receiver);
    }
    
    /**
     * Call onChange for all receivers.
     * Should be call recalculation of value represented by provider.
     */
    void fireChanged() {
        for (typename std::set<ReceiverT*>::iterator i = receivers.begin(); i != receivers.end(); ++i)
            i->onChange();
    }
    
};

template <typename ProviderT>
struct Receiver {
    
    ProviderT* provider;
    
    ///true only if data provides by provider was changed after recent value getting
    bool changed;

    Receiver(): changed(true) {}
    
    ~Receiver() {
        setProvider(0);
    }
    
    void setProvider(ProviderT* provider) {
        if (this->provider == provider) return;
        if (this->provider) provider.remove(this);
        if (provider) provider.add(this);
        onChange();
    }
    
    void onChange() {
        changed = true;
        //TODO callback?
    }
    
    ///@throw NoProvider when provider is not available
    void ensureHasProvider() throw (NoProvider) {
        if (!provider) throw NoProvider();	//TODO some name, maybe Provider should have virtual name or name field?
    }
    
protected:
    
    /**
     * Check if value can be read and throw exception if not.
     * Set changed flag to false.
     * 
     * Subclass can call this just before reading value.
     */
    void beforeGetValue() throw (NoProvider) {
        ensureHasProvider();
        changed = false;
    }
    
};

template <typename ValueT> struct ValueReceiver;

/**
 * Template for base class for all providers which provide one value, typically one double.
 */
template <typename ValueT>
struct ValueProvider: public Provider< ValueReceiver<ValueT> > {
    
    typedef ValueT ValueType;
    
    ValueT value;
    
    ValueT& operator()() { return value; }
    
    const ValueT& operator()() const { return value; }
    
    operator ValueT& () { return value; }
    
    operator const ValueT& () const { return value; }
    
};

template <typename ValueT>
struct ValueReceiver: public Receiver< ValueProvider<ValueT> > {
    
    /**
     * Get value from provider.
     * @return value from provider
     * @throw NoProvider when provider is not available
     */
    ValueT operator()() const throw (NoProvider) {
        Receiver< ValueProvider<ValueT> >::beforeGetValue();
        return Receiver< ValueProvider<ValueT> >::provider->value;
    }
    
};

template <typename ValueT> struct OnMeshInterpolatedReceiver;

/**
 * Template for base class for all providers which provide values in points describe by mesh,
 * use interpolation, and has vector of data.
 */
template <typename ModuleType, typename ValueT>
struct OnMeshInterpolatedProvider: public Provider< OnMeshInterpolatedReceiver<ValueT> > {
    
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
struct OnMeshInterpolatedReceiver: public Receiver<OnMeshInterpolatedProviderT> {
    
    /**
     * Get value from provider.
     * @return value from provider
     * @throw NoProvider when provider is not available
     */
    typename OnMeshInterpolatedProviderT::ValueConstVecPtr operator()(Mesh& mesh, InterpolationMethod method) const throw (NoProvider) {
        Receiver<OnMeshInterpolatedProviderT>::beforeGetValue();
        return (*Receiver<OnMeshInterpolatedProviderT>::provider)(mesh, method);
    }
    
};



}; //namespace plask

#endif  //PLASK__PROVIDER_H
