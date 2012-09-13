#ifndef PLASK__COMBINATE_PROVIDERS_H
#define PLASK__COMBINATE_PROVIDERS_H

#include <set>
#include "provider.h"

/** @file
This file includes templates and base classes for providers which combinates (for example: sum) values from other providers.
*/

namespace plask {

/**
 * Template of base class of combinate provider.
 *
 * Subclass should define operator() which should combine values from providers set.
 */
template <typename BaseProviderClass>
class CombinateProviderBase: public BaseProviderClass, public BaseProviderClass::Listener {
       
    /// Set of private providers which should be delete by this.
    std::set<BaseProviderClass*> private_providers;
    
protected:
    /// Set of providers which values are combinating.
    std::set<BaseProviderClass*> providers;
    
public:
    
    /// BaseProviderClass::Listener implementation, call fireChanged()
    virtual void onChange() { this->fireChanged(); }

    /// BaseProviderClass::Listener implementation, delete from_where if it is private, remove it from providers sets and call fireChanged()
    virtual void onDisconnect(Provider* from_where) {
        if (private_providers.find(from_where) != private_providers.end()) {
            delete from_where;
            private_providers.erase(from_where);
        }
        providers.erase(from_where);
        this->fireChanged();
    }
    
    /**
     * Append new provider to set of holded providers.
     * @param provider provider to append, can't be @c nullptr
     * @param providerIsPrivate @c true only if @p provider is private for this and will be deleted by destructor of this
     * @return @c *this
     */
    void connect(BaseProviderClass* to_add, bool providerIsPrivate = false) {
        providers.insert(to_add);
        if (providerIsPrivate) private_providers.insert(to_add);
        to_add.add(*this);
        this->fireChanged();
    }
    
    /**
     * Remove provider from set of holded providers.
     * @param to_remove provider to remove, will be delete if it is private
     */
    void disconnect(BaseProviderClass* to_remove) {
        to_remove.remove(to_remove);    //onDisconnect call-back do the rest
    }
    
    /// Delete all private providers.
    ~CombinateProviderBase() {
        for (auto p: providers) disconnect(p);
    }
    
    void ensureHasProviders() {
        if (providers.empty()) throw Exception("Combinate \"%1%\" provider has empty set of providers and some are required.", BaseProviderClass::NAME);
    }
    
};

/**
 * Template of base class of sum provider for providers with interpolation.
 */
template <typename BaseClass, typename ValueT, typename SpaceT, typename... ExtraArgs>
struct SumOnMeshProviderWithInterpolation: public CombinateProviderBase<BaseClass> {
    
    virtual DataVector<ValueT> operator()(const MeshD<SpaceT::DIMS>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method) const {
        this->ensureHasProviders();
        auto p = this->providers.begin();
        DataVector<ValueT> result = (*p)(dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
        ++p;
        if (p == this->providers.end()) return result;    //has one element
        result = result.claim();    //ensure has own memory
        do {
            result += (*p)(dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
            ++p;
        } while (p != this->providers.end());
        return result;
    }
};

}   // namespace plask

#endif // COMBINATE_PROVIDERS_H
