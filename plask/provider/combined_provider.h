#ifndef PLASK__COMBINATE_PROVIDERS_H
#define PLASK__COMBINATE_PROVIDERS_H

#include <set>
#include "provider.h"

/** @file
This file includes templates and base classes for providers which combinates (for example: sum) values from other providers.
*/

#include <boost/iterator/indirect_iterator.hpp>

namespace plask {

/**
 * Template of base class of combine provider.
 *
 * Subclass should define operator() which should combine values from providers set (which is available by begin() and end() iterators).
 */
template <typename BaseProviderClass>
class CombinedProviderBase: public BaseProviderClass, public BaseProviderClass::Listener {

    /// Set of private providers which should be delete by this.
    std::set<BaseProviderClass*> private_providers;

protected:
    /// Set of providers which values are combinating.
    std::set<BaseProviderClass*> providers;

public:

    /// Type of iterator over BaseProviderClass
    typedef boost::indirect_iterator<typename std::set<BaseProviderClass*>::iterator> iterator;

    /// Type of const iterator over BaseProviderClass
    typedef boost::indirect_iterator<typename std::set<BaseProviderClass*>::const_iterator> const_iterator;

    /// @return begin iterator over BaseProviderClass
    iterator begin() { return providers.begin(); }

    /// @return past-the-end iterator over BaseProviderClass
    iterator end() { return providers.end(); }

    /// @return const begin iterator over BaseProviderClass
    const_iterator begin() const { return providers.begin(); }

    /// @return const past-the-end iterator over BaseProviderClass
    const_iterator end() const { return providers.end(); }

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
     * Append new provider to set of held providers.
     * @param to_add provider to append, can't be @c nullptr
     * @param providerIsPrivate @c true only if @p provider is private for this and will be deleted by destructor of this
     */
    void connect(BaseProviderClass* to_add, bool providerIsPrivate = false) {
        providers.insert(to_add);
        if (providerIsPrivate) private_providers.insert(to_add);
        to_add->add(*this);
        this->fireChanged();
    }

    /**
     * Append new provider to set of held providers.
     * @param to_add provider to append, can't be @c nullptr,  will be deleted by destructor of this
     */
    void connect(std::unique_ptr<BaseProviderClass>&& to_add) {
        connect(to_add->release(), true);
    }

    /**
     * Remove provider from set of held providers.
     * @param to_remove provider to remove, will be delete if it is private
     */
    void disconnect(BaseProviderClass* to_remove) {
        to_remove->remove(to_remove);    // onDisconnect callback does the rest
    }

    /// Delete all private providers.
    ~CombinedProviderBase() {
        for (auto p: providers) disconnect(p);
    }

    /**
     * Check if providers set of this is empty.
     * @return @c true if this not includes any provider
     */
    bool empty() const { return providers.empty(); }

    /**
     * Get number of providers in set.
     * @return number of providers
     */
    std::size_t size() const { return providers.size(); }

    /**
     * Throw exception if providers set of this is empty.
     */
    void ensureHasProviders() {
        if (providers.empty()) throw Exception("Combine \"%1%\" provider has empty set of providers but some are required.", BaseProviderClass::NAME);
    }

};

/**
 * Template of base class of sum provider for providers with interpolation.
 */
template <typename BaseClass, typename ValueT, typename SpaceT, typename... ExtraArgs>
struct SumOnMeshProviderWithInterpolation: public CombinedProviderBase<BaseClass> {

    virtual DataVector<ValueT> operator()(const MeshD<SpaceT::DIMS>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method) const {
        this->ensureHasProviders();
        auto p = this->begin();
        DataVector<ValueT> result = (*p)(dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
        ++p;
        if (p == this->end()) return result;    // has one element
        result = result.claim();    // ensure has own memory
        do {
            result += (*p)(dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
            ++p;
        } while (p != this->end());
        return result;
    }
};

}   // namespace plask

#endif // COMBINATE_PROVIDERS_H
