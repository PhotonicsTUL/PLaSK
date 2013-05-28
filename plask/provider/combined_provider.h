#ifndef PLASK__COMBINATE_PROVIDERS_H
#define PLASK__COMBINATE_PROVIDERS_H

#include <set>
#include <boost/iterator/indirect_iterator.hpp>

#include "providerfor.h"

/** @file
This file contains templates and base classes for providers which combines (for example: sum) values from other providers.
*/


namespace plask {

/**
 * Template of base class of combine provider.
 *
 * Subclass should define operator() which should combine values from providers
 * (which are available by begin() and end() iterators).
 */
template <typename BaseProviderClass>
class CombinedProviderBase: public BaseProviderClass {

    /// Set of private providers which should be deleted by this.
    std::set<BaseProviderClass*> private_providers;

    void onChange(Provider& which, bool isDeleted) {
        if (isDeleted) {
            private_providers.erase(static_cast<BaseProviderClass*>(&which));
            providers.erase(static_cast<BaseProviderClass*>(&which));
        }
        this->fireChanged();
    }

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

    /**
     * Append new provider to set of the held providers.
     * @param to_add provider to append, can't be @c nullptr
     * @param providerIsPrivate @c true only if @p provider is private for this and will be deleted by destructor of this
     */
    void add(BaseProviderClass* to_add, bool providerIsPrivate = false) {
        providers.insert(to_add);
        if (providerIsPrivate) private_providers.insert(to_add);
        to_add->changed.connect(boost::bind(&CombinedProviderBase::onChange, this, _1, _2));
        this->fireChanged();
    }

    /**
     * Append new provider to the set of the held providers.
     * @param to_add provider to append, can't be @c nullptr, will be deleted by destructor of this
     */
    void add(std::unique_ptr<BaseProviderClass>&& to_add) {
        add(to_add->release(), true);
    }

    /**
     * Remove specified provider from the set of the held providers.
     * @param to_remove provider to remove, it will be deleted if it is private
     */
    void remove(BaseProviderClass* to_remove) {
        to_remove->changed.disconnect(boost::bind(&CombinedProviderBase::onChange, this, _1, _2));
        if (private_providers.erase(to_remove) > 0) delete to_remove;
        providers.erase(to_remove);
    }

    /**
     * Remove all providers from the set of the held providers.
     * Delete private providers.
     */
    void clear() {
        while (!providers.empty()) remove(*providers.begin());
    }

    /// Delete all private providers.
    ~CombinedProviderBase() {
        clear();
    }

    /**
     * Check if the providers set of this is empty.
     * @return @c true if this not contains any provider
     */
    bool empty() const { return providers.empty(); }

    /**
     * Get number of providers in the set.
     * @return number of providers
     */
    std::size_t size() const { return providers.size(); }

    /**
     * Throw exception if the providers set of this is empty.
     */
    void ensureHasProviders() const {
        if (providers.empty())
            throw Exception("Combine \"%1%\" provider has empty set of providers but some are required.", this->name());
    }

};


/**
 * Template of base class of sum provider for providers with interpolation.
 */
template <typename, typename, typename> struct FieldSumProviderImpl;

template <typename PropertyT, typename SpaceT, typename... ExtraArgs>
struct FieldSumProviderImpl<PropertyT, SpaceT, VariadicTemplateTypesHolder<ExtraArgs...>>: public CombinedProviderBase<ProviderFor<PropertyT, SpaceT>> {

    typedef typename ProviderFor<PropertyT,SpaceT>::ProvidedType ProvidedType;

    virtual ProvidedType operator()(const MeshD<SpaceT::DIM>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method=DEFAULT_INTERPOLATION) const {
        this->ensureHasProviders();
        auto p = this->begin();
        ProvidedType result = (*p)(dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
        ++p;
        if (p == this->end()) return result;    // has one element
        auto rwresult = result.claim();    // ensure has own memory
        do {
            rwresult += (*p)(dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
            ++p;
        } while (p != this->end());
        return rwresult;
    }
};

/**
 * Template of class of sum provider for providers with interpolation.
 */
template <typename PropertyT, typename SpaceT>
struct FieldSumProvider: public FieldSumProviderImpl<PropertyT, SpaceT, typename PropertyT::ExtraParams> {};


}   // namespace plask

#endif // COMBINATE_PROVIDERS_H
