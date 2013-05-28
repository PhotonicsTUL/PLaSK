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
 * Subclass should define operator() which should combine values from providers (which are available by begin() and end() iterators).
 */
template <typename BaseProviderClass>
class CombinedProviderBase: public BaseProviderClass, public Receiver<BaseProviderClass> {

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

    /// Receiver implementation, call fireChanged()
    virtual void onChange() { BaseProviderClass::fireChanged(); }

    /// Receiver implementation, delete from_where if it is private, remove it from providers sets and call fireChanged()
    virtual void onDisconnect(BaseProviderClass* from_where) {
        if (private_providers.find(from_where) != private_providers.end()) {
            delete from_where;
            private_providers.erase(from_where);
        }
        providers.erase(from_where);
        BaseProviderClass::fireChanged();
    }

    /**
     * Append new provider to set of the held providers.
     * @param to_add provider to append, can't be @c nullptr
     * @param providerIsPrivate @c true only if @p provider is private for this and will be deleted by destructor of this
     */
    void add(BaseProviderClass* to_add, bool providerIsPrivate = false) {
        providers.insert(to_add);
        if (providerIsPrivate) private_providers.insert(to_add);
        to_add->add(*this);
        BaseProviderClass::fireChanged();
    }

    /**
     * Append new provider to the set of the held providers.
     * @param to_add provider to append, can't be @c nullptr,  will be deleted by destructor of this
     */
    void add(std::unique_ptr<BaseProviderClass>&& to_add) {
        add(to_add->release(), true);
    }

    /**
     * Remove specified provider from the set of the held providers.
     * @param to_remove provider to remove, will be delete if it is private
     */
    void remove(BaseProviderClass* to_remove) {
        //to_remove->disconnect(this);    // onDisconnect callback does the rest
    }

    /**
     * Remove all providers from the set of the held providers
     */
    void clear() {
        for (auto p: providers) remove(p);
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
        //if (providers.empty())
        //    throw Exception("Combine \"%1%\" provider has empty set of providers but some are required.", BaseProviderClass::NAME);
    }

};


/**
 * Template of base class of sum provider for providers with interpolation.
 */
template <typename, typename, typename> struct FieldSumProviderImpl;

template <typename PropertyT, typename SpaceT, typename... ExtraArgs>
struct FieldSumProviderImpl<PropertyT,SpaceT,VariadicTemplateTypesHolder<ExtraArgs...>>: public CombinedProviderBase<ProviderFor<PropertyT, SpaceT>> {

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
