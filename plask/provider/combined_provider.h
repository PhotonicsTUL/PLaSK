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
template <typename BaseProviderT>
struct CombinedProviderBase: public BaseProviderT {

    typedef BaseProviderT BaseType;

  private:

    /// Set of private providers which should be deleted by this.
    std::set<BaseProviderT*> private_providers;

    void onChange(Provider& which, bool isDeleted) {
        if (isDeleted) {
            private_providers.erase(static_cast<BaseProviderT*>(&which));
            providers.erase(static_cast<BaseProviderT*>(&which));
        }
        this->fireChanged();
    }

  protected:
    /// Set of providers which values are combinating.
    std::set<BaseProviderT*> providers;

  public:

    /// Type of iterator over BaseProviderT
    typedef boost::indirect_iterator<typename std::set<BaseProviderT*>::iterator> iterator;

    /// Type of const iterator over BaseProviderT
    typedef boost::indirect_iterator<typename std::set<BaseProviderT*>::const_iterator> const_iterator;

    /// @return begin iterator over BaseProviderT
    iterator begin() { return providers.begin(); }

    /// @return past-the-end iterator over BaseProviderT
    iterator end() { return providers.end(); }

    /// @return const begin iterator over BaseProviderT
    const_iterator begin() const { return providers.begin(); }

    /// @return const past-the-end iterator over BaseProviderT
    const_iterator end() const { return providers.end(); }

    /**
     * Append new provider to set of the held providers.
     * @param to_add provider to append, can't be @c nullptr
     * @param providerIsPrivate @c true only if @p provider is private for this and will be deleted by destructor of this
     */
    void add(BaseProviderT* to_add, bool providerIsPrivate = false) {
        providers.insert(to_add);
        if (providerIsPrivate) private_providers.insert(to_add);
        to_add->changed.connect(boost::bind(&CombinedProviderBase::onChange, this, _1, _2));
        this->fireChanged();
    }

    /**
     * Append new provider to the set of the held providers.
     * @param to_add provider to append, can't be @c nullptr, will be deleted by destructor of this
     */
    void add(std::unique_ptr<BaseProviderT>&& to_add) {
        add(to_add->release(), true);
    }

    /**
     * Remove specified provider from the set of the held providers.
     * @param to_remove provider to remove, it will be deleted if it is private
     */
    void remove(BaseProviderT* to_remove) {
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
            throw Exception("Combined %1% provider has no components", this->name());
    }

};


/**
 * Template of base class of sum provider for providers with interpolation.
 */
template <typename, typename, typename> struct FieldSumProviderImpl;

template <typename PropertyT, typename SpaceT, typename... ExtraArgs>
struct FieldSumProviderImpl<PropertyT, SpaceT, VariadicTemplateTypesHolder<ExtraArgs...>>: public CombinedProviderBase<ProviderFor<PropertyT, SpaceT>> {

    typedef typename ProviderFor<PropertyT, SpaceT>::ProvidedType ProvidedType;
    typedef typename ProviderFor<PropertyT, SpaceT>::ValueType ValueType;

    struct SumLazyDataImpl: public LazyDataImpl<ValueType> {

        std::vector<LazyData<ValueType>> to_sum;

        std::size_t _size;

        SumLazyDataImpl(std::vector<LazyData<ValueType>>&& to_sum, std::size_t size)
            : to_sum(std::move(to_sum)), _size(size) {}

        ValueType at(std::size_t index) const override {
            ValueType sum = to_sum[0][index];
            for (std::size_t i = 1; i < to_sum.size(); ++i)
                sum += to_sum[i][index];
            return sum;
        }

        std::size_t size() const override {
            return _size;
        }

    };

    virtual ProvidedType operator()(shared_ptr<const MeshD<SpaceT::DIM>> dst_mesh, ExtraArgs... extra_args, InterpolationMethod method=INTERPOLATION_DEFAULT) const override {
        this->ensureHasProviders();
        std::vector<LazyData<ValueType>> providers;
        auto p = this->begin();
        providers.push_back((*p)(dst_mesh, std::forward<ExtraArgs>(extra_args)..., method));
        ++p;
        if (p == this->end()) return std::move(providers.front());    // has one element
        std::size_t size = providers.front().size();
        do {
            std::size_t last_size = providers.back().size();
            if (size != last_size)
                throw DataError("Data sources sizes differ ([%1%] - [%2])", size, last_size);
            providers.push_back((*p)(dst_mesh, std::forward<ExtraArgs>(extra_args)..., method));
            ++p;
        } while (p != this->end());
        return new SumLazyDataImpl(std::move(providers), size);
    }
};

/**
 * Template of class of sum provider for providers with interpolation.
 */
template <typename PropertyT, typename SpaceT>
struct FieldSumProvider: public FieldSumProviderImpl<PropertyT, SpaceT, typename PropertyT::ExtraParams> {};


}   // namespace plask

#endif // COMBINATE_PROVIDERS_H
