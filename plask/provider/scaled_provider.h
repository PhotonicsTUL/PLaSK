#ifndef PLASK__MULTIPLIED_PROVIDERS_H
#define PLASK__MULTIPLIED_PROVIDERS_H

#include "providerfor.h"

/** @file
This file contains templates for provider that scales source by some value
*/


namespace plask {

/**
 * Template of base of scaled provider.
 */
template <typename DstProviderT, typename SrcProviderT, typename ScaleT=double>
struct ScaledProviderBase: public DstProviderT {

    typedef SrcProviderT SourceType;
    typedef DstProviderT DestinationType;
    typedef ScaleT ScaleType;

  protected:
    /// Source provider
    SrcProviderT* source;

  private:
        /// True if the source should be deleted by this
    bool priv;

    void onChange(Provider& which, bool isDeleted) {
        if (isDeleted) {
            source = nullptr;
        }
        this->fireChanged();
    }

  public:

    /// Scale for the provider value
    ScaleT scale;

    /**
     * Create the provider
     * \param scale initial scale
     */
    ScaledProviderBase(ScaleT scale=1.): source(nullptr), priv(false), scale(scale) {}

    /**
     * Set source provider
     * \param src source provider
     * \param prv \c true only if \p src is private and should be deleted by destructor of this
     */
    void set(SrcProviderT* src, bool prv=false) {
        if (priv) delete source;
        source = src;
        priv = prv;
        if (src) src->changed.connect(boost::bind(&ScaledProviderBase::onChange, this, _1, _2));
        this->fireChanged();
    }

    /**
     * Set source provider
     * \param src source provider
     */
    void set(std::unique_ptr<SrcProviderT>&& src) {
        set(src->release(), true);
    }

    /**
     * Reset source provider
     */
    void reset() {
        if (source) source->changed.disconnect(boost::bind(&ScaledProviderBase::onChange, this, _1, _2));
        if (priv) delete source;
        source = nullptr;
        this->fireChanged();
    }

    /**
     * Obtain source provider
     * \return source provider
     */
    SrcProviderT* getSource() const {
        return source;
    }

    ~ScaledProviderBase() {
        reset();
    }

    /**
     * Throw exception if the providers set of this is empty.
     */
    void ensureHasProvider() const {
        if (!source)
            throw Exception("Scaled %1% provider has no source", this->name());
    }

};


/**
 * Template of base class of scaled provider for providers with interpolation.
 */
template <typename, typename, PropertyType, typename, typename, typename> struct ScaledFieldProviderImpl;

template <typename DstPropertyT, typename SrcPropertyT, PropertyType propertyType, typename SpaceT, typename ScaleT, typename... ExtraArgs>
struct ScaledFieldProviderImpl<DstPropertyT, SrcPropertyT, propertyType, SpaceT, ScaleT, VariadicTemplateTypesHolder<ExtraArgs...>>:
    public ScaledProviderBase<ProviderFor<DstPropertyT,SpaceT>, ProviderFor<SrcPropertyT,SpaceT>, ScaleT> {

    ScaledFieldProviderImpl(double scale=1.): ScaledProviderBase<ProviderFor<DstPropertyT,SpaceT>, ProviderFor<SrcPropertyT,SpaceT>, ScaleT>(scale) {}

    typedef typename ProviderFor<DstPropertyT,SpaceT>::ProvidedType ProvidedType;

    virtual ProvidedType operator()(shared_ptr<const MeshD<SpaceT::DIM>> dst_mesh, ExtraArgs... extra_args, InterpolationMethod method=INTERPOLATION_DEFAULT) const {
        this->ensureHasProvider();
        return (*this->source)(dst_mesh, std::forward<ExtraArgs>(extra_args)..., method) * this->scale;
    }
};

template <typename DstPropertyT, typename SrcPropertyT, typename SpaceT, typename ScaleT, typename... ExtraArgs>
struct ScaledFieldProviderImpl<DstPropertyT, SrcPropertyT, MULTI_FIELD_PROPERTY, SpaceT, ScaleT, VariadicTemplateTypesHolder<ExtraArgs...>>:
    public ScaledProviderBase<ProviderFor<DstPropertyT,SpaceT>, ProviderFor<SrcPropertyT,SpaceT>, ScaleT> {

    ScaledFieldProviderImpl(double scale=1.): ScaledProviderBase<ProviderFor<DstPropertyT,SpaceT>, ProviderFor<SrcPropertyT,SpaceT>, ScaleT>(scale) {}

    typedef typename ProviderFor<DstPropertyT,SpaceT>::ProvidedType ProvidedType;

    virtual ProvidedType operator()(size_t n, shared_ptr<const MeshD<SpaceT::DIM>> dst_mesh, ExtraArgs... extra_args, InterpolationMethod method=INTERPOLATION_DEFAULT) const override {
        this->ensureHasProvider();
        return (*this->source)(n, dst_mesh, std::forward<ExtraArgs>(extra_args)..., method) * this->scale;
    }
    
    virtual size_t size() const {
        this->ensureHasProvider();
        return this->source->size();
    }
};

/**
 * Template of class of scaled provider for providers with interpolation.
 */
template <typename DstPropertyT, typename SrcPropertyT, typename SpaceT, typename ScaleT=double>
struct ScaledFieldProvider: public ScaledFieldProviderImpl<DstPropertyT, SrcPropertyT, DstPropertyT::propertyType, SpaceT, ScaleT, typename DstPropertyT::ExtraParams> {
    static_assert(DstPropertyT::propertyType == SrcPropertyT::propertyType, "Source and destination property types do not match");
    static_assert(std::is_same<typename DstPropertyT::ExtraParams, typename SrcPropertyT::ExtraParams>::value, "Source and destination extra arguments do not match");
    ScaledFieldProvider(double scale=1.): ScaledFieldProviderImpl<DstPropertyT, SrcPropertyT, DstPropertyT::propertyType, SpaceT, ScaleT, typename DstPropertyT::ExtraParams>(scale) {}
};


}   // namespace plask

#endif // PLASK__MULTIPLIED_PROVIDERS_H
