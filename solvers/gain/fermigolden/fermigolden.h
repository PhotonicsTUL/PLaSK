#ifndef PLASK__SOLVER__GAIN_FERMIGOLDEN_FERMIGOLDEN_H
#define PLASK__SOLVER__GAIN_FERMIGOLDEN_FERMIGOLDEN_H

#include <plask/plask.hpp>

namespace plask { namespace gain { namespace fermigolden {

template <typename GeometryT> struct GainSpectrum;

/**
 * Gain solver using Fermi Golden Rule
 */
template <typename GeometryType>
struct PLASK_SOLVER_API FermiGoldenGainSolver: public SolverWithMesh<GeometryType, RectangularMesh<1>>
{
    /// Which level to compute
    enum WhichLevel {
        LEVEL_EC,
        LEVEL_HH,
        LEVEL_LH
    };

    /// Structure containing information about each active region
    struct ActiveRegionInfo
    {
        shared_ptr<StackContainer<2>> layers;   ///< Stack containing all layers in the active region
        Vec<2> origin;                          ///< Location of the active region stack origin
        ActiveRegionInfo(Vec<2> origin): layers(make_shared<StackContainer<2>>()), origin(origin) {}

        /// Return number of layers in the active region with surrounding barriers
        size_t size() const
        {
            return layers->getChildrenCount();
        }

        /// Return material of \p n-th layer
        shared_ptr<Material> getLayerMaterial(size_t n) const
        {
            auto block = static_cast<Block<2>*>(static_cast<Translation<2>*>(layers->getChildNo(n).get())->getChild().get());
            if (auto m = block->singleMaterial()) return m;
            throw plask::Exception("FermiGoldenGainSolver requires solid layers.");
        }

        /// Return translated bounding box of \p n-th layer
        Box2D getLayerBox(size_t n) const
        {
            return static_cast<GeometryObjectD<2>*>(layers->getChildNo(n).get())->getBoundingBox() + origin;
        }

        /// Return \p true if given layer is quantum well
        bool isQW(size_t n) const
        {
            return static_cast<Translation<2>*>(layers->getChildNo(n).get())->getChild()->hasRole("QW");
        }

        /// Return bounding box of the whole active region
        Box2D getBoundingBox() const
        {
            return layers->getBoundingBox() + origin;
        }

        /// Return \p true if the point is in the active region
        bool contains(const Vec<2>& point) const {
            return getBoundingBox().contains(point);
        }

        /// Return \p true if given point is inside quantum well
        bool inQW(const Vec<2>& point) const {
            if (!contains(point)) return false;
            assert(layers->getChildForHeight(point.c1-origin.c1));
            return layers->getChildForHeight(point.c1-origin.c1)->getChild()->hasRole("QW");
        }

        std::vector<shared_ptr<Material>> materials; ///< All materials in the active region
        std::vector<double> lens;               ///< Thicknesses of the layers in the active region

        double qwtotallen;                      ///< Total quantum wells thickness [Å]
        double totallen;                        ///< Total active region thickness [Å]
        double bottomlen;                       ///< Bottom spacer thickness [Å]
        double toplen;                          ///< Top spacer thickness [Å]

        /**
         * Summarize active region, check for appropriateness and compute some values
         * \param solver solver
         */
        void summarize(const FermiGoldenGainSolver<GeometryType>* solver)
        {
            auto bbox = layers->getBoundingBox();
            totallen = 1e4 * (bbox.upper[1] - bbox.lower[1] - bottomlen - toplen);  // 1e4: µm -> Å
            size_t qwn = 0;
            qwtotallen = 0.;
            bool lastbarrier = true;
            for (const auto& layer: layers->children) {
                auto block = static_cast<Block<2>*>(static_cast<Translation<2>*>(layer.get())->getChild().get());
                auto material = block->singleMaterial();
                if (!material) throw plask::Exception("FermiGoldenGainSolver requires solid layers.");
                if (static_cast<Translation<2>*>(layer.get())->getChild()->hasRole("QW")) {
                    /*if (!materialQW)
                        materialQW = material;
                    else if (*material != *materialQW)
                        throw Exception("%1%: Multiple quantum well materials in active region.", solver->getId());*/
                    auto bbox = static_cast<GeometryObjectD<2>*>(layer.get())->getBoundingBox();
                    qwtotallen += bbox.upper[1] - bbox.lower[1];
                    if (lastbarrier) ++qwn;
                    else solver->writelog(LOG_WARNING, "Considering two adjacent quantum wells as one");
                    lastbarrier = false;
                } else { //if (static_cast<Translation<2>*>(layer.get())->getChild()->hasRole("barrier")) // TODO 4.09.2014
                    /*if (!materialBarrier)
                        materialBarrier = material;
                    else if (!is_zero(material->Me(300).c00 - materialBarrier->Me(300).c00) ||
                             !is_zero(material->Me(300).c11 - materialBarrier->Me(300).c11) ||
                             !is_zero(material->Mhh(300).c00 - materialBarrier->Mhh(300).c00) ||
                             !is_zero(material->Mhh(300).c11 - materialBarrier->Mhh(300).c11) ||
                             !is_zero(material->Mlh(300).c00 - materialBarrier->Mlh(300).c00) ||
                             !is_zero(material->Mlh(300).c11 - materialBarrier->Mlh(300).c11) ||
                             !is_zero(material->CB(300) - materialBarrier->CB(300)) ||
                             !is_zero(material->VB(300) - materialBarrier->VB(300)))
                        throw Exception("%1%: Multiple barrier materials around active region.", solver->getId());*/
                    lastbarrier = true;
                } // TODO something must be added here because of spacers placed next to external barriers
            }
            qwtotallen *= 1e3; // µm -> nm
        }
    };

    shared_ptr<Material> materialSubstrate;   ///< Substrate material

    /// List of active regions
    std::vector<ActiveRegionInfo> regions;

    /// Receiver for temperature.
    ReceiverFor<Temperature,GeometryType> inTemperature;

    /// Receiver for carriers concentration in the active region
    ReceiverFor<CarriersConcentration,GeometryType> inCarriersConcentration;

    /// Provider for gain distribution
    typename ProviderFor<Gain,GeometryType>::Delegate outGain;

    FermiGoldenGainSolver(const std::string& name="");

    virtual ~FermiGoldenGainSolver();

    virtual std::string getClassName() const;

    virtual void loadConfiguration(plask::XMLReader& reader, plask::Manager& manager);

  private:

    template <WhichLevel level> Tensor2<double> Meff(const Material* material, double T, double e) {
        switch (level) {
            case LEVEL_EC: return material->Me(T, e);
            case LEVEL_HH: return material->Mhh(T, e);
            case LEVEL_LH: return material->Mlh(T, e);
        }
    }

    template <WhichLevel level> double UB(const Material* material, double T, double e) {
        switch (level) {
            case LEVEL_EC: return  material->CB(T, e);
            case LEVEL_HH: return -material->VB(T, e, '*', 'H');
            case LEVEL_LH: return -material->VB(T, e, '*', 'L');
        }
    }

    template <WhichLevel level> 
    double layerk2(size_t reg, double substra, double T, double E, size_t i) {
        const Material* material = regions[reg].materials[i].get();
        OmpLockGuard<OmpNestLock> lockq = material->lock();
        double e; if (strained) { double latt = material->lattC(T, 'a'); e = (substra - latt) / latt; } else e = 0.;
        return 2./(phys::hb_eV*phys::hb_eV) * phys::me * Meff<level>(material, T, e).c11 * (E - UB<level>(material, T, e));
    }

    /// Compute determinant for energy levels
    template <WhichLevel level>
    double level(size_t reg, double T, double E);

  protected:

    /// Estimate energy levels
    void estimateLevels();

    double lifetime;                ///< Stimulated emission lifetime [ps]
    double matrixelem;              ///< Optical matrix element [m0*eV]

    double T0;                      ///< Temperature used for compiting level estimates

    std::vector<std::vector<double>> levels_el, ///< Approximate electron levels
                                     levels_hh, ///< Approximate heavy hole levels
                                     levels_lh; ///< Approximate light hole levels

    bool strained;                  ///< Consider strain in QW?

    bool extern_levels;             ///< Are levers set externally?

    inline static double nm_to_eV(double wavelength) {
        return phys::h_eVc1e9 / wavelength;
    }

    /// Initialize the solver
    virtual void onInitialize();

    /// Invalidate the gain
    virtual void onInvalidate();

    /// Notify that gain was chaged
    void onInputChange(ReceiverBase&, ReceiverBase::ChangeReason)
    {
        outGain.fireChanged();  // the input changed, so we inform the world that everybody should get the new gain
    }

    /**
     * Detect active regions.
     * Store information about them in the \p regions field.
     */
    void detectActiveRegions();

    struct DataBase;
    struct GainData;
    struct DgdnData;

    /**
     * Compute the gain on the mesh. This method is called by gain provider.
     * \param what what to return (gain or its carriera derivative)
     * \param dst_mesh destination mesh
     * \param wavelength wavelength to compute gain for
     * \param interp interpolation method
     * \return gain distribution
     */
    const LazyData<double> getGain(Gain::EnumType what, const shared_ptr<const MeshD<2>>& dst_mesh, double wavelength, InterpolationMethod interp=INTERPOLATION_DEFAULT);

  public:

    double getLifeTime() const { return lifetime; }
    void setLifeTime(double iLifeTime)  { lifetime = iLifeTime; }

    double getMatrixElem() const { return matrixelem; }
    void setMatrixElem(double iMatrixElem)  { matrixelem = iMatrixElem; }

    friend struct GainSpectrum<GeometryType>;

    /**
     * Reg gain spectrum object for future use;
     */
    GainSpectrum<GeometryType> getGainSpectrum(const Vec<2>& point);
};


/**
 * Cached gain spectrum
 */
template <typename GeometryT>
struct GainSpectrum {

    FermiGoldenGainSolver<GeometryT>* solver; ///< Source solver
    boost::optional<Vec<2>> point;            ///< Point in which the gain is calculated

    /// Active region containg the point
    const typename FermiGoldenGainSolver<GeometryT>::ActiveRegionInfo* region;

    double T;                           ///< Temperature
    double n;                           ///< Carries concentration

    GainSpectrum(FermiGoldenGainSolver<GeometryT>* solver, const Vec<2> point): solver(solver), point(point), T(NAN), n(NAN) {
        for (const auto& reg: solver->regions) {
            if (reg.contains(point)) {
                region = &reg;
                solver->inTemperature.changedConnectMethod(this, &GainSpectrum::onTChange);
                solver->inCarriersConcentration.changedConnectMethod(this, &GainSpectrum::onNChange);
                return;
            };
        }
        throw BadInput(solver->getId(), "Point %1% does not belong to any active region", point);
    }

    GainSpectrum(FermiGoldenGainSolver<GeometryT>* solver, double T, double n, const typename FermiGoldenGainSolver<GeometryT>::ActiveRegionInfo* reg):
        solver(solver), T(T), n(n), region(reg) {}

    void onTChange(ReceiverBase&, ReceiverBase::ChangeReason) { T = NAN; }

    void onNChange(ReceiverBase&, ReceiverBase::ChangeReason) { n = NAN; }

    ~GainSpectrum() {
        solver->inTemperature.changedDisconnectMethod(this, &GainSpectrum::onTChange);
        solver->inCarriersConcentration.changedDisconnectMethod(this, &GainSpectrum::onNChange);
    }

    /**
     * Get gain at given wavelength
     * \param wavelength wavelength to get gain
     * \return gain
     */
    double getGain(double wavelength) {
        #pragma omp critical
        {
            if (isnan(T) and point) T = solver->inTemperature(make_shared<const OnePointMesh<2>>(*point))[0];
            if (isnan(n) and point) n = solver->inCarriersConcentration(make_shared<const OnePointMesh<2>>(*point))[0];
        }
//         return solver->getGainModule(wavelength, T, n, *region) // returns gain for single QW layer!
//             .Get_gain_at_n(solver->nm_to_eV(wavelength), region->qwlen); // earlier: qwtotallen
    }
};


}}} // # namespace plask::gain::fermigolden

#endif // PLASK__SOLVER__GAIN_FERMIGOLDEN_FERMIGOLDEN_H
