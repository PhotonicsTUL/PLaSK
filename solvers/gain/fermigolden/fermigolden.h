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
    enum WhichLevel: size_t {
        LEVELS_EL,
        LEVELS_HH,
        LEVELS_LH
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

        std::vector<shared_ptr<Material>> materials;///< All materials in the active region
        std::vector<double> thicknesses;            ///< Thicknesses of the layers in the active region
        std::vector<size_t> wells;                  ///< Division of the active region into separate quantum wells

        double total;                               ///< Total active region thickness [Å]
        double bottom;                              ///< Bottom spacer thickness [Å]
        double top;                                 ///< Top spacer thickness [Å]

        /**
         * Summarize active region, check for appropriateness and compute some values
         * \param solver solver
         */
        void summarize(const FermiGoldenGainSolver<GeometryType>* solver)
        {
            auto bbox = layers->getBoundingBox();
            total = bbox.upper[1] - bbox.lower[1] - bottom - top;
            materials.clear(); materials.reserve(layers->children.size());
            thicknesses.clear(); thicknesses.reserve(layers->children.size());
            for (const auto& layer: layers->children) {
                auto block = static_cast<Block<2>*>(static_cast<Translation<2>*>(layer.get())->getChild().get());
                auto material = block->singleMaterial();
                if (!material) throw plask::Exception("FermiGoldenGainSolver requires solid layers.");
                auto bbox = static_cast<GeometryObjectD<2>*>(layer.get())->getBoundingBox();
                double thck = bbox.upper[1] - bbox.lower[1];
                materials.push_back(material);
                thicknesses.push_back(thck);
            }
            if (materials.size() > 2) {
                Material* material = materials[0].get();
                double el0 = UB<LEVELS_EL>(material, solver->T0),
                       hh0 = UB<LEVELS_HH>(material, solver->T0),
                       lh0 = UB<LEVELS_LH>(material, solver->T0);
                material = materials[1].get();
                double el1 = UB<LEVELS_EL>(material, solver->T0),
                       hh1 = UB<LEVELS_HH>(material, solver->T0),
                       lh1 = UB<LEVELS_LH>(material, solver->T0);
                for (size_t i = 2; i < materials.size(); ++i) {
                    material = materials[i].get();
                    double el2 = UB<LEVELS_EL>(material, solver->T0);
                    double hh2 = UB<LEVELS_HH>(material, solver->T0);
                    double lh2 = UB<LEVELS_LH>(material, solver->T0);
                    if ((el0 < el1 && el1 > el2) || (hh0 > hh1 && hh1 < hh2) || (lh0 > lh1 && lh1 < lh2)) {
                        if (!(el0 < el1 && el1 > el2) || !(hh0 > hh1 && hh1 < hh2) || !(lh0 > lh1 && lh1 < lh2))
                            throw Exception("%1%: Quantum wells in conduction band do not coincide with wells is valence band", solver->getId());
                        wells.push_back(i-1);
                    }
                    else if (i == 2) wells.push_back(0);
                    if (el2 != el1) { el0 = el1; el1 = el2; }
                    if (hh2 != hh1) { hh0 = hh1; hh1 = hh2; }
                    if (lh2 != lh1) { lh0 = lh1; lh1 = lh2; }
                }
            }
            if (wells.back() < materials.size()-2) wells.push_back(materials.size()-1);
            // for (auto w: wells) std::cerr << w << " ";
            // std::cerr << "\n";
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

    template <WhichLevel which> static inline Tensor2<double> Meff(const Material* material, double T, double e=0.) {
        switch (which) {
            case LEVELS_EL: return material->Me(T, e);
            case LEVELS_HH: return material->Mhh(T, e);
            case LEVELS_LH: return material->Mlh(T, e);
        }
    }

    template <WhichLevel which> static inline double UB(const Material* material, double T, double e=0.) {
        switch (which) {
            case LEVELS_EL: return material->CB(T, e);
            case LEVELS_HH: return material->VB(T, e, '*', 'H');
            case LEVELS_LH: return material->VB(T, e, '*', 'L');
        }
    }

    template <WhichLevel which> 
    inline void getk2m(double& k2, double& m, const ActiveRegionInfo& region, double substra, double T, double E, size_t i) const {
        const Material* material = region.materials[i].get();
        OmpLockGuard<OmpNestLock> lockq = material->lock();
        double e; if (strained) { double latt = material->lattC(T, 'a'); e = (substra - latt) / latt; } else e = 0.;
        m = Meff<which>(material, T, e).c11;
        k2 = ((which == LEVELS_EL)? 2e-12 : -2e-12) * phys::me / (phys::hb_eV*phys::hb_J) * m * (E - UB<which>(material, T, e));
    }

    /// Compute determinant for energy levels
    template <WhichLevel which>
    double level(const ActiveRegionInfo& region, double T, double E, size_t start, size_t stop) const;

    template <WhichLevel which>
    double level(const ActiveRegionInfo& region, double T, double E) const {
        return level<which>(region, T, E, 0, region.materials.size()-1);
    }

    template <WhichLevel which>
    double level(const ActiveRegionInfo& region, double T, double E, size_t well) const {
        return level<which>(region, T, E, region.wells[well], region.wells[well+1]);
    }

    /// Find levels estimates
    template <WhichLevel which>
    void levelEstimates(std::vector<double>& levels, std::vector<double>& masses, std::vector<double>& widths,
                        const ActiveRegionInfo& region, double T, size_t qw) const;

    /// Compute concentration for electron quasi-Fermi level
    double getNc(double F, double T) const;

    /// Compute concentration for hole quasi-Fermi level
    double getNv(double F, double T) const;

  protected:

    double lifetime;                ///< Stimulated emission lifetime [ps]
    double matrixelem;              ///< Optical matrix element [m0*eV]

    double T0;                      ///< Temperature used for compiting level estimates

    double levelsep;                ///< Minimum separation between distinct levels

    bool strained;                  ///< Consider strain in QW?

    bool extern_levels;             ///< Are levels set externally?

    inline static double nm_to_eV(double wavelength) {
        return phys::h_eVc1e9 / wavelength;
    }

    /// Estimate energy levels
    void estimateLevels();

    /// Compute quasi-Fermi levels for given concentration and temperature
    std::tuple<double, double> findFermiLevels(double n, double T) const;

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

    std::vector<std::vector<double>>
        levels_el,                  ///< Approximate electron levels
        levels_hh,                  ///< Approximate heavy hole levels
        levels_lh,                  ///< Approximate light hole levels

        masses_el,                  ///< Effective masses for approximate electron levels
        masses_hh,                  ///< Effective masses for approximate heavy hole levels
        masses_lh,                  ///< Effective masses for approximate light hole levels

        widths_el,                  ///< Effective well widths for approximate electron levels
        widths_hh,                  ///< Effective well widths for approximate heavy hole levels
        widths_lh;                  ///< Effective well widths for approximate light hole levels

    bool quick_levels;              ///< Are levels computed quickly based on estimates

#ifndef NDEBUG
    double detEl(double E, size_t reg=0, size_t well=0) {
        if (well) return level<LEVELS_EL>(regions[reg], T0, E, well-1);
        else return level<LEVELS_EL>(regions[reg], T0, E);
    }
    double detHh(double E, size_t reg=0, size_t well=0) {
        if (well) return level<LEVELS_HH>(regions[reg], T0, E, well-1);
        else return level<LEVELS_HH>(regions[reg], T0, E);
    }
    double detLh(double E, size_t reg=0, size_t well=0) {
        if (well) return level<LEVELS_LH>(regions[reg], T0, E, well-1);
        else return level<LEVELS_LH>(regions[reg], T0, E);
    }
#endif

    double getLifeTime() const { return lifetime; }
    void setLifeTime(double iLifeTime)  { lifetime = iLifeTime; }

    double getMatrixElem() const { return matrixelem; }
    void setMatrixElem(double iMatrixElem)  { matrixelem = iMatrixElem; }

    /**
     * Reg gain spectrum object for future use
     */
    GainSpectrum<GeometryType> getGainSpectrum(const Vec<2>& point);

    friend struct GainSpectrum<GeometryType>;
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
