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
        EL = 0,
        HH = 1,
        LH = 2
    };

    struct ActiveRegionParams;

    /// Data for energy level
    struct Level {
        double E;                   ///< Level energy
        Tensor2<double> M;          ///< Representative well index
        double thickness;           ///< Cumulated wells thickness

        Level(double E, const Tensor2<double>& M): E(E), M(M) {}

        Level(double E, const Tensor2<double>& M, double t): E(E), M(M), thickness(t) {}

        // TODO Better effective thickness computation (based on wavefunction)?
        Level(double E, const Tensor2<double>& M, WhichLevel which, const ActiveRegionParams& params);
    };

    /// Structure containing information about each active region
    struct ActiveRegionInfo {
        shared_ptr<StackContainer<2>> layers;   ///< Stack containing all layers in the active region
        Vec<2> origin;                          ///< Location of the active region stack origin

        ActiveRegionInfo(Vec<2> origin): layers(plask::make_shared<StackContainer<2>>()), origin(origin) {}

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

        /// Return bounding box of the whole active region
        Box2D getBoundingBox() const
        {
            return layers->getBoundingBox() + origin;
        }

        /// Return \p true if the point is in the active region
        bool contains(const Vec<2>& point) const {
            return getBoundingBox().contains(point);
        }

        /// Return \p true if specified layer contains the point in the active region
        bool layerContains(const Vec<2>& point, size_t n) const {
            return getLayerBox(n).contains(point);
        }

        /// Return layer containing a specified point or maxint.
        size_t layerContaining(const Vec<2>& point) const {
            for (size_t i = 0; i != layers->children.size(); ++i)
                if (layerContains(point, i)) return i;
            return std::numeric_limits<size_t>::max();
        }

        bool isQW(size_t n) const {
            return isqw[n];
        }

        bool isQW(const Vec<2>& point) const {
            return isqw[layerContaining(point)];
        }

        std::vector<shared_ptr<Material>> materials;///< All materials in the active region
        std::vector<double> thicknesses;            ///< Thicknesses of the layers in the active region
        std::vector<size_t> wells;                  ///< Division of the active region into separate quantum wells
        std::vector<bool> isqw;                     ///< Flags denoting if specified layer is qw

        double total;                               ///< Total active region thickness [µm]
        double totalqw;                             ///< Total accepted quantum wells thickness [µm]
        double bottom;                              ///< Bottom spacer thickness [µm]
        double top;                                 ///< Top spacer thickness [µm]

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
                if (!material) throw plask::Exception("%s: Active region can consist only of solid layers", solver->getId());
                auto bbox = static_cast<GeometryObjectD<2>*>(layer.get())->getBoundingBox();
                double thck = bbox.upper[1] - bbox.lower[1];
                materials.push_back(material);
                thicknesses.push_back(thck);
            }
            if (materials.size() > 2) {
                double mel = std::numeric_limits<double>::max(), mhh = std::numeric_limits<double>::lowest(), mlh = std::numeric_limits<double>::lowest();
                Material* material = materials[0].get();
                double el0 = material->CB(solver->T0, 0., 'G'),
                       hh0 = material->VB(solver->T0, 0., 'G',  'H'),
                       lh0 = material->VB(solver->T0, 0., 'G',  'L');
                material = materials[1].get();
                double el1 = material->CB(solver->T0, 0., 'G'),
                       hh1 = material->VB(solver->T0, 0., 'G',  'H'),
                       lh1 = material->VB(solver->T0, 0., 'G',  'L');
                for (size_t i = 2; i < materials.size(); ++i) {
                    material = materials[i].get();
                    double el2 = material->CB(solver->T0, 0., 'G');
                    double hh2 = material->VB(solver->T0, 0., 'G',  'H');
                    double lh2 = material->VB(solver->T0, 0., 'G',  'L');
                    if ((el0 < el1 && el1 > el2) || (hh0 > hh1 && hh1 < hh2) || (lh0 > lh1 && lh1 < lh2)) {
                        if (!(el0 < el1 && el1 > el2) || !(hh0 > hh1 && hh1 < hh2) || !(lh0 > lh1 && lh1 < lh2))
                            throw Exception("%1%: Quantum wells in conduction band do not coincide with wells is valence band", solver->getId());
                        wells.push_back(i-1);
                        if (el1 < mel) mel = el1;
                        if (hh1 < mhh) mhh = hh1;
                        if (lh1 < mhh) mlh = lh1;
                    } else if (i == 2)
                        wells.push_back(0);
                    if (el2 != el1) { el0 = el1; el1 = el2; }
                    if (hh2 != hh1) { hh0 = hh1; hh1 = hh2; }
                    if (lh2 != lh1) { lh0 = lh1; lh1 = lh2; }
                }
                if (el1 < mel) mel = el1;
                if (hh1 < mhh) mhh = hh1;
                if (lh1 < mhh) mlh = lh1;
                isqw.resize(materials.size());
                isqw[0] = isqw[materials.size()-1] = false;
                totalqw = 0.;
                for (size_t i = 1; i < materials.size()-1; ++i) {
                    material = materials[i].get();
                    if (material->CB(solver->T0, 0., 'G') < mel && 
                        (material->VB(solver->T0, 0., 'G',  'H') > mhh || material->VB(solver->T0, 0., 'G',  'L') > mlh)) {
                        isqw[i] = true;
                        totalqw += thicknesses[i];
                    }
                }
            } else {
                isqw.assign(materials.size(), true);
            }
            if (wells.back() < materials.size()-2) wells.push_back(materials.size()-1);
            // for (auto w: wells) std::cerr << w << " ";
            // std::cerr << "\n";
        }
    };

    /// Structure containing active region data for current used
    struct ActiveRegionParams {
        const ActiveRegionInfo& region;
        std::vector<double> U[3];          ///< Band levels
        std::vector<Tensor2<double>> M[3]; ///< Effective masses

        std::vector<Level> levels[3];      ///< Approximate electron, heavy and light hole levels
        size_t nhh,                        ///< Number of electron–heavy hole pairs important for gain
               nlh;                        ///< Number of electron–light hole pairs important for gain

        ActiveRegionParams(const FermiGoldenGainSolver* solver, const ActiveRegionInfo& region, double T): region(region) {
            double n = region.materials.size();
            U[EL].reserve(n); U[HH].reserve(n); U[LH].reserve(n);
            M[EL].reserve(n); M[HH].reserve(n); M[LH].reserve(n);
            double substra = solver->strained? solver->materialSubstrate->lattC(T, 'a') : 0.;
            for (auto material: region.materials) {
                OmpLockGuard<OmpNestLock> lockq = material->lock();
                double e; if (solver->strained) { double latt = material->lattC(T, 'a'); e = (substra - latt) / latt; } else e = 0.;
                U[EL].push_back(material->CB(T, e, 'G'));
                U[HH].push_back(material->VB(T, e, 'G', 'H'));
                U[LH].push_back(material->VB(T, e, 'G', 'L'));
                M[EL].push_back(material->Me(T, e));
                M[HH].push_back(material->Mhh(T, e));
                M[LH].push_back(material->Mlh(T, e));
            }
        }

        ActiveRegionParams(const FermiGoldenGainSolver* solver, const ActiveRegionInfo& region):
            ActiveRegionParams(solver, region, solver->T0) {}

        explicit ActiveRegionParams(const FermiGoldenGainSolver* solver, const ActiveRegionParams& ref, double T):
            ActiveRegionParams(solver, ref.region, T) {
                nhh = ref.nhh;
                nlh = ref.nlh;
                for (size_t which = 0; which < 3; ++which) {
                    double shift = delta(WhichLevel(which), ref);
                    levels[which].reserve(ref.levels[which].size());
                    for (Level level: ref.levels[which])
                        levels[which].emplace_back(level.E+shift, level.M, level.thickness);
                }
        }

        double sideU(WhichLevel which) const {
            return 0.5 * (U[which][0] + U[which][U[which].size()-1]);
        }

        Tensor2<double> sideM(WhichLevel which) const {
            return 0.5 * (M[which][0] + M[which][M[which].size()-1]);
        }

        double delta(WhichLevel which, const ActiveRegionParams& ref) const {
            assert(U[which].size() == ref.U[which].size());
            double delta = 0;
            for (size_t i = 0; i < U[which].size(); ++i) {
                delta += U[which][i] - ref.U[which][i];
            }
            return delta / U[which].size();
        }
    };

    /// Substrate material
    shared_ptr<Material> materialSubstrate;

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

    /// Compute determinant for energy levels
    double level(WhichLevel which, double E, const ActiveRegionParams& params, size_t start, size_t stop) const;

    double level(WhichLevel which, double E, const ActiveRegionParams& params) const {
        return level(which, E, params, 0, params.region.materials.size()-1);
    }

    double level(WhichLevel which, double E, const ActiveRegionParams& params, size_t well) const {
        return level(which, E, params, params.region.wells[well], params.region.wells[well+1]);
    }

    /// Find levels estimates
    void estimateWellLevels(WhichLevel which, ActiveRegionParams& params, size_t qw) const;

    /// Find levels estimates
    void estimateAboveLevels(WhichLevel which, ActiveRegionParams& params) const;

#ifndef NDEBUG
  public:
#endif
    /// Compute concentration for electron quasi-Fermi level
    double getN(double F, double T, const ActiveRegionParams& params) const;

    /// Compute concentration for hole quasi-Fermi level
    double getP(double F, double T, const ActiveRegionParams& params) const;

  protected:

    double lifetime;                ///< Stimulated emission lifetime [ps]
    double matrixelem;              ///< Optical matrix element [m0*eV]

    double polarization;            ///< Polarization state ($e_{zz}$)

    double T0;                      ///< Temperature used for compiting level estimates

  protected:

    double levelsep;                ///< Minimum separation between distinct levels

    bool strained;                  ///< Consider strain in QW?

    inline static double nm_to_eV(double wavelength) {
        return phys::h_eVc1e9 / wavelength;
    }

    /// Estimate energy levels
    void estimateLevels();

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

    std::vector<ActiveRegionParams> params0; ///< Reference active region params

    bool quick_levels;                          ///< Are levels computed quickly based on estimates

#ifndef NDEBUG
    double detEl(double E, const ActiveRegionParams& params, size_t well=0) {
        if (well) return level(EL, E, params, well-1);
        else return level(EL, E, params);
    }
    double detHh(double E, const ActiveRegionParams& params, size_t well=0) {
        if (well) return level(HH, E, params, well-1);
        else return level(HH, E, params);
    }
    double detLh(double E, const ActiveRegionParams& params, size_t well=0) {
        if (well) return level(LH, E, params, well-1);
        else return level(LH, E, params);
    }
#endif

    /// Compute quasi-Fermi levels for given concentration and temperature
    void findFermiLevels(double& Fc, double& Fv, double n, double T, const ActiveRegionParams& params) const;

    /// Find gain before convolution
    double getGain0(double hw, double Fc, double Fv, double T, double nr, const ActiveRegionParams& params) const;

    double getT0() const { return T0; }
    void setT0(double T) { T0 = T; this->invalidate(); }

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
            if (isnan(T) and point) T = solver->inTemperature(plask::make_shared<const OnePointMesh<2>>(*point))[0];
            if (isnan(n) and point) n = solver->inCarriersConcentration(plask::make_shared<const OnePointMesh<2>>(*point))[0];
        }
//         return solver->getGainModule(wavelength, T, n, *region) // returns gain for single QW layer!
//             .Get_gain_at_n(solver->nm_to_eV(wavelength), region->qwlen); // earlier: qwtotallen
    }
};


}}} // # namespace plask::gain::fermigolden

#endif // PLASK__SOLVER__GAIN_FERMIGOLDEN_FERMIGOLDEN_H
