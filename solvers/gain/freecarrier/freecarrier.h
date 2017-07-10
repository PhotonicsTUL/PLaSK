#ifndef PLASK__SOLVER__GAIN_FREECARRIER_FREECARRIER_H
#define PLASK__SOLVER__GAIN_FREECARRIER_FREECARRIER_H

#include <plask/plask.hpp>

#include <boost/math/tools/roots.hpp>
using boost::math::tools::toms748_solve;

namespace plask { namespace gain { namespace freecarrier {

template <typename GeometryT> struct GainSpectrum;

/**
 * Gain solver using Fermi Golden Rule
 */
template <typename GeometryType>
struct PLASK_SOLVER_API FreeCarrierGainSolver: public SolverWithMesh<GeometryType, MeshAxis>
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
            throw plask::Exception("FreeCarrierGainSolver requires solid layers.");
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

        double averageNr(double lam, double T, double conc=0.) const {
            double nr = 0.;
            for (size_t i = 0; i != materials.size(); ++i)
                if (isQW(i)) nr += thicknesses[i] * materials[i]->Nr(lam, T, conc).real();
            return nr / totalqw;
        }

        std::vector<shared_ptr<Material>> materials;///< All materials in the active region
        std::vector<double> thicknesses;            ///< Thicknesses of the layers in the active region
        std::vector<size_t> wells;                  ///< Division of the active region into separate quantum wells

        double total;                               ///< Total active region thickness [µm]
        double totalqw;                             ///< Total accepted quantum wells thickness [µm]
        double bottom;                              ///< Bottom spacer thickness [µm]
        double top;                                 ///< Top spacer thickness [µm]

        enum ConsideredHoles: unsigned {
            NO_HOLES = 0,
            HEAVY_HOLES = 1,
            LIGHT_HOLES = 2,
            BOTH_HOLES = 3
        } holes;                                    ///< Type of holes existing in the active region


        /**
         * Summarize active region, check for appropriateness and compute some values
         * \param solver solver
         */
        void summarize(const FreeCarrierGainSolver<GeometryType>* solver);
    };

    /// Structure containing active region data for current used
    struct ActiveRegionParams {
        const ActiveRegionInfo& region;
        std::vector<double> U[3];          ///< Band levels
        std::vector<Tensor2<double>> M[3]; ///< Effective masses
        double Mt;                         ///< Momentum matrix element

        std::vector<Level> levels[3];      ///< Approximate electron, heavy and light hole levels
        double Eg;                         ///< Wells band gap
        size_t nhh,                        ///< Number of electron–heavy hole pairs important for gain
               nlh;                        ///< Number of electron–light hole pairs important for gain

        ActiveRegionParams(const FreeCarrierGainSolver* solver, const ActiveRegionInfo& region, double T, bool quiet=false);

        ActiveRegionParams(const FreeCarrierGainSolver* solver, const ActiveRegionInfo& region, bool quiet=false):
            ActiveRegionParams(solver, region, solver->T0, quiet) {}

        explicit ActiveRegionParams(const FreeCarrierGainSolver* solver, const ActiveRegionParams& ref, double T, bool quiet=false):
            ActiveRegionParams(solver, ref.region, T, quiet) {
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

    /// Receiver for band edges
    ReceiverFor<BandEdges,GeometryType> inBandEdges;

    /// Provider for gain distribution
    typename ProviderFor<Gain,GeometryType>::Delegate outGain;

    /// Provider for eergy levels
//     typename ProviderFor<EnergyLevels>::Delegate outEnergyLevels;

    FreeCarrierGainSolver(const std::string& name="");

    virtual ~FreeCarrierGainSolver();

    virtual std::string getClassName() const override;

    virtual void loadConfiguration(plask::XMLReader& reader, plask::Manager& manager) override;

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

    /// Walk the energy to bracket the Fermi level
    template <class F>
    std::pair<double, double> fermi_bracket_and_solve(F f, double guess, double step, boost::uintmax_t& max_iter) const {
        double a = guess - 0.5*step, b = guess + 0.5*step;
        double fa = f(a), fb = f(b);
        if (fa == 0.) return std::make_pair(a, a);
        if (fb == 0.) return std::make_pair(b, b);
        boost::uintmax_t count = max_iter - 1;
        if ((fa < 0.) == (fa < fb)) {
            while ((fa < 0.) == (fb < 0.)) {
                if(count == 0) return std::make_pair(a, b);
                a = b; fa = fb;
                b += step; fb = f(b);
                if (fb == 0.) return std::make_pair(b, b);
                --count;
            }
        } else {
            while ((fb < 0.) == (fa < 0.)) {
                if(count == 0) return std::make_pair(a, b);
                b = a; fb = fa;
                a -= step; fa = f(a);
                if (fa == 0.) return std::make_pair(a, a);
                --count;
            }
        }
        auto res = toms748_solve(f, a, b, fa, fb, [this](double l, double r){ return r-l < levelsep; }, count);
        max_iter += count;
        return res;
    }

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

    double T0;                      ///< Temperature used for compiting level estimates

  protected:

    double levelsep;                ///< Minimum separation between distinct levels

    bool strained;                  ///< Consider strain in QW?

    /// Estimate energy levels
    void estimateLevels();

    /// Initialize the solver
    virtual void onInitialize() override;

    /// Invalidate the gain
    virtual void onInvalidate() override;

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
    const LazyData<double> getGainData(Gain::EnumType what, const shared_ptr<const MeshD<2>>& dst_mesh, double wavelength, InterpolationMethod interp=INTERPOLATION_DEFAULT);

    /**
     * Compute the energy levels.
     * \param num active region number
     * \return energy levels for specified active region
     */
//     const EnergyLevels getEnergyLevels(size_t num=0);

  public:

    std::vector<ActiveRegionParams> params0;    ///< Reference active region params

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

    /// Find gain after convolution
    double getGain(double hw, double Fc, double Fv, double T, double nr, const ActiveRegionParams& params) const;

    double getT0() const { return T0; }
    void setT0(double T) { T0 = T; this->invalidate(); }

    double getLifeTime() const { return lifetime; }
    void setLifeTime(double iLifeTime)  { lifetime = iLifeTime; }

    double getMatrixElem() const { return matrixelem; }
    void setMatrixElem(double iMatrixElem)  { matrixelem = iMatrixElem; }

    bool getStrained() const { return strained; }
    void setStrained(bool value) { strained = value; this->invalidate(); }

    /**
     * Reg gain spectrum object for future use
     */
    shared_ptr<GainSpectrum<GeometryType>> getGainSpectrum(const Vec<2>& point);

    friend struct GainSpectrum<GeometryType>;
};


/**
 * Cached gain spectrum
 */
template <typename GeometryT>
struct GainSpectrum {

    typedef typename FreeCarrierGainSolver<GeometryT>::ActiveRegionParams ActiveRegionParams;

    FreeCarrierGainSolver<GeometryT>* solver; ///< Source solver
    boost::optional<Vec<2>> point;            ///< Point in which the gain is calculated

    size_t reg;                         ///< Active region containg the point
    double temp;                        ///< Temperature
    double conc;                        ///< Concentration
    double Fc, Fv;                      ///< Quasi-fermi levels
    /// Active region params
    std::unique_ptr<ActiveRegionParams> params;


    GainSpectrum(FreeCarrierGainSolver<GeometryT>* solver, const Vec<2> point): solver(solver), point(point) {
        for (size_t i = 0; i != solver->regions.size(); ++i) {
            if (solver->regions[i].contains(point)) {
                reg = i;
                solver->inTemperature.changedConnectMethod(this, &GainSpectrum::onChange);
                solver->inCarriersConcentration.changedConnectMethod(this, &GainSpectrum::onChange);
                temp = solver->inTemperature(plask::make_shared<const OnePointMesh<2>>(point))[0];
                conc = solver->inCarriersConcentration(plask::make_shared<const OnePointMesh<2>>(point))[0];
                updateParams();
                return;
            };
        }
        throw BadInput(solver->getId(), "Point {0} does not belong to any active region", point);
    }

    GainSpectrum(FreeCarrierGainSolver<GeometryT>* solver, double T, double n, size_t reg):
        solver(solver), temp(T), conc(n), reg(reg) {
        updateParams();
    }

    void updateParams() {
        params.reset(new ActiveRegionParams(solver, solver->params0[reg], temp));
        Fc = Fv = NAN;
        solver->findFermiLevels(Fc, Fv, conc, temp, *params);
    }

    void onChange(ReceiverBase&, ReceiverBase::ChangeReason) {
        temp = solver->inTemperature(plask::make_shared<const OnePointMesh<2>>(*point))[0];
        conc = solver->inCarriersConcentration(plask::make_shared<const OnePointMesh<2>>(*point))[0];
        updateParams();
    }

    ~GainSpectrum() {
        solver->inTemperature.changedDisconnectMethod(this, &GainSpectrum::onChange);
        solver->inCarriersConcentration.changedDisconnectMethod(this, &GainSpectrum::onChange);
    }

    /**
     * Get gain at given wavelength
     * \param wavelength wavelength to get gain
     * \return gain
     */
    double getGain(double wavelength) const {
        double nr = solver->regions[reg].averageNr(wavelength, temp, conc);
        return solver->getGain(phys::h_eVc1e9 / wavelength, Fc, Fv, temp, nr, *params);
    }
};


}}} // # namespace plask::gain::freecarrier

#endif // PLASK__SOLVER__GAIN_FREECARRIER_FREECARRIER_H
