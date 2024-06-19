/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#ifndef PLASK__SOLVER__GAIN_FREECARRIER_FREECARRIER_H
#define PLASK__SOLVER__GAIN_FREECARRIER_FREECARRIER_H

#include <plask/plask.hpp>

#include <boost/math/tools/roots.hpp>
using boost::math::tools::toms748_solve;

namespace plask { namespace gain { namespace freecarrier {

template <typename BaseT> struct GainSpectrum;

extern OmpNestedLock gain_omp_lock;

/**
 * Gain solver using Fermi Golden Rule
 */
template <typename BaseT> struct PLASK_SOLVER_API FreeCarrierGainSolver : public BaseT {
    /// Which level to compute
    enum WhichLevel : size_t { EL = EnergyLevels::ELECTRONS, HH = EnergyLevels::HEAVY_HOLES, LH = EnergyLevels::LIGHT_HOLES };

    typedef typename BaseT::SpaceType GeometryType;

    enum { DIM = GeometryType::DIM };

    struct ActiveRegionParams;

    /// Data for energy level
    struct Level {
        double E;           ///< Level energy
        Tensor2<double> M;  ///< Representative well index
        double thickness;   ///< Cumulated wells thickness

        Level(double E, const Tensor2<double>& M) : E(E), M(M) {}

        Level(double E, const Tensor2<double>& M, double t) : E(E), M(M), thickness(t) {}

        // TODO Better effective thickness computation (based on wavefunction)?
        Level(double E, const Tensor2<double>& M, WhichLevel which, const ActiveRegionParams& params);
    };

    /// Structure containing information about each active region
    struct ActiveRegionInfo {
        shared_ptr<StackContainer<DIM>> layers;  ///< Stack containing all layers in the active region
        Vec<DIM> origin;                         ///< Location of the active region stack origin

        ActiveRegionInfo(Vec<DIM> origin) : layers(plask::make_shared<StackContainer<DIM>>()), origin(origin) {}

        /// Return number of layers in the active region with surrounding barriers
        size_t size() const { return layers->getChildrenCount(); }

        /// Return material of \p n-th layer
        shared_ptr<Material> getLayerMaterial(size_t n) const {
            auto block = static_cast<Block<DIM>*>(static_cast<Translation<DIM>*>(layers->getChildNo(n).get())->getChild().get());
            if (auto m = block->singleMaterial()) return m;
            throw plask::Exception("freeCarrierGainSolver requires solid layers.");
        }

        /// Return translated bounding box of \p n-th layer
        typename Primitive<DIM>::Box getLayerBox(size_t n) const {
            return static_cast<GeometryObjectD<DIM>*>(layers->getChildNo(n).get())->getBoundingBox() + origin;
        }

        /// Return \p true if given layer is quantum well
        bool isQW(size_t n) const { return static_cast<Translation<DIM>*>(layers->getChildNo(n).get())->getChild()->hasRole("QW"); }

        /// Return bounding box of the whole active region
        typename Primitive<DIM>::Box getBoundingBox() const { return layers->getBoundingBox() + origin; }

        /// Return vertical position of the center of the active region

        /// Return \p true if the point is in the active region
        bool contains(const Vec<DIM>& point) const { return getBoundingBox().contains(point); }

        /// Return \p true if given point is inside quantum well
        bool inQW(const Vec<DIM>& point) const {
            if (!contains(point)) return false;
            assert(layers->getChildForHeight(point.vert() - origin.vert()));
            return layers->getChildForHeight(point.vert() - origin.vert())->getChild()->hasRole("QW");
        }

        double averageNr(double lam, double T, double conc = 0.) const {
            double nr = 0.;
            for (size_t i = 0; i != materials.size(); ++i)
                if (isQW(i)) nr += thicknesses[i] * materials[i]->Nr(lam, T, conc).real();
            return nr / totalqw;
        }

        std::vector<shared_ptr<Material>> materials;  ///< All materials in the active region
        std::vector<double> thicknesses;              ///< Thicknesses of the layers in the active region
        std::vector<size_t> wells;                    ///< Division of the active region into separate quantum wells

        double total;    ///< Total active region thickness (µm)
        double totalqw;  ///< Total accepted quantum wells thickness (µm)
        double bottom;   ///< Bottom spacer thickness (µm)
        double top;      ///< Top spacer thickness (µm)

        enum ConsideredHoles : unsigned {
            NO_HOLES = 0,
            HEAVY_HOLES = 1,
            LIGHT_HOLES = 2,
            BOTH_HOLES = 3
        } holes;  ///< Type of holes existing in the active region

        /**
         * Summarize active region, check for appropriateness and compute some values
         * \param solver solver
         */
        void summarize(const FreeCarrierGainSolver<BaseT>* solver);
    };

    /// Structure containing active region data for current used
    struct PLASK_SOLVER_API ActiveRegionParams {
        const ActiveRegionInfo& region;
        std::vector<double> U[3];           ///< Band levels
        std::vector<Tensor2<double>> M[3];  ///< Effective masses
        double Mt;                          ///< Momentum matrix element

        std::vector<Level> levels[3];  ///< Approximate electron, heavy and light hole levels
        double Eg;                     ///< Wells band gap
        size_t nhh,                    ///< Number of electron–heavy hole pairs important for gain
            nlh;                       ///< Number of electron–light hole pairs important for gain

        ActiveRegionParams(const FreeCarrierGainSolver* solver,
                           const ActiveRegionInfo& region,
                           double T,
                           bool quiet = false,
                           double mt = 0.);

        ActiveRegionParams(const FreeCarrierGainSolver* solver, const ActiveRegionInfo& region, bool quiet = false, double mt = 0.)
            : ActiveRegionParams(solver, region, solver->T0, quiet, mt) {}

        explicit ActiveRegionParams(const FreeCarrierGainSolver* solver, const ActiveRegionParams& ref, double T, bool quiet = true)
            : ActiveRegionParams(solver, ref.region, T, quiet, ref.Mt) {
            nhh = ref.nhh;
            nlh = ref.nlh;
            for (size_t which = 0; which < 3; ++which) {
                double shift = delta(WhichLevel(which), ref);
                levels[which].reserve(ref.levels[which].size());
                for (Level level : ref.levels[which]) levels[which].emplace_back(level.E + shift, level.M, level.thickness);
            }
        }

        double sideU(WhichLevel which) const { return 0.5 * (U[which][0] + U[which][U[which].size() - 1]); }

        Tensor2<double> sideM(WhichLevel which) const { return 0.5 * (M[which][0] + M[which][M[which].size() - 1]); }

        double delta(WhichLevel which, const ActiveRegionParams& ref) const {
            assert(U[which].size() == ref.U[which].size());
            double delta = 0;
            for (size_t i = 0; i < U[which].size(); ++i) {
                delta += U[which][i] - ref.U[which][i];
            }
            return delta / double(U[which].size());
        }
    };

    /// List of active regions
    std::vector<ActiveRegionInfo> regions;

    /// Receiver for temperature.
    ReceiverFor<Temperature, GeometryType> inTemperature;

    /// Receiver for carriers concentration in the active region
    ReceiverFor<CarriersConcentration, GeometryType> inCarriersConcentration;

    /// Receiver for band edges
    ReceiverFor<BandEdges, GeometryType> inBandEdges;

    /// Receiver for quasi Fermi levels
    ReceiverFor<FermiLevels, GeometryType> inFermiLevels;

    /// Provider for gain distribution
    typename ProviderFor<Gain, GeometryType>::Delegate outGain;

    /// Provider for energy levels
    typename ProviderFor<EnergyLevels, GeometryType>::Delegate outEnergyLevels;

    FreeCarrierGainSolver(const std::string& name = "");

    virtual ~FreeCarrierGainSolver();

    void loadConfiguration(plask::XMLReader& reader, plask::Manager& manager) override;

  protected:
    /// Substrate material
    shared_ptr<Material> substrateMaterial;

    /// Is substrate material explicitly set?
    bool explicitSubstrate = false;

    /**
     * Detect active regions.
     * Store information about them in the \p regions field.
     */
    virtual void detectActiveRegions() = 0;

    /// Compute determinant for energy levels
    double level(WhichLevel which, double E, const ActiveRegionParams& params, size_t start, size_t stop) const;

    double level(WhichLevel which, double E, const ActiveRegionParams& params) const {
        return level(which, E, params, 0, params.region.materials.size() - 1);
    }

    double level(WhichLevel which, double E, const ActiveRegionParams& params, size_t well) const {
        return level(which, E, params, params.region.wells[well], params.region.wells[well + 1]);
    }

    /// Find levels estimates
    void estimateWellLevels(WhichLevel which, ActiveRegionParams& params, size_t qw) const;

    /// Find levels estimates
    void estimateAboveLevels(WhichLevel which, ActiveRegionParams& params) const;

    /// Walk the energy to bracket the Fermi level
    template <class F>
    std::pair<double, double> fermi_bracket_and_solve(F f, double guess, double step, boost::uintmax_t& max_iter) const {
        double a = guess - 0.5 * step, b = guess + 0.5 * step;
        double fa = f(a), fb = f(b);
        if (fa == 0.) return std::make_pair(a, a);
        if (fb == 0.) return std::make_pair(b, b);
        boost::uintmax_t count = max_iter - 1;
        if ((fa < 0.) == (fa < fb)) {
            while ((fa < 0.) == (fb < 0.)) {
                if (count == 0) return std::make_pair(a, b);
                a = b;
                fa = fb;
                b += step;
                fb = f(b);
                if (fb == 0.) return std::make_pair(b, b);
                --count;
            }
        } else {
            while ((fb < 0.) == (fa < 0.)) {
                if (count == 0) return std::make_pair(a, b);
                b = a;
                fb = fa;
                a -= step;
                fa = f(a);
                if (fa == 0.) return std::make_pair(a, a);
                --count;
            }
        }
        auto res = toms748_solve(
            f, a, b, fa, fb, [this](double l, double r) { return r - l < levelsep; }, count);
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
    double lifetime;    ///< Stimulated emission lifetime [ps]
    double matrixelem;  ///< Optical matrix element [m0*eV]

    double T0;  ///< Temperature used for compiting level estimates

  protected:
    double levelsep;  ///< Minimum separation between distinct levels

    bool strained;  ///< Consider strain in QW?

    /// Estimate energy levels
    void estimateLevels();

    /// Initialize the solver
    void onInitialize() override;

    /// Invalidate the gain
    void onInvalidate() override;

    /// Notify that gain was changed
    void onInputChange(ReceiverBase&, ReceiverBase::ChangeReason) {
        outGain.fireChanged();  // the input changed, so we inform the world that everybody should get the new gain
    }

    /**
     * Compute the gain on the mesh. This method is called by gain provider.
     * \param what what to return (gain or its carriers derivative)
     * \param dst_mesh destination mesh
     * \param wavelength wavelength to compute gain for
     * \param interp interpolation method
     * \return gain distribution
     */
    virtual const LazyData<Tensor2<double>> getGainData(Gain::EnumType what,
                                                        const shared_ptr<const MeshD<DIM>>& dst_mesh,
                                                        double wavelength,
                                                        InterpolationMethod interp = INTERPOLATION_DEFAULT) = 0;

    /**
     * Compute the energy levels.
     * \return energy levels for specified active region
     */
    virtual const LazyData<std::vector<double>> getEnergyLevels(EnergyLevels::EnumType which,
                                                                const shared_ptr<const MeshD<DIM>>& dst_mesh,
                                                                InterpolationMethod interp = INTERPOLATION_DEFAULT) = 0;

  public:
    std::vector<ActiveRegionParams> params0;  ///< Reference active region params

    bool quick_levels;  ///< Are levels computed quickly based on estimates

#ifndef NDEBUG
    double detEl(double E, const ActiveRegionParams& params, size_t well = 0) {
        if (well)
            return level(EL, E, params, well - 1);
        else
            return level(EL, E, params);
    }
    double detHh(double E, const ActiveRegionParams& params, size_t well = 0) {
        if (well)
            return level(HH, E, params, well - 1);
        else
            return level(HH, E, params);
    }
    double detLh(double E, const ActiveRegionParams& params, size_t well = 0) {
        if (well)
            return level(LH, E, params, well - 1);
        else
            return level(LH, E, params);
    }
#endif

    /// Get substrate material
    shared_ptr<Material> getSubstrate() const { return substrateMaterial; }

    /// Set substrate material
    void setSubstrate(shared_ptr<Material> material) {
        bool invalid = substrateMaterial != material;
        substrateMaterial = material;
        explicitSubstrate = bool(material);
        if (invalid) this->invalidate();
    }

    /// Compute quasi-Fermi levels for given concentration and temperature
    void findFermiLevels(double& Fc, double& Fv, double n, double T, const ActiveRegionParams& params) const;

    /// Find gain before convolution
    Tensor2<double> getGain0(double hw, double Fc, double Fv, double T, double nr, const ActiveRegionParams& params) const;

    /// Find gain after convolution
    Tensor2<double> getGain(double hw, double Fc, double Fv, double T, double nr, const ActiveRegionParams& params) const;

    double getT0() const { return T0; }
    void setT0(double T) {
        T0 = T;
        this->invalidate();
    }

    double getLifeTime() const { return lifetime; }
    void setLifeTime(double iLifeTime) { lifetime = iLifeTime; }

    double getMatrixElem() const { return matrixelem; }
    void setMatrixElem(double iMatrixElem) {
        matrixelem = iMatrixElem;
        this->invalidate();
    }

    bool getStrained() const { return strained; }
    void setStrained(bool value) {
        strained = value;
        this->invalidate();
    }

    friend struct GainSpectrum<BaseT>;

    shared_ptr<GainSpectrum<BaseT>> getGainSpectrum(const Vec<DIM>& point) {
        this->initCalculation();
        return make_shared<GainSpectrum<BaseT>>(this, point);
    }

    typedef GainSpectrum<BaseT> GainSpectrumType;
};

/**
 * Cached gain spectrum
 */
template <typename BaseT> struct GainSpectrum {
    typedef typename FreeCarrierGainSolver<BaseT>::ActiveRegionParams ActiveRegionParams;

    FreeCarrierGainSolver<BaseT>* solver;               ///< Source solver
    plask::optional<Vec<BaseT::SpaceType::DIM>> point;  ///< Point in which the gain is calculated

    size_t reg;     ///< Active region containing the point
    double temp;    ///< Temperature
    double conc;    ///< Concentration
    double Fc, Fv;  ///< Quasi-fermi levels
    /// Active region params
    std::unique_ptr<ActiveRegionParams> params;

    GainSpectrum(FreeCarrierGainSolver<BaseT>* solver, const Vec<BaseT::SpaceType::DIM> point) : solver(solver), point(point) {
        for (size_t i = 0; i != solver->regions.size(); ++i) {
            if (solver->regions[i].contains(point)) {
                reg = i;
                solver->inTemperature.changedConnectMethod(this, &GainSpectrum::onChange);
                solver->inCarriersConcentration.changedConnectMethod(this, &GainSpectrum::onChange);
                temp = solver->inTemperature(plask::make_shared<const OnePointMesh<BaseT::SpaceType::DIM>>(point))[0];
                conc = solver->inCarriersConcentration(CarriersConcentration::PAIRS,
                                                       plask::make_shared<const OnePointMesh<BaseT::SpaceType::DIM>>(point))[0];
                updateParams();
                return;
            };
        }
        throw BadInput(solver->getId(), "point {0} does not belong to any active region", point);
    }

    GainSpectrum(FreeCarrierGainSolver<BaseT>* solver, double T, double n, size_t reg)
        : solver(solver), temp(T), conc(n), reg(reg) {
        updateParams();
    }

    void updateParams() {
        params.reset(new ActiveRegionParams(solver, solver->params0[reg], temp));
        Fc = Fv = NAN;
        solver->findFermiLevels(Fc, Fv, conc, temp, *params);
    }

    void onChange(ReceiverBase&, ReceiverBase::ChangeReason) {
        temp = solver->inTemperature(plask::make_shared<const OnePointMesh<BaseT::SpaceType::DIM>>(*point))[0];
        conc = solver->inCarriersConcentration(CarriersConcentration::PAIRS,
                                               plask::make_shared<const OnePointMesh<BaseT::SpaceType::DIM>>(*point))[0];
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
    Tensor2<double> getGain(double wavelength) const {
        double nr = solver->regions[reg].averageNr(wavelength, temp, conc);
        return solver->getGain(phys::h_eVc1e9 / wavelength, Fc, Fv, temp, nr, *params);
    }
};

}}}  // namespace plask::gain::freecarrier

#endif  // PLASK__SOLVER__GAIN_FREECARRIER_FREECARRIER_H
