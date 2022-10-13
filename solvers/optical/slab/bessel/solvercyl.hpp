#ifndef PLASK__SOLVER__SLAB_SOLVERCYL_H
#define PLASK__SOLVER__SLAB_SOLVERCYL_H

#include <plask/plask.hpp>

#include "../solver.hpp"
#include "../reflection.hpp"
#include "expansioncyl-fini.hpp"
#include "expansioncyl-infini.hpp"


namespace plask { namespace optical { namespace slab {

/**
 * Reflection transformation solver in Cartesian 2D geometry.
 */
struct PLASK_SOLVER_API BesselSolverCyl: public SlabSolver<SolverWithMesh<Geometry2DCylindrical,MeshAxis>> {

    friend struct ExpansionBessel;
    friend struct ExpansionBesselFini;
    friend struct ExpansionBesselInfini;

    enum BesselDomain {
        DOMAIN_FINITE,
        DOMAIN_INFINITE
    };

    enum InfiniteWavevectors {
        WAVEVECTORS_UNIFORM,
        WAVEVECTORS_NONUNIFORM,
        // WAVEVECTORS_LEGENDRE,
        WAVEVECTORS_LAGUERRE,
        WAVEVECTORS_MANUAL
    };

    enum Rule {
        RULE_INVERSE_0,
        RULE_INVERSE_1,
        RULE_INVERSE_2,
        RULE_DIRECT
    };

    const char* ruleName() {
        switch (rule) {
            case RULE_INVERSE_0: return "inverse";
            case RULE_INVERSE_1: return "inverse (mod 1)";
            case RULE_INVERSE_2: return "inverse (mod 2)";
            case RULE_DIRECT: return "direct";
        }
        return "unknown";
    }

    std::string getClassName() const override { return "optical.BesselCyl"; }

    struct Mode {
        double lam0;                    ///< Wavelength for which integrals are computed
        dcomplex k0;                    ///< Stored mode frequency
        int m;                          ///< Stored angular parameter
        double power;                   ///< Mode power [mW]
        double tolx;                    ///< Tolerance for mode comparison

        Mode(const std::unique_ptr<ExpansionBessel>& expansion, double tolx):
            lam0(expansion->lam0), k0(expansion->k0), m(expansion->m), power(1.), tolx(tolx) {}

        bool operator==(const Mode& other) const {
            return m == other.m && is_equal(k0, other.k0) && is_equal(lam0, other.lam0) &&
                   ((isnan(lam0) && isnan(other.lam0)) || lam0 == other.lam0);
        }

        bool operator==(const std::unique_ptr<ExpansionBessel>& other) const {
            return m == other->m && is_equal(k0, other->k0) && is_equal(lam0, other->lam0) &&
                   ((isnan(lam0) && isnan(other->lam0)) || lam0 == other->lam0);
        }

        template <typename T>
        bool operator!=(const T& other) const {
            return !(*this == other);
        }

      private:

        /// Compare mode arguments
        template <typename T>
        bool is_equal(T a, T b) const {
            return abs(a-b) <= tolx;
        }
    };

  protected:

    /// Domain over which the field is expanded (finite or infinite)
    BesselDomain domain;

    /// Angular dependency index
    int m;

    /// Maximum order of the orthogonal base
    size_t size;

    void onInitialize() override;

    void onInvalidate() override;

    void computeIntegrals() override {
        expansion->computeIntegrals();
    }

    /// Coefficients matrix expansion rule
    Rule rule;

    /// Scale for k-points for infinite expansion
    double kscale;

    /// Maximum k-vector/k0 for infinite expansion
    double kmax;

    /// How integration points and weight should be computed
    InfiniteWavevectors kmethod;

  public:

    /// Abscissa for manual wavevectors
    std::vector<double> klist;

    /// Weights for manual wavevectors
    boost::optional<std::vector<double>> kweights;

    /// Class responsible for computing expansion coefficients
    std::unique_ptr<ExpansionBessel> expansion;

    /// Computed modes
    std::vector<Mode> modes;

    /// Return list of wavevectors
    std::vector<double> getKlist() {
        initCalculation();
        computeIntegrals();
        return expansion->getKpts();
    }

    /// Set manual k-list
    void setKlist(const std::vector<double>& values) {
        if (kmethod != WAVEVECTORS_MANUAL) {
            invalidate();
            this->writelog(LOG_WARNING, "Setting Hankel transform method to Manual");
            kmethod = WAVEVECTORS_MANUAL;
        }
        klist = values;
    }

    /// Return list of wavevector weights
    std::vector<double> getKweights() {
        if (domain == DOMAIN_INFINITE) {
            initCalculation();
            computeIntegrals();
            ExpansionBesselInfini* ex = dynamic_cast<ExpansionBesselInfini*>(expansion.get());
            if (ex) return std::vector<double>(ex->kdelts.begin(), ex->kdelts.end());
        }
        return std::vector<double>();
    }

    /// Set manual k-weights
    void setKweights(const std::vector<double>& values) {
        if (kmethod != WAVEVECTORS_MANUAL) {
            invalidate();
            this->writelog(LOG_WARNING, "Setting Hankel transform method to Manual");
            kmethod = WAVEVECTORS_MANUAL;
        }
        kweights.reset(values);
    }

    void clearKweights() {
        kweights.reset();
    }

    void clearModes() override {
        modes.clear();
    }

    bool setExpansionDefaults(bool with_k0=true) override {
        bool changed = false;
        if (expansion->getLam0() != getLam0()) { changed = true; expansion->setLam0(getLam0()); }
        if (with_k0) {
            if (expansion->getK0() != getK0()) { changed = true; expansion->setK0(getK0()); }
        }
        if (expansion->getM() != getM()) { changed = true; expansion->setM(getM()); }
        return changed;
    }

    /// Expected integration estimate error
    double integral_error;

    /// Maximum number of integration points in a single segment
    size_t max_integration_points;

    /// Lateral PMLs
    PML pml;

    /// Provider for computed modal extinction
    typename ProviderFor<ModeLoss>::Delegate outLoss;

    BesselSolverCyl(const std::string& name="");

    void loadConfiguration(XMLReader& reader, Manager& manager) override;

    /**
     * Find the mode around the specified effective index.
     * This method remembers the determined mode, for retrieval of the field profiles.
     * \param start initial wavelength value to search the mode around
     * \return determined effective index
     */
    size_t findMode(dcomplex start, int m=1);

    /// Get order of the orthogonal base
    size_t getSize() const { return size; }
    /// Set order of the orthogonal base
    void setSize(size_t n) {
        size = n;
        invalidate();
    }

    /// Get scale for infinite wavevectors
    double getKscale() const { return kscale; }
    /// Set scale for infinite wavevectors
    void setKscale(double s) { kscale = s; }

    /// Get maximum wavevector/k0 for infinite domain
    double getKmax() const { return kmax; }
    /// Set maximum wavevector/k0 for infinite domain
    void setKmax(double s) { kmax = s; }

    /// Get method of infinite k-space integration
    InfiniteWavevectors getKmethod() const { return kmethod; }
    /// Set method of infinite k-space integration
    void setKmethod(InfiniteWavevectors k) { kmethod = k; }

    /// Get current domain
    BesselDomain getDomain() const { return domain; }
    /// Set new domain
    void setDomain(BesselDomain dom) {
        domain = dom;
        invalidate();
    }

    /// Get current domain
    Rule getRule() const { return rule; }
    /// Set new domain
    void setRule(Rule r) {
        rule = r;
        invalidate();
    }

    /// Get order of the orthogonal base
    unsigned getM() const { return m; }
    /// Set order of the orthogonal base
    void setM(unsigned n) { m = n; }

    Expansion& getExpansion() override { return *expansion; }

    /**
     * Get electric field at the given mesh for reflected light.
     * \param incident incident field vector
     * \param side incidence direction
     * \param dst_mesh target mesh
     * \param method interpolation method
     * \param part part of the field (forward-, backward-propagating, or total) that is wanted
     */
    LazyData<Vec<3,dcomplex>> getScatteredFieldE(const cvector& incident,
                                                 Transfer::IncidentDirection side,
                                                 const shared_ptr<const MeshD<2>>& dst_mesh,
                                                 InterpolationMethod method,
                                                 PropagationDirection part = PROPAGATION_TOTAL) {
        if (!Solver::initCalculation()) setExpansionDefaults(false);
        if (!transfer) initTransfer(*expansion, true);
        return transfer->getScatteredFieldE(incident, side, dst_mesh, method, part);
    }

    /**
     * Get magnetic field at the given mesh for reflected light.
     * \param incident incident field vector
     * \param side incidence direction
     * \param dst_mesh target mesh
     * \param method interpolation method
     * \param part part of the field (forward-, backward-propagating, or total) that is wanted
     */
    LazyData<Vec<3,dcomplex>> getScatteredFieldH(const cvector& incident,
                                                 Transfer::IncidentDirection side,
                                                 const shared_ptr<const MeshD<2>>& dst_mesh,
                                                 InterpolationMethod method,
                                                 PropagationDirection part = PROPAGATION_TOTAL) {
        if (!Solver::initCalculation()) setExpansionDefaults(false);
        if (!transfer) initTransfer(*expansion, true);
        return transfer->getScatteredFieldH(incident, side, dst_mesh, method, part);
    }

    /**
     * Get light intensity for reflected light.
     * \param incident incident field vector
     * \param side incidence direction
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    LazyData<double> getScatteredFieldMagnitude(const cvector& incident,
                                                Transfer::IncidentDirection side,
                                                const shared_ptr<const MeshD<2>>& dst_mesh,
                                                InterpolationMethod method) {
        if (!Solver::initCalculation()) setExpansionDefaults(false);
        if (!transfer) initTransfer(*expansion, true);
        return transfer->getScatteredFieldMagnitude(incident, side, dst_mesh, method);
    }

    /**
     * Compute electric field coefficients for given \a z
     * \param num mode number
     * \param z position within the layer
     * \return electric field coefficients
     */
    cvector getFieldVectorE(size_t num, double z) {
        applyMode(modes[num]);
        return transfer->getFieldVectorE(z);
    }

    /**
     * Compute magnetic field coefficients for given \a z
     * \param num mode number
     * \param z position within the layer
     * \return magnetic field coefficients
     */
    cvector getFieldVectorH(size_t num, double z) {
        applyMode(modes[num]);
        return transfer->getFieldVectorH(z);
    }

    /**
     * Get ½ E·conj(E) integral between \a z1 and \a z2
     * \param num mode number
     * \param z1 lower integration bound
     * \param z2 upper integration bound
     * \return computed integral
     */
    double getIntegralEE(size_t num, double z1, double z2) {
        applyMode(modes[num]);
        this->writelog(LOG_WARNING, "Integral functions give wrong results and will be removed in the future!");
        return transfer->getFieldIntegral(FIELD_E, z1, z2, modes[num].power);
    }

    /**
     * Get ½ H·conj(H) integral between \a z1 and \a z2
     * \param num mode number
     * \param z1 lower integration bound
     * \param z2 upper integration bound
     * \return computed integral
     */
    double getIntegralHH(size_t num, double z1, double z2) {
        applyMode(modes[num]);
        this->writelog(LOG_WARNING, "Integral functions give wrong results and will be removed in the future!");
        return transfer->getFieldIntegral(FIELD_H, z1, z2, modes[num].power);
    }

    /**
     * Compute electric field coefficients for given \a z for reflected light
     * \param incident incident field vector
     * \param side incidence side
     * \param z position within the layer
     * \return electric field coefficients
     */
    cvector getScatteredFieldVectorE(const cvector& incident, Transfer::IncidentDirection side, double z) {
        if (!Solver::initCalculation()) setExpansionDefaults(false);
        if (!transfer) initTransfer(*expansion, true);
        return transfer->getScatteredFieldVectorE(incident, side, z, PROPAGATION_TOTAL);
    }

    /**
     * Compute magnetic field coefficients for given \a z for reflected light
     * \param incident incident field vector
     * \param side incidence side
     * \param z position within the layer
     * \return magnetic field coefficients
     */
    cvector getScatteredFieldVectorH(const cvector& incident, Transfer::IncidentDirection side, double z) {
        if (!Solver::initCalculation()) setExpansionDefaults(false);
        if (!transfer) initTransfer(*expansion, true);
        return transfer->getScatteredFieldVectorH(incident, side, z, PROPAGATION_TOTAL);
    }

    /**
     * Get ½ E·conj(E) integral between \a z1 and \a z2 for reflected light
     * \param incident incident field vector
     * \param side incidence side
     * \param z1 lower integration bound
     * \param z2 upper integration bound
     * \return computed integral
     */
    double getScatteredIntegralEE(const cvector& incident, Transfer::IncidentDirection side, double z1, double z2) {
        if (!Solver::initCalculation()) setExpansionDefaults(false);
        if (!transfer) initTransfer(*expansion, true);
        this->writelog(LOG_WARNING, "Integral functions give wrong results and will be removed in the future!");
        return transfer->getScatteredFieldIntegral(FIELD_E, incident, side, z1, z2);
    }

    /**
     * Get ½ H·conj(H) integral between \a z1 and \a z2
     * \param incident incident field vector
     * \param side incidence side
     * \param z1 lower integration bound
     * \param z2 upper integration bound
     * \return computed integral
     */
    double getScatteredIntegralHH(const cvector& incident, Transfer::IncidentDirection side, double z1, double z2) {
        if (!Solver::initCalculation()) setExpansionDefaults(false);
        if (!transfer) initTransfer(*expansion, true);
        this->writelog(LOG_WARNING, "Integral functions give wrong results and will be removed in the future!");
        this->writelog(LOG_WARNING, "Integral functions give wrong results and will be removed in the future!");
        return transfer->getScatteredFieldIntegral(FIELD_H, incident, side, z1, z2);
    }

    /// Check if the current parameters correspond to some mode and insert it
    size_t setMode() {
        if (abs2(this->getDeterminant()) > root.tolf_max*root.tolf_max)
            throw BadInput(this->getId(), "Cannot set the mode, determinant too large");
        return insertMode();
    }

  protected:
    /// Insert mode to the list or return the index of the exiting one
    size_t insertMode() {
        static bool warn = true;
        if (warn && ((emission != EMISSION_TOP && emission != EMISSION_BOTTOM) || domain == DOMAIN_INFINITE)) {
            if (domain == DOMAIN_INFINITE)
                writelog(LOG_WARNING, "Mode fields are not normalized (infinite domain)");
            else
                writelog(LOG_WARNING, "Mode fields are not normalized (emission direction not specified)");
            warn = false;
        }
        Mode mode(expansion, root.tolx);
        for (size_t i = 0; i != modes.size(); ++i)
            if (modes[i] == mode) return i;
        modes.push_back(mode);
        outWavelength.fireChanged();
        outLoss.fireChanged();
        outLightMagnitude.fireChanged();
        outLightE.fireChanged();
        outLightH.fireChanged();
        return modes.size()-1;
    }

    size_t nummodes() const override { return modes.size(); }

    double applyMode(size_t n) override {
        if (n >= modes.size()) throw BadInput(this->getId(), "Mode {0} has not been computed", n);
        applyMode(modes[n]);
        return modes[n].power;
    }

    void applyMode(const Mode& mode) {
        writelog(LOG_DEBUG, "Current mode <m: {:d}, lam: {}nm>", mode.m, str(2e3*PI/mode.k0, "({:.3f}{:+.3g}j)"));
        expansion->setLam0(mode.lam0);
        expansion->setK0(mode.k0);
        expansion->setM(mode.m);
    }

    /**
     * Return mode modal loss
     * \param n mode number
     */
    double getModalLoss(size_t n) {
        if (n >= modes.size()) throw NoValue(ModeLoss::NAME);
        return 2e4 * modes[n].k0.imag();  // 2e4  2/µm -> 2/cm
    }

    double getWavelength(size_t n) override;

#ifndef NDEBUG
  public:
    cmatrix epsV_k(size_t layer);
    cmatrix epsTss(size_t layer);
    cmatrix epsTsp(size_t layer);
    cmatrix epsTps(size_t layer);
    cmatrix epsTpp(size_t layer);
    cmatrix muV_k();
    cmatrix muTss();
    cmatrix muTsp();
    cmatrix muTps();
    cmatrix muTpp();
#endif

};



}}} // # namespace plask::optical::slab

#endif // PLASK__SOLVER__SLAB_SOLVERCYL_H
