#ifndef PLASK__SOLVER__OPTICAL__SLAB_FOURIER_REFLECTION_3D_H
#define PLASK__SOLVER__OPTICAL__SLAB_FOURIER_REFLECTION_3D_H

#include <plask/plask.hpp>

#include "../solver.hpp"
#include "../reflection.hpp"
#include "expansion3d.hpp"

#ifdef minor
#   undef minor
#endif

namespace plask { namespace optical { namespace slab {

/**
 * Reflection transformation solver in Cartesian 3D geometry.
 */
struct PLASK_SOLVER_API FourierSolver3D: public SlabSolver<SolverOver<Geometry3D>> {

    friend struct ExpansionPW3D;

    std::string getClassName() const override { return "optical.Fourier3D"; }

    /// Indication of parameter to search
    enum What {
        WHAT_WAVELENGTH,    ///< Search for wavelength
        WHAT_K0,            ///< Search for normalized frequency
        WHAT_KLONG,         ///< Search for longitudinal wavevector
        WHAT_KTRAN          ///< Search for transverse wavevector
    };

    /// Expansion rule
    enum ExpansionRule {
        RULE_OLD,
        RULE_DIRECT,
        RULE_INVERSE,
        RULE_COMBINED
    };

    struct Mode {
        Expansion::Component symmetry_long;     ///< Mode symmetry in long direction
        Expansion::Component symmetry_tran;     ///< Mode symmetry in tran direction
        double lam0;                            ///< Wavelength for which integrals are computed
        dcomplex k0;                            ///< Stored mode frequency
        dcomplex klong;                         ///< Stored mode effective index
        dcomplex ktran;                         ///< Stored mode transverse wavevector
        double power;                           ///< Mode power [mW]
        double tolx;                            ///< Tolerance for mode comparison

        Mode(const ExpansionPW3D& expansion, double tolx):
            symmetry_long(expansion.symmetry_long),
            symmetry_tran(expansion.symmetry_tran),
            lam0(expansion.lam0),
            k0(expansion.k0),
            klong(expansion.klong),
            ktran(expansion.ktran),
            power(1.),
            tolx(tolx) {}

        bool operator==(const Mode& other) const {
            return is_equal(k0, other.k0) && is_equal(klong, other.klong) && is_equal(ktran, other.ktran)
                && symmetry_long == other.symmetry_long && symmetry_tran == other.symmetry_tran &&
                ((isnan(lam0) && isnan(other.lam0)) || lam0 == other.lam0)
            ;
        }

        bool operator==(const ExpansionPW3D& other) const {
            return is_equal(k0, other.k0) && is_equal(klong, other.klong) && is_equal(ktran, other.ktran)
                && symmetry_long == other.symmetry_long && symmetry_tran == other.symmetry_tran &&
                ((isnan(lam0) && isnan(other.lam0)) || lam0 == other.lam0)
            ;
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

    /// Maximum order of the orthogonal base in longitudinal direction
    size_t size_long;
    /// Maximum order of the orthogonal base in transverse direction
    size_t size_tran;

  protected:

    dcomplex klong,                             ///< Longitudinal wavevector [1/µm]
             ktran;                             ///< Transverse wavevector [1/µm]

    Expansion::Component symmetry_long,         ///< Symmetry along longitudinal axis
                         symmetry_tran;         ///< Symmetry along transverse axis

    void onInitialize() override;

    void onInvalidate() override;

    void computeIntegrals() override {
        expansion.computeIntegrals();
    }

    /// Type of discrete cosine transform. Can be only 1 or two
    int dct;

    /// Expansion rule
    ExpansionRule expansion_rule;

  public:

    /// Class responsible for computing expansion coefficients
    ExpansionPW3D expansion;

    /// Computed modes
    std::vector<Mode> modes;

    void clearModes() override {
        modes.clear();
    }

    bool setExpansionDefaults(bool with_k0=true) override {
        bool changed = false;
        if (expansion.getLam0() != getLam0()) { changed = true; expansion.setLam0(getLam0()); }
        if (with_k0) {
            if (expansion.getK0() != getK0()) { changed = true; expansion.setK0(getK0()); }
        }
        if (expansion.getKlong() != getKlong()) { changed = true; expansion.setKlong(getKlong()); }
        if (expansion.getKtran() != getKtran()) { changed = true; expansion.setKtran(getKtran()); }
        if (expansion.getSymmetryLong() != getSymmetryLong()) { changed = true; expansion.setSymmetryLong(getSymmetryLong()); }
        if (expansion.getSymmetryTran() != getSymmetryTran()) { changed = true; expansion.setSymmetryTran(getSymmetryTran()); }
        return changed;
    }

    /// Mesh multiplier for finer computation of the refractive indices in the longitudinal direction
    size_t refine_long;
    /// Mesh multiplier for finer computation of the refractive indices in the transverse direction
    size_t refine_tran;

    /// Smoothing of the normal-direction functions
    double grad_smooth;

    /// Longitudinal PMLs
    PML pml_long;
    /// Transverse PMLs
    PML pml_tran;

    /// Provider for gradient functions
    ProviderFor<GradientFunctions, Geometry3D>::Delegate outGradients;

    FourierSolver3D(const std::string& name="");

    void loadConfiguration(XMLReader& reader, Manager& manager) override;

    /**
     * Find the mode around the specified effective index.
     * This method remembers the determined mode, for retrieval of the field profiles.
     * \param what what to search for
     * \param start initial value of \a what to search the mode around
     * \return determined effective index
     */
    size_t findMode(What what, dcomplex start);

    /// Get order of the orthogonal base in the longitudinal direction
    size_t getLongSize() const { return size_long; }

    /// Get order of the orthogonal base in the transverse direction
    size_t getTranSize() const { return size_tran; }

    /// Set order of the orthogonal base in the longitudinal direction
    void setLongSize(size_t n) {
        size_long = n;
        invalidate();
    }
    /// Set order of the orthogonal base in the transverse direction
    void setTranSize(size_t n) {
        size_tran = n;
        invalidate();
    }

    /// Set order of the orthogonal base
    void setSizes(size_t nl, size_t nt) {
        size_long = nl;
        size_tran = nt;
        invalidate();
    }

    /// Return current mode symmetry
    Expansion::Component getSymmetryLong() const { return symmetry_long; }

    /// Set new mode symmetry
    void setSymmetryLong(Expansion::Component symmetry) {
        if (symmetry != Expansion::E_UNSPECIFIED && geometry && !geometry->isSymmetric(Geometry3D::DIRECTION_LONG))
            throw BadInput(getId(), "Longitudinal symmetry not allowed for asymmetric structure");
        if ((symmetry_long == Expansion::E_UNSPECIFIED) != (symmetry == Expansion::E_UNSPECIFIED))
            invalidate();
        if (klong != 0. && symmetry != Expansion::E_UNSPECIFIED) {
            Solver::writelog(LOG_WARNING, "Resetting klong to 0.");
            klong = 0.;
            expansion.setKlong(0.);
        }
        symmetry_long = symmetry;
    }

    /// Return current mode symmetry
    Expansion::Component getSymmetryTran() const { return symmetry_tran; }

    /// Set new mode symmetry
    void setSymmetryTran(Expansion::Component symmetry) {
        if (symmetry != Expansion::E_UNSPECIFIED && geometry && !geometry->isSymmetric(Geometry3D::DIRECTION_TRAN))
            throw BadInput(getId(), "Transverse symmetry not allowed for asymmetric structure");
        if ((symmetry_tran == Expansion::E_UNSPECIFIED) != (symmetry == Expansion::E_UNSPECIFIED))
            invalidate();
        if (ktran != 0. && symmetry != Expansion::E_UNSPECIFIED) {
            Solver::writelog(LOG_WARNING, "Resetting ktran to 0.");
            ktran = 0.;
            expansion.setKtran(0.);
        }
        symmetry_tran = symmetry;
    }

    /// Get info if the expansion is symmetric
    bool symmetricLong() const { return expansion.symmetric_long(); }

    /// Get info if the expansion is symmetric
    bool symmetricTran() const { return expansion.symmetric_tran(); }

    /// Get longitudinal wavevector
    dcomplex getKlong() const { return klong; }

    /// Set longitudinal wavevector
    void setKlong(dcomplex k)  {
        if (k != 0. && (expansion.symmetric_long() || symmetry_long != Expansion::E_UNSPECIFIED)) {
            Solver::writelog(LOG_WARNING, "Resetting longitudinal mode symmetry");
            symmetry_long = Expansion::E_UNSPECIFIED;
            invalidate();
        }
        klong = k;
    }

    /// Get transverse wavevector
    dcomplex getKtran() const { return ktran; }

    /// Set transverse wavevector
    void setKtran(dcomplex k)  {
        if (k != 0. && (expansion.symmetric_tran() || symmetry_tran != Expansion::E_UNSPECIFIED)) {
            Solver::writelog(LOG_WARNING, "Resetting transverse mode symmetry");
            symmetry_tran = Expansion::E_UNSPECIFIED;
            invalidate();
        }
        ktran = k;
    }

    /// Get current normal smooth
    double getGradSmooth() const { return grad_smooth; }

    /// Set current smooth
    void setGradSmooth(double value) {
        bool changed = grad_smooth != value;
        grad_smooth = value;
        if (changed) this->invalidate();
    }

    /// Get type of the DCT
    int getDCT() const { return dct; }
    /// Set type of the DCT
    void setDCT(int n) {
        if (n != 1 && n != 2)
            throw BadInput(getId(), "Bad DCT type (can be only 1 or 2)");
        if (dct != n) {
            dct = n;
            if (expansion.symmetric_long() || expansion.symmetric_tran()) invalidate();
        }
    }
    /// True if DCT == 2
    bool dct2() const { return dct == 2; }

    /// Get expansion rule
    ExpansionRule getRule() const { return expansion_rule; }
    /// Set expansion rule
    void setRule(ExpansionRule rule) {
        if (rule != expansion_rule) {
            expansion_rule = rule;
            invalidate();
        }
    }

    // /// Get mesh at which material parameters are sampled along longitudinal axis
    // RegularAxis getLongMesh() const { return expansion.mesh->lon(); }
    //
    // /// Get mesh at which material parameters are sampled along transverse axis
    // RegularAxis getTranMesh() const { return expansion.mesh->tran(); }

    Expansion& getExpansion() override { return expansion; }

    /// Return minor field coefficients dimension
    size_t minor() const { return expansion.Nl; }

    /**
     * Get incident field vector for given polarization.
     * \param side incidence side
     * \param polarization polarization of the perpendicularly incident light
     * \param lam wavelength
     * \return incident field vector
     */
    cvector incidentVector(Transfer::IncidentDirection side, Expansion::Component polarization, dcomplex lam=NAN);

    /**
     * Compute incident vector with Gaussian profile
     * \param side incidence side
     * \param polarization polarization of the perpendicularly incident light
     * \param sigma_long,sigma_tran standard deviations in longitudinal and transverse directions
     * \param center_long,center_tran position of the beam center in longitudinal and transverse directions
     * \param lam wavelength
     * \return incident field vector
     */
    cvector incidentGaussian(Transfer::IncidentDirection side, Expansion::Component polarization, double sigma_long, double sigma_tran,
                             double center_long=0., double center_tran=0., dcomplex lam=NAN);

  private:

    size_t initIncidence(Transfer::IncidentDirection side, Expansion::Component polarization, dcomplex lam);

  public:

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
                                                 const shared_ptr<const MeshD<3>>& dst_mesh,
                                                 InterpolationMethod method,
                                                 PropagationDirection part = PROPAGATION_TOTAL) {
        if (!Solver::initCalculation()) setExpansionDefaults(false);
        if (!transfer) initTransfer(expansion, true);
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
                                                 const shared_ptr<const MeshD<3>>& dst_mesh,
                                                 InterpolationMethod method,
                                                 PropagationDirection part = PROPAGATION_TOTAL) {
        if (!Solver::initCalculation()) setExpansionDefaults(false);
        if (!transfer) initTransfer(expansion, true);
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
                                                const shared_ptr<const MeshD<3>>& dst_mesh,
                                                InterpolationMethod method) {
        if (!Solver::initCalculation()) setExpansionDefaults(false);
        if (!transfer) initTransfer(expansion, true);
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
        return transfer->getFieldIntegral(FIELD_E, z1, z2, modes[num].power);
    }

    /**
     * Get ½ H·conj(H) integral between \a z1 and \a z2 for reflected light
     * \param num mode number
     * \param z1 lower integration bound
     * \param z2 upper integration bound
     * \return computed integral
     */
    double getIntegralHH(size_t num, double z1, double z2) {
        applyMode(modes[num]);
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
        if (!transfer) initTransfer(expansion, true);
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
        if (!transfer) initTransfer(expansion, true);
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
        if (!transfer) initTransfer(expansion, true);
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
        if (!transfer) initTransfer(expansion, true);
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
        if (warn && emission != EMISSION_TOP && emission != EMISSION_BOTTOM) {
            writelog(LOG_WARNING, "Mode fields are not normalized unless emission is set to 'top' or 'bottom'");
            warn = false;
        }
        Mode mode(expansion, root.tolx);
        for (size_t i = 0; i != modes.size(); ++i)
            if (modes[i] == mode) return i;
        modes.push_back(mode);
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

    /**
     * Return mode effective index
     * \param n mode number
     */
    dcomplex getEffectiveIndex(size_t n) {
        if (n >= modes.size()) throw NoValue(ModeEffectiveIndex::NAME);
        return modes[n].klong / modes[n].k0;
    }

    void applyMode(const Mode& mode) {
        writelog(LOG_DEBUG, "Current mode <lam: {}nm, klong: {}/um, ktran: {}/um, symmetry: ({},{})>",
                 str(2e3*PI/mode.k0, "({:.3f}{:+.3g}j)", "{:.3f}"),
                 str(mode.klong, "({:.3f}{:+.3g}j)", "{:.3f}"),
                 str(mode.ktran, "({:.3f}{:+.3g}j)", "{:.3f}"),
                 (mode.symmetry_long == Expansion::E_LONG)? "El" : (mode.symmetry_long == Expansion::E_TRAN)? "Et" : "none",
                 (mode.symmetry_tran == Expansion::E_LONG)? "El" : (mode.symmetry_tran == Expansion::E_TRAN)? "Et" : "none"
                );
        if (mode != expansion) {
            expansion.setLam0(mode.lam0);
            expansion.setK0(mode.k0);
            expansion.klong = mode.klong;
            expansion.ktran = mode.ktran;
            expansion.symmetry_long = mode.symmetry_long;
            expansion.symmetry_tran = mode.symmetry_tran;
            clearFields();
        }
    }

    double getWavelength(size_t n) override;

    LazyData<double> getGradients(GradientFunctions::EnumType what,
                                  const shared_ptr<const MeshD<3>>& dst_mesh,
                                  InterpolationMethod interp);

};


}}} // namespace

#endif
