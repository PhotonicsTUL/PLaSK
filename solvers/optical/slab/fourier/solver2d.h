#ifndef PLASK__SOLVER__OPTICAL__SLAB_FOURIER_REFLECTION_2D_H
#define PLASK__SOLVER__OPTICAL__SLAB_FOURIER_REFLECTION_2D_H

#include <plask/plask.hpp>

#include "../solver.h"
#include "../reflection.h"
#include "expansion2d.h"

namespace plask { namespace solvers { namespace slab {

/**
 * Reflection transformation solver in Cartesian 2D geometry.
 */
struct PLASK_SOLVER_API FourierSolver2D: public SlabSolver<SolverOver<Geometry2DCartesian>> {

    friend struct ExpansionPW2D;
    
    std::string getClassName() const override { return "optical.Fourier2D"; }

    /// Indication of parameter to search
    enum What {
        WHAT_WAVELENGTH,    ///< Search for wavelength
        WHAT_K0,            ///< Search for normalized frequency
        WHAT_NEFF,         ///< Search for longitudinal effective index
        WHAT_KTRAN          ///< Search for transverse wavevector
    };

    struct Mode {
        Expansion::Component symmetry;          ///< Mode horizontal symmetry
        Expansion::Component polarization;      ///< Mode polarization
        double lam0;                            ///< Wavelength for which integrals are computed
        dcomplex k0;                            ///< Stored mode frequency
        dcomplex beta;                          ///< Stored mode effective index
        dcomplex ktran;                         ///< Stored mode transverse wavevector
        double power;                           ///< Mode power [mW]

        Mode(const ExpansionPW2D& expansion):
            symmetry(expansion.symmetry),
            polarization(expansion.polarization),
            lam0(expansion.lam0),
            k0(expansion.k0),
            beta(expansion.beta),
            ktran(expansion.ktran),
            power(1.) {}

        bool operator==(const Mode& other) const {
            return is_zero(k0 - other.k0) && is_zero(beta - other.beta) && is_zero(ktran - other.ktran)
                && symmetry == other.symmetry && polarization == other.polarization &&
                ((isnan(lam0) && isnan(other.lam0)) || lam0 == other.lam0)
            ;
        }

        bool operator==(const ExpansionPW2D& other) const {
            return is_zero(k0 - other.k0) && is_zero(beta - other.beta) && is_zero(ktran - other.ktran)
                && symmetry == other.symmetry && polarization == other.polarization &&
                ((isnan(lam0) && isnan(other.lam0)) || lam0 == other.lam0)
            ;
        }

        template <typename T>
        bool operator!=(const T& other) const {
            return !(*this == other);
        }
    };

  protected:

    dcomplex beta,                     ///< Longitudinal wavevector [1/µm]
             ktran;                     ///< Transverse wavevector [1/µm]

    Expansion::Component symmetry;      ///< Indicates symmetry if `symmetric`
    Expansion::Component polarization;  ///< Indicates polarization if `separated`

    /// Maximum order of the orthogonal base
    size_t size;

    void onInitialize() override;

    void onInvalidate() override;

    void computeIntegrals() override {
        expansion.computeIntegrals();
    }

    /// Type of discrete cosine transform. Can be only 1 or two
    int dct;

  public:

    /// Class responsible for computing expansion coefficients
    ExpansionPW2D expansion;

    /// Computed modes
    std::vector<Mode> modes;

    void clearModes() override {
        modes.clear();
    }
    
    void setExpansionDefaults(bool with_k0=true) override {
        expansion.setLam0(getLam0());
        if (with_k0) expansion.setK0(getK0());
        expansion.setBeta(getBeta());
        expansion.setKtran(getKtran());
        expansion.setSymmetry(getSymmetry());
        expansion.setPolarization(getPolarization());
    }
    
    /// Mesh multiplier for finer computation of the refractive indices
    size_t refine;

    /// Factor by which the number of coefficients is multiplied for FFT.
    /// Afterwards the coefficients are truncated to the required number.
    double oversampling;

    /// Lateral PMLs
    PML pml;

    /// Mirror reflectivities
    boost::optional<std::pair<double,double>> mirrors;

    /// Provider for computed effective index
    ProviderFor<EffectiveIndex>::Delegate outNeff;

    FourierSolver2D(const std::string& name="");

    void loadConfiguration(XMLReader& reader, Manager& manager) override;

    /**
     * Find the mode around the specified effective index.
     * This method remembers the determined mode, for retrieval of the field profiles.
     * \param what what to search for
     * \param start initial value of \a what to search the mode around
     * \return determined effective index
     */
    size_t findMode(FourierSolver2D::What what, dcomplex start);

    /// Get order of the orthogonal base
    size_t getSize() const { return size; }
    /// Set order of the orthogonal base
    void setSize(size_t n) {
        size = n;
        invalidate();
    }

    /// Get type of the DCT
    int getDCT() const { return dct; }
    /// Set type of the DCT
    void setDCT(int n) {
        if (n != 1 && n != 2)
            throw BadInput(getId(), "Bad DCT type (can be only 1 or 2)");
        if (dct != n) {
            dct = n;
            if (expansion.symmetric()) invalidate();
        }
    }
    /// True if DCT == 2
    bool dct2() const { return dct == 2; }

    /// Get transverse wavevector
    dcomplex getKtran() const { return ktran; }

    /// Set transverse wavevector
    void setKtran(dcomplex k)  {
        if (k != 0. && (expansion.symmetric() || expansion.symmetry != Expansion::E_UNSPECIFIED)) {
            Solver::writelog(LOG_WARNING, "Resetting mode symmetry");
            expansion.symmetry = Expansion::E_UNSPECIFIED;
            invalidate();
        }
        if (k != ktran && transfer) transfer->fields_determined = Transfer::DETERMINED_NOTHING;
        ktran = k;
    }

    /// Set longitudinal wavevector
    void setBeta(dcomplex k)  {
        if (k != 0. && (expansion.separated() || expansion.polarization != Expansion::E_UNSPECIFIED)) {
            Solver::writelog(LOG_WARNING, "Resetting polarizations separation");
            polarization = Expansion::E_UNSPECIFIED;
            invalidate();
        }
        if (k != beta && transfer) transfer->fields_determined = Transfer::DETERMINED_NOTHING;
        beta = k;
    }

    /// Get longitudinal wavevector
    dcomplex getBeta() const { return beta; }

    /// Return current mode symmetry
    Expansion::Component getSymmetry() const { return symmetry; }

    /// Set new mode symmetry
    void setSymmetry(Expansion::Component sym) {
        if (geometry && !geometry->isSymmetric(Geometry2DCartesian::DIRECTION_TRAN))
            throw BadInput(getId(), "Symmetry not allowed for asymmetric structure");
        if ((symmetry == Expansion::E_UNSPECIFIED) != (sym == Expansion::E_UNSPECIFIED))
            invalidate();
        if (ktran != 0. && sym != Expansion::E_UNSPECIFIED) {
            Solver::writelog(LOG_WARNING, "Resetting ktran to 0.");
            ktran = 0.;
            expansion.setKtran(0.);
        }
        symmetry = sym;
    }

    /// Return current mode polarization
    Expansion::Component getPolarization() const { return polarization; }

    /// Set new mode polarization
    void setPolarization(Expansion::Component pol) {
        if ((polarization == Expansion::E_UNSPECIFIED) != (pol == Expansion::E_UNSPECIFIED))
            invalidate();
        if (beta != 0. && pol != Expansion::E_UNSPECIFIED) {
            Solver::writelog(LOG_WARNING, "Resetting beta to 0.");
            beta = 0.;
            expansion.setBeta(0.);
        }
        polarization = pol;
    }

    /// Get info if the expansion is symmetric
    bool symmetric() const { return expansion.symmetric(); }

    /// Get info if the expansion is separated
    bool separated() const { return expansion.separated(); }

    Expansion& getExpansion() override { return expansion; }

    RegularAxis getMesh() const { return *expansion.xmesh; }

  private:

    /**
     * Get incident field vector for given polarization.
     * \param polarization polarization of the perpendicularly incident light
     * \param savidx pointer to which optionally save nonzero incident index
     * \return incident field vector
     */
    cvector incidentVector(Expansion::Component polarization, size_t* savidx=nullptr) {
        size_t idx;
        if (polarization == Expansion::E_UNSPECIFIED)
            throw BadInput(getId(), "Wrong incident polarization specified for the reflectivity computation");
        if (expansion.symmetric() && expansion.symmetry != polarization)
            throw BadInput(getId(), "Current symmetry is inconsistent with the specified incident polarization");
        if (expansion.separated()) {
            expansion.polarization = polarization;
            idx = expansion.iE(0);
        } else {
            idx = (polarization == Expansion::E_TRAN)? expansion.iEx(0) : expansion.iEz(0);
        }
        if (savidx) *savidx = idx;
        cvector incident(expansion.matrixSize(), 0.);
        incident[idx] = (polarization == Expansion::E_TRAN)? 1. : -1.;
        return incident;
    }

    /**
     * Compute sum of amplitudes for reflection/transmission coefficient
     * \param amplitudes amplitudes to sum
     */
    double sumAmplitutes(const cvector& amplitudes) {
        double result = 0.;
        int N = getSize();
        if (expansion.separated()) {
            if (expansion.symmetric()) {
                for (int i = 0; i <= N; ++i)
                    result += real(amplitudes[expansion.iE(i)]);
                result = 2.*result - real(amplitudes[expansion.iE(0)]);
            } else {
                for (int i = -N; i <= N; ++i)
                    result += real(amplitudes[expansion.iE(i)]);
            }
        } else {
            if (expansion.symmetric()) {
                for (int i = 0; i <= N; ++i)
                    result += real(amplitudes[expansion.iEx(i)]);
                result = 2.*result - real(amplitudes[expansion.iEx(0)]);
            } else {
                for (int i = -N; i <= N; ++i) {
                    result += real(amplitudes[expansion.iEx(i)]);
                }
            }
        }
        return result;
    }

  public:

    /**
     * Get amplitudes of reflected diffraction orders
     * \param polarization polarization of the perpendicularly incident light
     * \param incidence incidence side
     */
    cvector getReflectedAmplitudes(Expansion::Component polarization, Transfer::IncidentDirection incidence);

    /**
     * Get amplitudes of transmitted diffraction orders
     * \param polarization polarization of the perpendicularly incident light
     * \param incidence incidence side
     */
    cvector getTransmittedAmplitudes(Expansion::Component polarization, Transfer::IncidentDirection incidence);

    /**
     * Get reflection coefficient
     * \param polarization polarization of the perpendicularly incident light
     * \param incidence incidence side
     */
    double getReflection(Expansion::Component polarization, Transfer::IncidentDirection incidence) {
        return sumAmplitutes(getReflectedAmplitudes(polarization, incidence));
    }

    /**
     * Get reflection coefficient
     * \param polarization polarization of the perpendicularly incident light
     * \param incidence incidence side
     */
    double getTransmission(Expansion::Component polarization, Transfer::IncidentDirection incidence) {
        return sumAmplitutes(getTransmittedAmplitudes(polarization, incidence));
    }

    /**
     * Get electric field at the given mesh for reflected light.
     * \param polarization incident field polarization
     * \param incident incidence direction
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    LazyData<Vec<3,dcomplex>> getReflectedFieldE(Expansion::Component polarization,
                                                 Transfer::IncidentDirection incident,
                                                 shared_ptr<const MeshD<2>> dst_mesh,
                                                 InterpolationMethod method) {
        initCalculation();
        initTransfer(expansion, true);
        return transfer->getReflectedFieldE(incidentVector(polarization), incident, dst_mesh, method);
    }

    /**
     * Get magnetic field at the given mesh for reflected light.
     * \param polarization incident field polarization
     * \param incident incidence direction
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    LazyData<Vec<3,dcomplex>> getReflectedFieldH(Expansion::Component polarization,
                                                 Transfer::IncidentDirection incident,
                                                 shared_ptr<const MeshD<2>> dst_mesh,
                                                 InterpolationMethod method) {
        initCalculation();
        initTransfer(expansion, true);
        return transfer->getReflectedFieldH(incidentVector(polarization), incident, dst_mesh, method);
    }

    /**
     * Get light intensity for reflected light.
     * \param polarization incident field polarization
     * \param incident incidence direction
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    LazyData<double> getReflectedFieldMagnitude(Expansion::Component polarization,
                                                Transfer::IncidentDirection incident,
                                                shared_ptr<const MeshD<2>> dst_mesh,
                                                InterpolationMethod method) {
        initCalculation();
        initTransfer(expansion, true);
        return transfer->getReflectedFieldMagnitude(incidentVector(polarization), incident, dst_mesh, method);
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
     * Compute electric field coefficients for given \a z
     * \param polarization incident field polarization
     * \param incident incidence direction
     * \param z position within the layer
     * \return electric field coefficients
     */
    cvector getReflectedFieldVectorE(Expansion::Component polarization, Transfer::IncidentDirection incident, double z) {
        initCalculation();
        initTransfer(expansion, true);
        return transfer->getReflectedFieldVectorE(incidentVector(polarization), incident, z);
    }
    
    /**
     * Compute magnetic field coefficients for given \a z
     * \param polarization incident field polarization
     * \param incident incidence direction
     * \param z position within the layer
     * \return magnetic field coefficients
     */
    cvector getReflectedFieldVectorH(Expansion::Component polarization, Transfer::IncidentDirection incident, double z) {
        initCalculation();
        initTransfer(expansion, true);
        return transfer->getReflectedFieldVectorH(incidentVector(polarization), incident, z);
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
            writelog(LOG_WARNING, "Mode fields are not normalized");
            warn = false;
        }
        Mode mode(expansion);
        for (size_t i = 0; i != modes.size(); ++i)
            if (modes[i] == mode) return i;
        modes.push_back(mode);
        outNeff.fireChanged();
        outLightMagnitude.fireChanged();
        outElectricField.fireChanged();
        outMagneticField.fireChanged();
        return modes.size()-1;
    }

    size_t nummodes() const override { return modes.size(); }

    /**
     * Return mode effective index
     * \param n mode number
     */
    dcomplex getEffectiveIndex(size_t n) {
        if (n >= modes.size()) throw NoValue(EffectiveIndex::NAME);
        return modes[n].beta / modes[n].k0;
    }

    /// Compute mirror losses for specified effective mode
    double getMirrorLosses(double n) {
        double L = geometry->getExtrusion()->getLength();
        if (isinf(L)) return 0.;
        const double lambda = real(2e3*M_PI / k0);
        double R1, R2;
        if (mirrors) {
            std::tie(R1,R2) = *mirrors;
        } else {
            const double n1 = real(geometry->getFrontMaterial()->Nr(lambda, 300.)),
                         n2 = real(geometry->getBackMaterial()->Nr(lambda, 300.));
            R1 = (n-n1) / (n+n1); R1 *= R1;
            R2 = (n-n2) / (n+n2); R2 *= R2;
        }
        return 0.5 * std::log(R1*R2) / L;
    }

    void applyMode(const Mode& mode) {
        writelog(LOG_DEBUG, "Current mode <lam: {:.2f}nm, neff: {}, ktran: {}/um, polarization: {}, symmetry: {}>",
                 real(2e3*M_PI/mode.k0),
                 str(mode.beta/mode.k0, "{:.3f}{:+.3g}j"),
                 str(mode.ktran, "({:.3g}{:+.3g}j)", "{:.3g}"),
                 (mode.polarization == Expansion::E_LONG)? "El" : (mode.polarization == Expansion::E_TRAN)? "Et" : "none",
                 (mode.symmetry == Expansion::E_LONG)? "El" : (mode.symmetry == Expansion::E_TRAN)? "Et" : "none"
                );
        if (mode != expansion) {
            expansion.setLam0(mode.lam0);
            expansion.setK0(mode.k0);
            expansion.beta = mode.beta;
            expansion.ktran = mode.ktran;
            expansion.symmetry = mode.symmetry;
            expansion.polarization = mode.polarization;
            clearFields();
        }
    }
    
    /**
     * Compute electric field
     * \param num mode number
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    LazyData<Vec<3,dcomplex>> getE(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method) override;

    /**
     * Compute magnetic field
     * \param num mode number
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    LazyData<Vec<3,dcomplex>> getH(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method) override;

    /**
     * Compute light intensity
     * \param num mode number
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    LazyData<double> getMagnitude(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method) override;

  public:

    /**
     * Proxy class for accessing reflected fields
     */
    struct Reflected {

        FourierSolver2D* parent;
        
        Expansion::Component polarization;
        
        Transfer::IncidentDirection side;
        
        double wavelength;
        
        /// Provider of the optical electric field
        typename ProviderFor<LightE,Geometry2DCartesian>::Delegate outElectricField;

        /// Provider of the optical magnetic field
        typename ProviderFor<LightH,Geometry2DCartesian>::Delegate outMagneticField;

        /// Provider of the optical field intensity
        typename ProviderFor<LightMagnitude,Geometry2DCartesian>::Delegate outLightMagnitude;

        /// Return one as the number of the modes
        static size_t size() { return 1; }

        LazyData<Vec<3,dcomplex>> getElectricField(size_t, const shared_ptr<const MeshD<2>>& dst_mesh, InterpolationMethod method) {
            parent->expansion.setLam0(parent->lam0);
            parent->expansion.setK0(2e3*M_PI / wavelength);
            parent->expansion.setBeta(parent->beta);
            parent->expansion.setKtran(parent->ktran);
            parent->expansion.setSymmetry(parent->symmetry);
            if (!parent->expansion.initialized && parent->expansion.beta == 0.)
                parent->expansion.setPolarization(polarization);
            else
                parent->expansion.setPolarization(parent->polarization);
            return parent->getReflectedFieldE(polarization, side, dst_mesh, method);
        }
        
        LazyData<Vec<3,dcomplex>> getMagneticField(size_t, const shared_ptr<const MeshD<2>>& dst_mesh, InterpolationMethod method) {
            parent->expansion.setLam0(parent->lam0);
            parent->expansion.setK0(2e3*M_PI / wavelength);
            parent->expansion.setBeta(parent->beta);
            parent->expansion.setKtran(parent->ktran);
            parent->expansion.setSymmetry(parent->symmetry);
            if (!parent->expansion.initialized && parent->expansion.beta == 0.)
                parent->expansion.setPolarization(polarization);
            else
                parent->expansion.setPolarization(parent->polarization);
            return parent->getReflectedFieldH(polarization, side, dst_mesh, method);
        }
        
        LazyData<double> getLightMagnitude(size_t, const shared_ptr<const MeshD<2>>& dst_mesh, InterpolationMethod method) {
            parent->expansion.setLam0(parent->lam0);
            parent->expansion.setK0(2e3*M_PI / wavelength);
            parent->expansion.setBeta(parent->beta);
            parent->expansion.setKtran(parent->ktran);
            parent->expansion.setSymmetry(parent->symmetry);
            if (!parent->expansion.initialized && parent->expansion.beta == 0.)
                parent->expansion.setPolarization(polarization);
            else
                parent->expansion.setPolarization(parent->polarization);
            return parent->getReflectedFieldMagnitude(polarization, side, dst_mesh, method);
        }
        
        /**
         * Construct proxy.
         * \param wavelength incident light wavelength
         * \param polarization polarization of the perpendicularly incident light
         * \param side incidence side
         */
        Reflected(FourierSolver2D* parent, double wavelength, Expansion::Component polarization, Transfer::IncidentDirection side):
            parent(parent), polarization(polarization), side(side), wavelength(wavelength),
            outElectricField(this, &FourierSolver2D::Reflected::getElectricField, size),
            outMagneticField(this, &FourierSolver2D::Reflected::getMagneticField, size),
            outLightMagnitude(this, &FourierSolver2D::Reflected::getLightMagnitude, size)
        {}
    };
};


}}} // namespace

#endif

