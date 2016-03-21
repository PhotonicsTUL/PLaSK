#ifndef PLASK__SOLVER__OPTICAL__SLAB_FOURIER_REFLECTION_3D_H
#define PLASK__SOLVER__OPTICAL__SLAB_FOURIER_REFLECTION_3D_H

#include <plask/plask.hpp>

#include "../solver.h"
#include "../reflection.h"
#include "expansion3d.h"

namespace plask { namespace solvers { namespace slab {

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

    struct Mode {
        FourierSolver3D* solver;                ///< Solver this mode belongs to
        Expansion::Component symmetry_long;     ///< Mode symmetry in long direction
        Expansion::Component symmetry_tran;     ///< Mode symmetry in tran direction
        boost::optional<double> lam0;           ///< Wavelength for which integrals are computed
        dcomplex k0;                            ///< Stored mode frequency
        dcomplex klong;                         ///< Stored mode effective index
        dcomplex ktran;                         ///< Stored mode transverse wavevector
        double power;                           ///< Mode power [mW]

        Mode(FourierSolver3D* solver): solver(solver), 
            power((solver->emission == EMISSION_TOP || solver->emission == EMISSION_BOTTOM)? 1e-9 : 1.) {}

        bool operator==(const Mode& other) const {
            return is_zero(k0 - other.k0) && is_zero(klong - other.klong) && is_zero(ktran - other.ktran)
                && (!solver->expansion.symmetric_long() || symmetry_long == other.symmetry_long)
                && (!solver->expansion.symmetric_tran() || symmetry_tran == other.symmetry_tran)
            ;
        }
    };

    struct ParamGuard {
        FourierSolver3D* solver;
        boost::optional<double> lam0;
        dcomplex k0, klong, ktran;
        bool recomp;
        ParamGuard(FourierSolver3D* solver): solver(solver),
            lam0(solver->lam0), k0(solver->k0), klong(solver->klong), ktran(solver->ktran) {}
        ~ParamGuard() {
            solver->setLam0(lam0);
            solver->klong = klong; solver->ktran = ktran;
            solver->setK0(k0);
        }
    };

    /// Maximum order of the orthogonal base in longitudinal direction
    size_t size_long;
    /// Maximum order of the orthogonal base in transverse direction
    size_t size_tran;

  protected:

    dcomplex klong,                             ///< Longitudinal wavevector [1/µm]
             ktran;                             ///< Transverse wavevector [1/µm]

    /// Class responsoble for computing expansion coefficients
    ExpansionPW3D expansion;

    void onInitialize() override;

    void onInvalidate() override;

    void computeIntegrals() override {
        expansion.computeIntegrals();
    }

    /// Type of discrete cosine transform. Can be only 1 or two
    int dct;

  public:

    /// Computed modes
    std::vector<Mode> modes;

    void clear_modes() override {
        modes.clear();
    }

    /// Mesh multiplier for finer computation of the refractive indices in the longitudinal direction
    size_t refine_long;
    /// Mesh multiplier for finer computation of the refractive indices in the transverse direction
    size_t refine_tran;

    /// Factor by which the number of coefficients is multiplied for FFT along longitudinal direction.
    /// Afterwards the coefficients are truncated to the required number.
    double oversampling_long;
    /// Factor by which the number of coefficients is multiplied for FFT along transverse direction.
    /// Afterwards the coefficients are truncated to the required number.
    double oversampling_tran;

    /// Longitudinal PMLs
    PML pml_long;
    /// Transverse PMLs
    PML pml_tran;

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
    Expansion::Component getSymmetryLong() const { return expansion.symmetry_long; }

    /// Set new mode symmetry
    void setSymmetryLong(Expansion::Component symmetry) {
        if (symmetry != Expansion::E_UNSPECIFIED && geometry && !geometry->isSymmetric(Geometry3D::DIRECTION_LONG))
            throw BadInput(getId(), "Longitudinal symmetry not allowed for asymmetric structure");
        if ((expansion.symmetric_long() && symmetry == Expansion::E_UNSPECIFIED) ||
            (!expansion.symmetric_long() && symmetry != Expansion::E_UNSPECIFIED))
            invalidate();
        if (klong != 0. && symmetry != Expansion::E_UNSPECIFIED) {
            Solver::writelog(LOG_WARNING, "Resetting klong to 0.");
            klong = 0.;
        }
        if (transfer) transfer->fields_determined = Transfer::DETERMINED_NOTHING;
        expansion.symmetry_long = symmetry;
    }

    /// Return current mode symmetry
    Expansion::Component getSymmetryTran() const { return expansion.symmetry_tran; }

    /// Set new mode symmetry
    void setSymmetryTran(Expansion::Component symmetry) {
        if (symmetry != Expansion::E_UNSPECIFIED && geometry && !geometry->isSymmetric(Geometry3D::DIRECTION_TRAN))
            throw BadInput(getId(), "Transverse symmetry not allowed for asymmetric structure");
        if ((expansion.symmetric_tran() && symmetry == Expansion::E_UNSPECIFIED) ||
            (!expansion.symmetric_tran() && symmetry != Expansion::E_UNSPECIFIED))
            invalidate();
        if (ktran != 0. && symmetry != Expansion::E_UNSPECIFIED) {
            Solver::writelog(LOG_WARNING, "Resetting ktran to 0.");
            ktran = 0.;
        }
        if (transfer) transfer->fields_determined = Transfer::DETERMINED_NOTHING;
        expansion.symmetry_tran = symmetry;
    }

    /// Get longitudinal wavevector
    dcomplex getKlong() const { return klong; }

    /// Set longitudinal wavevector
    void setKlong(dcomplex k)  {
        if (k != 0. && expansion.symmetric_long()) {
            Solver::writelog(LOG_WARNING, "Resetting longitudinal mode symmetry");
            expansion.symmetry_long = Expansion::E_UNSPECIFIED;
            invalidate();
        }
        if (k != klong && transfer) transfer->fields_determined = Transfer::DETERMINED_NOTHING;
        klong = k;
    }

    /// Get transverse wavevector
    dcomplex getKtran() const { return ktran; }

    /// Set transverse wavevector
    void setKtran(dcomplex k)  {
        if (k != 0. && expansion.symmetric_tran()) {
            Solver::writelog(LOG_WARNING, "Resetting transverse mode symmetry");
            expansion.symmetry_tran = Expansion::E_UNSPECIFIED;
            invalidate();
        }
        if (k != ktran && transfer) transfer->fields_determined = Transfer::DETERMINED_NOTHING;
        ktran = k;
    }

    /// Get type of the DCT
    int getDCT() const { return dct; }
    /// Set type of the DCT
    void setDCT(int n) {
        if (n != 1 && n != 2)
            throw BadInput(getId(), "Bad DCT type (can be only 1 or 2)");
        dct = n;
    }
    /// True if DCT == 2
    bool dct2() const { return dct == 2; }

    /// Get mesh at which material parameters are sampled along longitudinal axis
    RegularAxis getLongMesh() const { return expansion.long_mesh; }

    /// Get mesh at which material parameters are sampled along transverse axis
    RegularAxis getTranMesh() const { return expansion.tran_mesh; }

    Expansion& getExpansion() override { return expansion; }

    /// Return minor field coefficients dimension
    size_t minor() const { return expansion.Nl; }
    
  private:

    /**
     * Get incident field vector for given polarization.
     * \param polarization polarization of the perpendicularly incident light
     * \param savidx pointer to which optionally save nonzero incident index
     * \return incident field vector
     */
    cvector incidentVector(Expansion::Component polarization, size_t* savidx=nullptr) {
        if (polarization == ExpansionPW3D::E_UNSPECIFIED)
            throw BadInput(getId(), "Wrong incident polarization specified for the reflectivity computation");
        if (expansion.symmetry_long == Expansion::Component(3-polarization))
            throw BadInput(getId(), "Current longitudinal symmetry is inconsistent with the specified incident polarization");
        if (expansion.symmetry_tran == Expansion::Component(3-polarization))
            throw BadInput(getId(), "Current transverse symmetry is inconsistent with the specified incident polarization");
        size_t idx = (polarization == ExpansionPW3D::E_LONG)? expansion.iEx(0,0) : expansion.iEy(0,0);
        if (savidx) *savidx = idx;
        cvector incident(expansion.matrixSize(), 0.);
        incident[idx] = 1.;
        return incident;
    }

  public:

    /**
     * Get amplitudes of reflected diffraction orders
     * \param polarization polarization of the perpendicularly incident light
     * \param incidence incidence side
     * \param savidx pointer to which optionally save nonzero incident index
     */
    cvector getReflectedAmplitudes(Expansion::Component polarization,
                                   Transfer::IncidentDirection incidence,
                                   size_t* savidx=nullptr);

    /**
     * Get amplitudes of transmitted diffraction orders
     * \param polarization polarization of the perpendicularly incident light
     * \param incidence incidence side
     * \param savidx pointer to which optionally save nonzero incident index
     */
    cvector getTransmittedAmplitudes(Expansion::Component polarization,
                                     Transfer::IncidentDirection incidence,
                                     size_t* savidx=nullptr);

    /**
     * Get reflection coefficient
     * \param polarization polarization of the perpendicularly incident light
     * \param incidence incidence side
     */
    double getReflection(Expansion::Component polarization, Transfer::IncidentDirection incidence);

    /**
     * Get reflection coefficient
     * \param polarization polarization of the perpendicularly incident light
     * \param incidence incidence side
     */
    double getTransmission(Expansion::Component polarization, Transfer::IncidentDirection incidence);

    /**
     * Get electric field at the given mesh for reflected light.
     * \param polarization incident field polarization
     * \param incident incidence direction
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    LazyData<Vec<3,dcomplex>> getReflectedFieldE(Expansion::Component polarization,
                                                 Transfer::IncidentDirection incident,
                                                 const shared_ptr<const MeshD<3>>& dst_mesh,
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
                                                 const shared_ptr<const MeshD<3>>& dst_mesh,
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
                                                const shared_ptr<const MeshD<3>>& dst_mesh,
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
        ParamGuard guard(this);
        if (modes[num].k0 != k0 || modes[num].klong != klong || modes[num].ktran != ktran) {
            setK0(modes[num].k0);
            klong = modes[num].klong;
            ktran = modes[num].ktran;
            transfer->fields_determined = Transfer::DETERMINED_NOTHING;
        }
        return transfer->getFieldVectorE(z);
    }
    
    /**
     * Compute magnetic field coefficients for given \a z
     * \param num mode number
     * \param z position within the layer
     * \return magnetic field coefficients
     */
    cvector getFieldVectorH(size_t num, double z) {
        ParamGuard guard(this);
        if (modes[num].k0 != k0 || modes[num].klong != klong || modes[num].ktran != ktran) {
            setK0(modes[num].k0);
            klong = modes[num].klong;
            ktran = modes[num].ktran;
            transfer->fields_determined = Transfer::DETERMINED_NOTHING;
        }
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
            writelog(LOG_WARNING, "Mode fields are not normalized unless emission is set to 'top' or 'bottom'");
            warn = false;
        }
        Mode mode(this);
        mode.lam0 = lam0;
        mode.k0 = k0; mode.klong = klong; mode.ktran = ktran;
        mode.symmetry_long = expansion.symmetry_long; mode.symmetry_tran = expansion.symmetry_tran;
        for (size_t i = 0; i != modes.size(); ++i)
            if (modes[i] == mode) return i;
        modes.push_back(mode);
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
        return modes[n].klong / modes[n].k0;
    }

    void logCurrentMode() {
        writelog(LOG_DEBUG, "Current mode <lam: {}nm, klong: {}/um, ktran: {}/um, symmetry: ({})>",
                 str(2e3*M_PI/k0, "({:.3f}{:+.3g}j)", "{:.3f}"),
                 str(klong, "({:.3f}{:+.3g}j)", "{:.3f}"),
                 str(ktran, "({:.3f}{:+.3g}j)", "{:.3f}"),
                 (expansion.symmetry_long == Expansion::E_LONG)? "El" : (expansion.symmetry_long == Expansion::E_TRAN)? "Et" : "none",
                 (expansion.symmetry_tran == Expansion::E_LONG)? "El" : (expansion.symmetry_tran == Expansion::E_TRAN)? "Et" : "none"
                );
    }
     
    LazyData<Vec<3,dcomplex>> getE(size_t num, shared_ptr<const MeshD<3>> dst_mesh, InterpolationMethod method) override;

    LazyData<Vec<3,dcomplex>> getH(size_t num, shared_ptr<const MeshD<3>> dst_mesh, InterpolationMethod method) override;

    LazyData<double> getMagnitude(size_t num, shared_ptr<const MeshD<3>> dst_mesh, InterpolationMethod method) override;

  public:

     /**
      * Proxy class for accessing reflected fields
      */
    struct Reflected {

        FourierSolver3D* parent;
         
        Expansion::Component polarization;
        
        Transfer::IncidentDirection side;
        
        double wavelength;
        
        /// Provider of the optical electric field
        typename ProviderFor<LightE,Geometry3D>::Delegate outElectricField;

        /// Provider of the optical magnetic field
        typename ProviderFor<LightH,Geometry3D>::Delegate outMagneticField;

        /// Provider of the optical field intensity
        typename ProviderFor<LightMagnitude,Geometry3D>::Delegate outLightMagnitude;

        /// Return one as the number of the modes
        static size_t size() { return 1; }

        LazyData<Vec<3,dcomplex>> getElectricField(size_t, const shared_ptr<const MeshD<3>>& dst_mesh, InterpolationMethod method) {
            FourierSolver3D::ParamGuard guard(parent);
            parent->setWavelength(wavelength);
            return parent->getReflectedFieldE(polarization, side, dst_mesh, method);
        }
        
        LazyData<Vec<3,dcomplex>> getMagneticField(size_t, const shared_ptr<const MeshD<3>>& dst_mesh, InterpolationMethod method) {
            FourierSolver3D::ParamGuard guard(parent);
            parent->setWavelength(wavelength);
            return parent->getReflectedFieldH(polarization, side, dst_mesh, method);
        }
        
        LazyData<double> getLightMagnitude(size_t, const shared_ptr<const MeshD<3>>& dst_mesh, InterpolationMethod method) {
            FourierSolver3D::ParamGuard guard(parent);
            parent->setWavelength(wavelength);
            return parent->getReflectedFieldMagnitude(polarization, side, dst_mesh, method);
        }
        
        /**
         * Construct proxy.
         * \param wavelength incident light wavelength
         * \param polarization polarization of the perpendicularly incident light
         * \param side incidence side
         */
        Reflected(FourierSolver3D* parent, double wavelength, Expansion::Component polarization, Transfer::IncidentDirection side):
            parent(parent), polarization(polarization), side(side), wavelength(wavelength),
            outElectricField(this, &FourierSolver3D::Reflected::getElectricField, size),
            outMagneticField(this, &FourierSolver3D::Reflected::getMagneticField, size),
            outLightMagnitude(this, &FourierSolver3D::Reflected::getLightMagnitude, size)
        {}
     };
};


}}} // namespace

#endif

