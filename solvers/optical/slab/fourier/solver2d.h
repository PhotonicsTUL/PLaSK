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
struct PLASK_SOLVER_API FourierSolver2D: public SlabSolver<Geometry2DCartesian> {

    std::string getClassName() const override { return "optical.Fourier2D"; }

    /// Indication of parameter to search
    enum What {
        WHAT_WAVELENGTH,    ///< Search for wavelength
        WHAT_K0,            ///< Search for normalized frequency
        WHAT_NEFF,         ///< Search for longitudinal effective index
        WHAT_KTRAN          ///< Search for transverse wavevector
    };

    struct Mode {
        FourierSolver2D* solver;                            ///< Solver this mode belongs to
        Expansion::Component symmetry;                      ///< Mode horizontal symmetry
        Expansion::Component polarization;                  ///< Mode polarization
        dcomplex k0;                                        ///< Stored mode frequency
        dcomplex beta;                                      ///< Stored mode effective index
        dcomplex ktran;                                     ///< Stored mode transverse wavevector
        double power;                                       ///< Mode power [mW]

        Mode(FourierSolver2D* solver): solver(solver), power(1e-9) {}

        bool operator==(const Mode& other) const {
            return is_zero(k0 - other.k0) && is_zero(beta - other.beta) && is_zero(ktran - other.ktran)
                && (!solver->expansion.symmetric() || symmetry == other.symmetry)
                && (!solver->expansion.separated() || polarization == other.polarization)
            ;
        }
    };

  protected:

    /// Maximum order of the orthogonal base
    size_t size;

    /// Class responsible for computing expansion coefficients
    ExpansionPW2D expansion;

    void onInitialize() override;

    void onInvalidate() override;

    void computeIntegrals() override {
        expansion.computeMaterialCoefficients();
    }

    /// Type of discrete cosine transform. Can be only 1 or two
    int dct;

  public:

    /// Computed modes
    std::vector<Mode> modes;

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
        dct = n;
    }
    /// True if DCT == 2
    bool dct2() const { return dct == 2; }

    /// Set transverse wavevector
    void setKtran(dcomplex k)  {
        if (k != 0. && expansion.symmetric()) {
            Solver::writelog(LOG_WARNING, "Resetting mode symmetry");
            expansion.symmetry = Expansion::E_UNSPECIFIED;
            invalidate();
        }
        if (k != ktran && transfer) transfer->fields_determined = Transfer::DETERMINED_NOTHING;
        ktran = k;
    }

    /// Set transverse wavevector
    void setKlong(dcomplex k)  {
        if (k != 0. && expansion.separated()) {
            Solver::writelog(LOG_WARNING, "Resetting polarizations separation");
            expansion.polarization = Expansion::E_UNSPECIFIED;
            invalidate();
        }
        if (k != klong && transfer) transfer->fields_determined = Transfer::DETERMINED_NOTHING;
        klong = k;
    }

    /// Return current mode symmetry
    Expansion::Component getSymmetry() const { return expansion.symmetry; }

    /// Set new mode symmetry
    void setSymmetry(Expansion::Component symmetry) {
        if (geometry && !geometry->isSymmetric(Geometry2DCartesian::DIRECTION_TRAN))
            throw BadInput(getId(), "Symmetry not allowed for asymmetric structure");
        if ((expansion.symmetric() && symmetry == Expansion::E_UNSPECIFIED) ||
            (!expansion.symmetric() && symmetry != Expansion::E_UNSPECIFIED))
            invalidate();
        if (ktran != 0. && symmetry != Expansion::E_UNSPECIFIED) {
            Solver::writelog(LOG_WARNING, "Resetting ktran to 0.");
            ktran = 0.;
        }
        if (transfer) transfer->fields_determined = Transfer::DETERMINED_NOTHING;
        expansion.symmetry = symmetry;
    }

    /// Return current mode polarization
    Expansion::Component getPolarization() const { return expansion.polarization; }

    /// Set new mode polarization
    void setPolarization(Expansion::Component polarization) {
        if ((expansion.separated() && polarization == Expansion::E_UNSPECIFIED) ||
            (!expansion.separated() && polarization != Expansion::E_UNSPECIFIED))
            invalidate();
        if (klong != 0. && polarization != Expansion::E_UNSPECIFIED) {
            Solver::writelog(LOG_WARNING, "Resetting klong to 0.");
            klong = 0.;
        }
        expansion.polarization = polarization;
    }

    /// Get info if the expansion is separated
    bool separated() const { return expansion.separated(); }

    Expansion& getExpansion() override { return expansion; }

    RegularAxis getMesh() const { return expansion.xmesh; }

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
     * \param Ei incident field vector
     * \param incident incidence direction
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    DataVector<Vec<3,dcomplex>> getReflectedFieldE(Expansion::Component polarization,
                                                   Transfer::IncidentDirection incident,
                                                   shared_ptr<const MeshD<2>> dst_mesh,
                                                   InterpolationMethod method) {
        if (!expansion.initialized && klong == 0.) expansion.polarization = polarization;
        initCalculation();
        initTransfer(expansion, true);
        return transfer->getReflectedFieldE(incidentVector(polarization), incident, dst_mesh, method);
    }

    /**
     * Get magnetic field at the given mesh for reflected light.
     * \param Ei incident field vector
     * \param incident incidence direction
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    DataVector<Vec<3,dcomplex>> getReflectedFieldH(Expansion::Component polarization,
                                                   Transfer::IncidentDirection incident,
                                                   shared_ptr<const MeshD<2>> dst_mesh,
                                                   InterpolationMethod method) {
        if (!expansion.initialized && klong == 0.) expansion.polarization = polarization;
        initCalculation();
        initTransfer(expansion, true);
        return transfer->getReflectedFieldH(incidentVector(polarization), incident, dst_mesh, method);
    }

    /**
     * Get light intensity for reflected light.
     * \param Ei incident field vector
     * \param incident incidence direction
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    DataVector<double> getReflectedFieldMagnitude(Expansion::Component polarization,
                                                  Transfer::IncidentDirection incident,
                                                  shared_ptr<const MeshD<2>> dst_mesh,
                                                  InterpolationMethod method) {
        if (!expansion.initialized && klong == 0.) expansion.polarization = polarization;
        initCalculation();
        initTransfer(expansion, true);
        return transfer->getReflectedFieldMagnitude(incidentVector(polarization), incident, dst_mesh, method);
    }

  protected:

    /// Insert mode to the list or return the index of the exiting one
    size_t insertMode() {
        Mode mode(this);
        mode.k0 = k0; mode.beta = klong; mode.ktran = ktran;
        mode.symmetry = expansion.symmetry; mode.polarization = expansion.polarization;
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
    double getMirrorLosses(dcomplex n) {
        const double lambda = real(2e3*M_PI / k0);
        double R1, R2;
        if (mirrors) {
            std::tie(R1,R2) = *mirrors;
        } else {
            const double n1 = real(geometry->getFrontMaterial()->Nr(lambda, 300.)),
                         n2 = real(geometry->getBackMaterial()->Nr(lambda, 300.));
            R1 = abs((n-n1) / (n+n1));
            R2 = abs((n-n2) / (n+n2));
        }
        return lambda * std::log(R1*R2) / (4e3 * M_PI * geometry->getExtrusion()->getLength());
    }

    /**
     * Compute electric field
     * \param num mode number
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    const DataVector<const Vec<3,dcomplex>> getE(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method) override;

    /**
     * Compute magnetic field
     * \param num mode number
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    const DataVector<const Vec<3,dcomplex>> getH(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method) override;

    /**
     * Compute light intensity
     * \param num mode number
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    const DataVector<const double> getIntensity(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method) override;

  public:

    /**
     * Proxy class for accessing reflected fields
     */
    struct Reflected {

        /// Provider of the optical electric field
        typename ProviderFor<LightE,Geometry2DCartesian>::Delegate outElectricField;

        /// Provider of the optical magnetic field
        typename ProviderFor<LightH,Geometry2DCartesian>::Delegate outMagneticField;

        /// Provider of the optical field intensity
        typename ProviderFor<LightMagnitude,Geometry2DCartesian>::Delegate outLightMagnitude;

        /// Return one as the number of the modes
        static size_t size() { return 1; }

        /**
         * Construct proxy.
         * \param wavelength incident light wavelength
         * \param polarization polarization of the perpendicularly incident light
         * \param side incidence side
         */
        Reflected(FourierSolver2D* parent, double wavelength, Expansion::Component polarization, Transfer::IncidentDirection side):
            outElectricField([=](size_t, const shared_ptr<const MeshD<2>>& dst_mesh, InterpolationMethod method) -> DataVector<const Vec<3,dcomplex>> {
                parent->setWavelength(wavelength);
                return parent->getReflectedFieldE(polarization, side, dst_mesh, method); }, size),
            outMagneticField([=](size_t, const shared_ptr<const MeshD<2>>& dst_mesh, InterpolationMethod method) -> DataVector<const Vec<3,dcomplex>> {
                parent->setWavelength(wavelength);
                return parent->getReflectedFieldH(polarization, side, dst_mesh, method); }, size),
            outLightMagnitude([=](size_t, const shared_ptr<const MeshD<2>>& dst_mesh, InterpolationMethod method) -> DataVector<const double> {
                parent->setWavelength(wavelength);
                return parent->getReflectedFieldMagnitude(polarization, side, dst_mesh, method); }, size)
        {}
    };
};


}}} // namespace

#endif

