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

    std::string getClassName() const { return "optical.Fourier2D"; }

    struct Mode {
        FourierSolver2D* solver;                            ///< Solver this mode belongs to
        ExpansionPW2D::Component symmetry;                  ///< Mode horizontal symmetry
        ExpansionPW2D::Component polarization;              ///< Mode polarization
        dcomplex k0;                                        ///< Stored mode frequency
        dcomplex beta;                                      ///< Stored mode effective index
        dcomplex ktran;                                     ///< Stored mode transverse wavevector
        double power;                                       ///< Mode power [mW]

        Mode(FourierSolver2D* solver): solver(solver), power(1.) {}

        bool operator==(const Mode& other) const {
            return is_zero(k0 - other.k0) && is_zero(beta - other.beta) && is_zero(ktran - other.ktran)
                && (!solver->expansion.symmetric || symmetry == other.symmetry)
                && (!solver->expansion.separated || polarization == other.polarization)
            ;
        }
    };

  protected:

    /// Maximum order of the orthogonal base
    size_t size;

    /// Class responsible for computing expansion coefficients
    ExpansionPW2D expansion;

    void onInitialize();

    void onInvalidate();

    void computeCoefficients() override {
        expansion.computeMaterialCoefficients();
    }

  public:

    /// Computed modes
    std::vector<Mode> modes;

    /// Mesh multiplier for finer computation of the refractive indices
    size_t refine;

    /// Lateral PMLs
    PML pml;

    /// Mirror reflectivities
    boost::optional<std::pair<double,double>> mirrors;

    /// Provider for computed effective index
    ProviderFor<EffectiveIndex>::Delegate outNeff;

    FourierSolver2D(const std::string& name="");

    void loadConfiguration(XMLReader& reader, Manager& manager);

    /**
     * Find the mode around the specified effective index.
     * This method remembers the determined mode, for retrieval of the field profiles.
     * \param neff initial effective index to search the mode around
     * \return determined effective index
     */
    size_t findMode(dcomplex neff);

    /// Get order of the orthogonal base
    size_t getSize() const { return size; }
    /// Set order of the orthogonal base
    void setSize(size_t n) {
        size = n;
        invalidate();
    }

    /// Return current mode symmetry
    ExpansionPW2D::Component getSymmetry() const { return expansion.symmetry; }
    /// Set new mode symmetry
    void setSymmetry(ExpansionPW2D::Component symmetry) {
        if (geometry && !geometry->isSymmetric(Geometry2DCartesian::DIRECTION_TRAN))
            throw BadInput(getId(), "Symmetry not allowed for asymmetric structure");
        if (expansion.initialized) {
            if (expansion.symmetric && symmetry == ExpansionPW2D::E_UNSPECIFIED)
                throw Exception("%1%: Cannot remove mode symmetry now -- invalidate the solver first", getId());
            if (!expansion.symmetric && symmetry != ExpansionPW2D::E_UNSPECIFIED)
                throw Exception("%1%: Cannot add mode symmetry now -- invalidate the solver first", getId());
        }
        if (ktran != 0.) {
            Solver::writelog(LOG_WARNING, "Resetting ktran to 0.");
            ktran = 0.;
        }
        if (transfer) transfer->fields_determined = Transfer::DETERMINED_NOTHING;
        expansion.symmetry = symmetry;
    }

    /// Set transverse wavevector
    void setKtran(dcomplex k)  {
        if (k != 0.) {
            if (expansion.symmetric) {
                if (expansion.initialized)
                    throw Exception("%1%: Cannot remove mode symmetry now -- invalidate the solver first", getId());
                else
                    Solver::writelog(LOG_WARNING, "Resetting mode symmetry");
            }
            expansion.symmetric = Expansion::E_UNSPECIFIED;
        }
        if (k != ktran && transfer) transfer->fields_determined = Transfer::DETERMINED_NOTHING;
        ktran = k;
    }

    /// Return current mode polarization
    ExpansionPW2D::Component getPolarization() const { return expansion.polarization; }
    /// Set new mode polarization
    void setPolarization(ExpansionPW2D::Component polarization) {
        if (expansion.initialized) {
            if (expansion.separated && polarization == ExpansionPW2D::E_UNSPECIFIED)
                throw Exception("%1%: Cannot remove polarizations separation now -- invalidate the solver first", getId());
            if (!expansion.separated && polarization != ExpansionPW2D::E_UNSPECIFIED)
                throw Exception("%1%: Cannot add polarizations separation now -- invalidate the solver first", getId());
        }
        expansion.polarization = polarization;
    }

    /**
     * Get mesh at which material parameters are sampled
     */
    RegularAxis getXmesh() const { return expansion.xmesh; }

  private:

    /**
     * Get incident field vector for given polarization.
     * \param polarization polarization of the perpendicularly incident light
     * \param savidx pointer to which optionally save nonzero incident index
     * \return incident field vector
     */
    cvector incidentVector(ExpansionPW2D::Component polarization, size_t* savidx=nullptr) {
        size_t idx;
        if (polarization == ExpansionPW2D::E_UNSPECIFIED)
            throw BadInput(getId(), "Wrong incident polarization specified for the reflectivity computation");
        if (expansion.symmetric) {
            if (expansion.symmetry == ExpansionPW2D::E_UNSPECIFIED)
                expansion.symmetry = polarization;
            else if (expansion.symmetry != polarization)
                throw BadInput(getId(), "Current symmetry is inconsistent with the specified incident polarization");
        }
        if (expansion.separated) {
            expansion.polarization = polarization;
            idx = expansion.iE(0);
        } else {
            idx = (polarization == ExpansionPW2D::E_TRAN)? expansion.iEx(0) : expansion.iEz(0);
        }
        if (savidx) *savidx = idx;
        cvector incident(expansion.matrixSize(), 0.);
        incident[idx] = 1.;
        return incident;
    }

    /**
     * Compute sum of amplitudes for reflection/transmission coefficient
     * \param amplitudes amplitudes to sum
     */
    double sumAmplitutes(const cvector& amplitudes) {
        double result = 0.;
        int N = getSize();
        if (expansion.separated) {
            if (expansion.symmetric) {
                for (int i = 0; i <= N; ++i)
                    result += real(amplitudes[expansion.iE(i)]);
                result = 2.*result - real(amplitudes[expansion.iE(0)]);
            } else {
                for (int i = -N; i <= N; ++i)
                    result += real(amplitudes[expansion.iE(i)]);
            }
        } else {
            if (expansion.symmetric) {
                for (int i = 0; i <= N; ++i)
                    result += real(amplitudes[expansion.iEx(i)]) + real(amplitudes[expansion.iEz(i)]);
                result = 2.*result - real(amplitudes[expansion.iEx(0)]) - real(amplitudes[expansion.iEz(0)]);
            } else {
                for (int i = -N; i <= N; ++i) {
                    result += real(amplitudes[expansion.iEx(i)]) + real(amplitudes[expansion.iEz(i)]);
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
     * \param savidx pointer to which optionally save nonzero incident index
     */
    cvector getReflectedAmplitudes(ExpansionPW2D::Component polarization, Transfer::IncidentDirection incidence, size_t* savidx=nullptr);

    /**
     * Get amplitudes of transmitted diffraction orders
     * \param polarization polarization of the perpendicularly incident light
     * \param incidence incidence side
     * \param savidx pointer to which optionally save nonzero incident index
     */
    cvector getTransmittedAmplitudes(ExpansionPW2D::Component polarization, Transfer::IncidentDirection incidence, size_t* savidx=nullptr);

    /**
     * Get reflection coefficient
     * \param polarization polarization of the perpendicularly incident light
     * \param incidence incidence side
     */
    double getReflection(ExpansionPW2D::Component polarization, Transfer::IncidentDirection incidence);

    /**
     * Get reflection coefficient
     * \param polarization polarization of the perpendicularly incident light
     * \param incidence incidence side
     */
    double getTransmission(ExpansionPW2D::Component polarization, Transfer::IncidentDirection incidence);

    /**
     * Get electric field at the given mesh for reflected light.
     * \param Ei incident field vector
     * \param incident incidence direction
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    DataVector<Vec<3,dcomplex>> getReflectedFieldE(ExpansionPW2D::Component polarization,
                                                   Transfer::IncidentDirection incident,
                                                   shared_ptr<const MeshD<2>> dst_mesh,
                                                   InterpolationMethod method) {
        initCalculation();
        return transfer->getReflectedFieldE(incidentVector(polarization), incident, dst_mesh, method);
    }

    /**
     * Get magnetic field at the given mesh for reflected light.
     * \param Ei incident field vector
     * \param incident incidence direction
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    DataVector<Vec<3,dcomplex>> getReflectedFieldH(ExpansionPW2D::Component polarization,
                                                   Transfer::IncidentDirection incident,
                                                   shared_ptr<const MeshD<2>> dst_mesh,
                                                   InterpolationMethod method) {
        initCalculation();
        return transfer->getReflectedFieldH(incidentVector(polarization), incident, dst_mesh, method);
    }

    /**
     * Get light intensity for reflected light.
     * \param Ei incident field vector
     * \param incident incidence direction
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    DataVector<double> getReflectedFieldMagnitude(ExpansionPW2D::Component polarization,
                                                  Transfer::IncidentDirection incident,
                                                  shared_ptr<const MeshD<2>> dst_mesh,
                                                  InterpolationMethod method) {
        initCalculation();
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

    size_t nummodes() const { return modes.size(); }

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
    virtual const DataVector<const Vec<3,dcomplex>> getE(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method) override;

    /**
     * Compute magnetic field
     * \param num mode number
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    virtual const DataVector<const Vec<3,dcomplex>> getH(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method) override;

    /**
     * Compute light intensity
     * \param num mode number
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    virtual const DataVector<const double> getIntensity(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method) override;

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
        Reflected(FourierSolver2D* parent, double wavelength, ExpansionPW2D::Component polarization, Transfer::IncidentDirection side):
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

