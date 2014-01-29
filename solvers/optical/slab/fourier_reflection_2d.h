#ifndef PLASK__SOLVER__OPTICAL__SLAB_FOURIER_REFLECTION_2D_H
#define PLASK__SOLVER__OPTICAL__SLAB_FOURIER_REFLECTION_2D_H

#include <plask/plask.hpp>

#include "reflection_base.h"
#include "expansion_pw2d.h"

namespace plask { namespace solvers { namespace slab {

/**
 * Reflection transformation solver in Cartesian 2D geometry.
 */
struct FourierReflection2D: public ReflectionSolver<Geometry2DCartesian> {

    std::string getClassName() const { return "optical.FourierReflection2D"; }

    struct Mode {
        FourierReflection2D* solver;                            ///< Solver this mode belongs to
        ExpansionPW2D::Component symmetry;                      ///< Mode horizontal symmetry
        ExpansionPW2D::Component polarization;                  ///< Mode polarization
        dcomplex k0;                                            ///< Stored mode frequency
        dcomplex beta;                                          ///< Stored mode effective index
        dcomplex ktran;                                         ///< Stored mode transverse wavevector
        double power;                                           ///< Mode power [mW]

        Mode(FourierReflection2D* solver): solver(solver), power(1.) {}

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

    /// Class responsoble for computing expansion coefficients
    ExpansionPW2D expansion;

    void onInitialize();

    void onInvalidate();

    void onNewK0() override {
        expansion.computeMaterialCoefficients();
    }
    
  public:

    /// Computed modes
    std::vector<Mode> modes;

    /// Mesh multiplier for finer computation of the refractive indices
    size_t refine;

    /// Lateral PMLs
    PML pml;

    /// Provider for computed effective index
    ProviderFor<EffectiveIndex>::Delegate outNeff;

    FourierReflection2D(const std::string& name="");

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
        fields_determined = DETERMINED_NOTHING;
        expansion.symmetry = symmetry;
    }

    /// Set transverse wavevector
    void setKtran(dcomplex k)  {
        if (expansion.initialized && (expansion.symmetric && k != 0.))
            throw Exception("%1%: Cannot remove mode symmetry now -- invalidate the solver first", getId());
        if (k != ktran) fields_determined = DETERMINED_NOTHING;
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
     * Get period
     */
    double getPeriod() {
        bool not_initialized(!expansion.initialized);
        if (not_initialized) expansion.init(false);
        double result = (expansion.right - expansion.left) * (expansion.symmetric? 2. : 1.);
        if (not_initialized) expansion.free();
        return result;
    }

    /**
     * Get refractive index after expansion
     */
    DataVector<const Tensor3<dcomplex>> getRefractiveIndexProfile(const RectilinearMesh2D& dst_mesh,
                                            InterpolationMethod interp=INTERPOLATION_DEFAULT);

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
            throw BadInput(getId(), "Wrong incident polarization specified for reflectivity computation");
        if (expansion.symmetric) {
            if (expansion.symmetry == ExpansionPW2D::E_UNSPECIFIED)
                expansion.symmetry = polarization;
            else if (expansion.symmetry != polarization)
                throw BadInput(getId(), "Current symmetry is inconsistent with specified incident polarization");
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
    
  public:
    
    /**
     * Get amplitudes of reflected diffraction orders
     * \param polarization polarization of the perpendicularly incident light
     * \param incidence incidence side
     * \param savidx pointer to which optionally save nonzero incident index
     */
    cvector getReflectedAmplitudes(ExpansionPW2D::Component polarization, IncidentDirection incidence, size_t* savidx=nullptr);

    /**
     * Get reflection coefficient
     * \param polarization polarization of the perpendicularly incident light
     * \param incidence incidence side
     */
    double getReflection(ExpansionPW2D::Component polarization, IncidentDirection incidence);

    /**
     * Get amplitudes of transmitted diffraction orders
     * \param polarization polarization of the perpendicularly incident light
     * \param incidence incidence side
     */
    cvector getTransmittedAmplitudes(ExpansionPW2D::Component polarization, IncidentDirection incidence);

    /**
     * Get reflection coefficient
     * \param polarization polarization of the perpendicularly incident light
     * \param incidence incidence side
     */
    double getTransmission(ExpansionPW2D::Component polarization, IncidentDirection incidence);

    /**
     * Get electric field at the given mesh for reflected light.
     * \param Ei incident field vector
     * \param incident incidence direction
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    DataVector<Vec<3,dcomplex>> getReflectedFieldE(ExpansionPW2D::Component polarization, IncidentDirection incident,
                                                   const MeshD<2>& dst_mesh, InterpolationMethod method) {
        return ReflectionSolver<Geometry2DCartesian>::getReflectedFieldE(incidentVector(polarization), incident, dst_mesh, method);
    }

    /**
     * Get magnetic field at the given mesh for reflected light.
     * \param Ei incident field vector
     * \param incident incidence direction
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    DataVector<Vec<3,dcomplex>> getReflectedFieldH(ExpansionPW2D::Component polarization, IncidentDirection incident,
                                                   const MeshD<2>& dst_mesh, InterpolationMethod method) {
        return ReflectionSolver<Geometry2DCartesian>::getReflectedFieldH(incidentVector(polarization), incident, dst_mesh, method);
    }

    /**
     * Get light intensity for reflected light.
     * \param Ei incident field vector
     * \param incident incidence direction
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    DataVector<double> getReflectedFieldIntensity(ExpansionPW2D::Component polarization, IncidentDirection incident,
                                                  const MeshD<2>& dst_mesh, InterpolationMethod method) {
        return ReflectionSolver<Geometry2DCartesian>::getReflectedFieldIntensity(incidentVector(polarization), incident, dst_mesh, method);
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
        outLightIntensity.fireChanged();
        return modes.size()-1;
    }

    size_t nummodes() const { return outNeff.size(); }

    /**
     * Return mode effective index
     * \param n mode number
     */
    dcomplex getEffectiveIndex(size_t n) {
        if (n >= modes.size()) throw NoValue(EffectiveIndex::NAME);
        return modes[n].beta / modes[n].k0;
    }

    /**
     * Compute electric field
     * \param num mode number
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    const DataVector<const Vec<3,dcomplex>> getE(size_t num, const MeshD<2>& dst_mesh, InterpolationMethod method);

    /**
     * Compute magnetic field
     * \param num mode number
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    const DataVector<const Vec<3,dcomplex>> getH(size_t num, const MeshD<2>& dst_mesh, InterpolationMethod method);

    /**
     * Compute light intensity
     * \param num mode number
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    const DataVector<const double> getIntensity(size_t num, const MeshD<2>& dst_mesh, InterpolationMethod method);

  public:
    
    /**
     * Proxy class for accessing reflected fields
     */
    struct Reflected {
        
        /// Provider of the optical electric field
        typename ProviderFor<OpticalElectricField,Geometry2DCartesian>::Delegate outElectricField;

        /// Provider of the optical magnetic field
        typename ProviderFor<OpticalMagneticField,Geometry2DCartesian>::Delegate outMagneticField;

        /// Provider of the optical field intensity
        typename ProviderFor<LightIntensity,Geometry2DCartesian>::Delegate outLightIntensity;
    
        /// Return one as the number of the modes
        static size_t size() { return 1; }
        
        /**
         * Construct proxy.
         * \param wavelength incident light wavelength
         * \param polarization polarization of the perpendicularly incident light
         * \param side incidence side
         */
        Reflected(FourierReflection2D* parent, double wavelength, ExpansionPW2D::Component polarization, FourierReflection2D::IncidentDirection side):
            outElectricField([=](size_t, const MeshD<2>& dst_mesh, InterpolationMethod method) -> DataVector<const Vec<3,dcomplex>> {
                parent->setWavelength(wavelength);
                return parent->getReflectedFieldE(polarization, side, dst_mesh, method); }, size),
            outMagneticField([=](size_t, const MeshD<2>& dst_mesh, InterpolationMethod method) -> DataVector<const Vec<3,dcomplex>> {
                parent->setWavelength(wavelength);
                return parent->getReflectedFieldH(polarization, side, dst_mesh, method); }, size),
            outLightIntensity([=](size_t, const MeshD<2>& dst_mesh, InterpolationMethod method) -> DataVector<const double> {
                parent->setWavelength(wavelength);
                return parent->getReflectedFieldIntensity(polarization, side, dst_mesh, method); }, size)
        {}
    };
};


}}} // namespace

#endif

