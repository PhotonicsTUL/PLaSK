#ifndef PLASK__SOLVER_SLAB_REFLECTIONBASE_H
#define PLASK__SOLVER_SLAB_REFLECTIONBASE_H

#include "matrices.h"
#include "diagonalizer.h"
#include "slab_base.h"

namespace plask { namespace solvers { namespace slab {

/**
 * Base class for all solvers using reflection matrix method.
 */
template <typename GeometryT>
struct PLASK_SOLVER_API ReflectionSolver: public SlabSolver<GeometryT> {

    /// Struct containng data for computing field in a layer
    struct LayerFields {
        cvector F, B;
    };

  protected:

    /// Indicates what has been determined
    enum Determined {
        DETERMINED_NOTHING = 0, ///< Nothing has been determined
        DETERMINED_RESONANT,    ///< Resonant field has been determined
        DETERMINED_REFLECTED    ///< Reflected field has been determined
    };

    cmatrix interface_field_matrix;             ///< Determined field at the interface
    dcomplex* interface_field;                  ///< Pointer to the interface field data

    cmatrix A;                                  ///< The (diagonalized field) admittance matrix for the interface
    cmatrix M;                                  ///< The final matrix which must fulfill M * E = 0

    dcomplex* evals;                            ///< Found eigenvalues of matrix M
    double* rwork;                              ///< temporary space
    int lwork;                                  ///< temporary space
    dcomplex* work;                             ///< temporary space

    dcomplex k0,                                ///< Normalized frequency [1/µm]
             klong,                             ///< Longitudinal wavevector [1/µm]
             ktran;                             ///< Transverse wavevector [1/µm]

    cmatrix P;                                  ///< current reflection matrix
    bool allP;                                  ///< do we need to keep all the P matrices?

    Determined fields_determined;               ///< Are the diagonalized fields determined for all layers?
    std::vector<LayerFields> fields;            ///< Vector of fields computed for each layer

    bool emitting;                              ///< \c True if the structure is emitting vertically.

  private:

    cdiagonal phas;                             ///< current phase shift matrix
    int* ipiv;                                  ///< pivot vector
    std::vector<cmatrix> memP;                  ///< reflection matrices for each layer

    void onInputChanged(ReceiverBase&, ReceiverBase::ChangeReason) {
        this->recompute_coefficients = true;
    }

  public:

    ~ReflectionSolver();

    /// Get current wavelength
    dcomplex getWavelength() const { return 2e3*M_PI / k0; }
    /// Set current wavelength
    void setWavelength(dcomplex lambda, bool recompute=true) {
        dcomplex k = 2e3*M_PI / lambda;
        if (k != k0) {
            fields_determined = DETERMINED_NOTHING;
            k0 = k;
            this->recompute_coefficients |= recompute;
        }
    }

    /// Get current k0
    dcomplex getK0() const { return k0; }
    /// Set current k0
    void setK0(dcomplex k, bool recompute=true) {
        if (k != k0) {
            fields_determined = DETERMINED_NOTHING;
            k0 = k;
            if (k0 == 0.) k0 = 1e-12;
            this->recompute_coefficients |= recompute;
        }
    }

    /// Get longitudinal wavevector
    dcomplex getKlong() const { return klong; }
    /// Set longitudinal wavevector
    void setKlong(dcomplex k)  {
        if (k != klong) fields_determined = DETERMINED_NOTHING;
        klong = k;
    }

    /// Get transverse wavevector
    dcomplex getKtran() const { return ktran; }
    /// Set transverse wavevector
    void setKtran(dcomplex k)  {
        if (k != ktran) fields_determined = DETERMINED_NOTHING;
        ktran = k;
    }

    /// Get discontinuity matrix determinant for the current parameters
    dcomplex getDeterminant() {
        this->initCalculation();
        return determinant();
    }

    /// Direction specification for reflection calculations
    enum IncidentDirection {
        INCIDENCE_TOP,      ///< Incident light propagating from top (downwards)
        INCIDENCE_BOTTOM    ///< Incident light propagating from bottom (upwards)
    };

    /**
     * Get vector of reflection coefficients
     * \param incident incident field profile
     * \param side incident light side
     */
    cvector getReflectionVector(const cvector& incident, IncidentDirection direction);

    /**
     * Get vector of transmission coefficients
     * \param incident incident field profile
     * \param side incident light side
     */
    cvector getTransmissionVector(const cvector& incident, IncidentDirection side);

    /**
     * Get current expansion coefficients at the matching interface
     * \return vector of current expansion coefficients at the interface
     */
    cvector getInterfaceVector();

    /**
     * Compute electric field coefficients for given \a z
     * \param z position within the layer
     * \param n layer number
     * \return electric field coefficients
     */
    cvector getFieldVectorE(double z, int n);

    /**
     * Compute magnetic field coefficients for given \a z
     * \param z position within the layer
     * \param n layer number
     * \return magnetic field coefficients
     */
    cvector getFieldVectorH(double z, int n);

    /// Get \c emitting attribute
    bool getEmitting() const { return emitting; }
    /// Set \c emitting attribute
    void setEmitting(bool value) {
        emitting = value;
        this->invalidate();
    }

  protected:

    ReflectionSolver(const std::string& name): SlabSolver<GeometryT>(name),
        interface_field(nullptr), evals(nullptr), rwork(nullptr), work(nullptr),
        k0(NAN), klong(0.), ktran(0.),
        emitting(true), ipiv(nullptr) {
        this->inTemperature.changedConnectMethod(this, &ReflectionSolver::onInputChanged);
        this->inGain.changedConnectMethod(this, &ReflectionSolver::onInputChanged);
    }

    /// Initialize memory for calculations
    void init();

    /// Cleanup memory
    void cleanup();

    /// Init diagonalization
    void initDiagonalization() {
    // Get new coefficients if needed
        if (this->recompute_coefficients) {
            this->computeCoefficients();
            this->recompute_coefficients = false;
        }
        this->diagonalizer->initDiagonalization(k0, klong, ktran);
    }

    /// Compute discontinuity matrix determinant for the current parameters
    dcomplex determinant();

    /// Get admittance (A) and discontinuity (M) matrices for the whole structure
    void getFinalMatrix() {
        getAM(0, this->interface-1, false);
        getAM(this->stack.size()-1, this->interface, true);
    }

    /**
     * Get admittance (A) and discontinuity (M) matrices for half of the structure
     * \param start start of the transfer
     * \param end end of the transfer
     * \param add if \c true then M matrix is added to the previous value
     * \param mfac factor to multiply M matrix befere addition
     */
    void getAM(size_t start, size_t end, bool add, double mfac=1.);

    /**
     * Find reflection matrix for the part of the structure
     * \param start starting layer
     * \param end last layer (reflection matrix is computed for this layer)
     */
    void findReflection(int start, int end);

    /**
     * Store P matrix if we want it for field compuation
     * \param n layer number
     */
    void storeP(size_t n);

    /**
     * Determine coefficients in each layer necessary for fields calculations.
     */
    void determineFields();

    /**
     * Determine coefficients in each layer necessary for fields calculations.
     * This method is called for reflected fields.
     * \param incident incident field vector
     * \param side incidence side
     */
    void determineReflectedFields(const cvector& incident, IncidentDirection side);

    /**
     * Compute electric field at the given mesh.
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    DataVector<Vec<3,dcomplex>> computeFieldE(const shared_ptr<const MeshD<GeometryT::DIM>>& dst_mesh, InterpolationMethod method);

    /**
     * Compute magnetic field at the given mesh.
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    DataVector<Vec<3,dcomplex>> computeFieldH(const shared_ptr<const MeshD<GeometryT::DIM> > &dst_mesh, InterpolationMethod method);

    /**
     * Compute light intensity.
     * \param power mode power
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    DataVector<double> computeFieldMagnitude(double power, const shared_ptr<const MeshD<GeometryT::DIM> > &dst_mesh, InterpolationMethod method);

    /**
     * Get electric field at the given mesh for resonant mode.
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    DataVector<Vec<3,dcomplex>> getFieldE(const shared_ptr<const MeshD<GeometryT::DIM>>& dst_mesh, InterpolationMethod method) {
        determineFields();
        return computeFieldE(dst_mesh, method);
    }

    /**
     * Get magnetic field at the given mesh for resonant mode.
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    DataVector<Vec<3,dcomplex>> getFieldH(const shared_ptr<const MeshD<GeometryT::DIM>>& dst_mesh, InterpolationMethod method) {
        determineFields();
        return computeFieldH(dst_mesh, method);
    }

    /**
     * Get light intensity for resonant mode.
     * \param power mode power
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    DataVector<double> getFieldIntensity(double power, const shared_ptr<const MeshD<GeometryT::DIM>>& dst_mesh, InterpolationMethod method) {
        determineFields();
        return computeFieldMagnitude(power, dst_mesh, method);
    }

    /**
     * Get electric field at the given mesh for reflected light.
     * \param incident incident field vector
     * \param side incidence side
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    DataVector<Vec<3,dcomplex>> getReflectedFieldE(const cvector& incident, IncidentDirection side, const shared_ptr<const MeshD<GeometryT::DIM>>& dst_mesh, InterpolationMethod method) {
        determineReflectedFields(incident, side);
        return computeFieldE(dst_mesh, method);
    }

    /**
     * Get magnetic field at the given mesh for reflected light.
     * \param incident incident field vector
     * \param side incidence side
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    DataVector<Vec<3,dcomplex>> getReflectedFieldH(const cvector& incident, IncidentDirection side, const shared_ptr<const MeshD<GeometryT::DIM>>& dst_mesh, InterpolationMethod method) {
        determineReflectedFields(incident, side);
        return computeFieldH(dst_mesh, method);
    }

    /**
     * Get light intensity for reflected light.
     * \param incident incident field vector
     * \param side incidence side
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    DataVector<double> getReflectedFieldIntensity(const cvector& incident, IncidentDirection side, const shared_ptr<const MeshD<GeometryT::DIM>>& dst_mesh, InterpolationMethod method) {
        determineReflectedFields(incident, side);
        return computeFieldMagnitude(1., dst_mesh, method);
    }
};


}}} // namespace plask::solvers::slab

#endif // PLASK__SOLVER_SLAB_REFLECTIONBASE_H
