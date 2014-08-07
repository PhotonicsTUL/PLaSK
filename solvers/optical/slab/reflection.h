#ifndef PLASK__SOLVER_SLAB_REFLECTIONBASE_H
#define PLASK__SOLVER_SLAB_REFLECTIONBASE_H

#include "matrices.h"
#include "transfer.h"
#include "solver.h"


namespace plask { namespace solvers { namespace slab {

/**
 * Base class for all solvers using reflection matrix method.
 */
struct PLASK_SOLVER_API ReflectionTransfer: public Transfer {

    /// Struct containing data for computing field in a layer
    struct LayerFields {
        cvector F, B;
    };

  protected:

    cmatrix interface_field_matrix;             ///< Determined field at the interface
    dcomplex* interface_field;                  ///< Pointer to the interface field data

    cmatrix A;                                  ///< The (diagonalized field) admittance matrix for the interface
    cmatrix M;                                  ///< The final matrix which must fulfill M * E = 0

    dcomplex* evals;                            ///< Found eigenvalues of matrix M
    double* rwork;                              ///< temporary space
    int lwork;                                  ///< temporary space
    dcomplex* work;                             ///< temporary space

    cmatrix P;                                  ///< current reflection matrix
    bool allP;                                  ///< do we need to keep all the P matrices?

    std::vector<LayerFields> fields;            ///< Vector of fields computed for each layer

  private:

    cdiagonal phas;                             ///< current phase shift matrix
    int* ipiv;                                  ///< pivot vector
    std::vector<cmatrix> memP;                  ///< reflection matrices for each layer

  public:

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

    ReflectionTransfer(Diagonalizer* diagonalizer);

    ~ReflectionTransfer();

  protected:

    /// Init diagonalization
    void initDiagonalization() {
    // Get new coefficients if needed
        if (solver->recompute_coefficients) {
            solver->computeCoefficients();
            solver->recompute_coefficients = false;
        }
        this->diagonalizer->initDiagonalization(solver->k0, solver->klong, solver->ktran);
    }

    /// Compute discontinuity matrix determinant for the current parameters
    dcomplex determinant();

    /// Get admittance (A) and discontinuity (M) matrices for the whole structure
    void getFinalMatrix() {
        getAM(0, solver->interface-1, false);
        getAM(solver->stack.size()-1, solver->interface, true);
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
    DataVector<Vec<3,dcomplex>> computeFieldE(const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method);

    /**
     * Compute magnetic field at the given mesh.
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    DataVector<Vec<3,dcomplex>> computeFieldH(const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method);

    /**
     * Compute light magnitude.
     * \param power mode power
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    DataVector<double> computeFieldMagnitude(double power, const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method);

    /**
     * Get electric field at the given mesh for resonant mode.
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    DataVector<Vec<3,dcomplex>> getFieldE(const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method) {
        determineFields();
        return computeFieldE(dst_mesh, method);
    }

    /**
     * Get magnetic field at the given mesh for resonant mode.
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    DataVector<Vec<3,dcomplex>> getFieldH(const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method) {
        determineFields();
        return computeFieldH(dst_mesh, method);
    }

    /**
     * Get light magnitude for resonant mode.
     * \param power mode power
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    DataVector<double> getFieldMagnitude(double power, const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method) {
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
    DataVector<Vec<3,dcomplex>> getReflectedFieldE(const cvector& incident, IncidentDirection side, const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method) {
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
    DataVector<Vec<3,dcomplex>> getReflectedFieldH(const cvector& incident, IncidentDirection side, const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method) {
        determineReflectedFields(incident, side);
        return computeFieldH(dst_mesh, method);
    }

    /**
     * Get light magnitude for reflected light.
     * \param incident incident field vector
     * \param side incidence side
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    DataVector<double> getReflectedFieldMagnitude(const cvector& incident, IncidentDirection side, const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method) {
        determineReflectedFields(incident, side);
        return computeFieldMagnitude(1., dst_mesh, method);
    }
};


}}} // namespace plask::solvers::slab

#endif // PLASK__SOLVER_SLAB_REFLECTIONBASE_H
