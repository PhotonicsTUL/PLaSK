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

    cvector getReflectionVector(const cvector& incident, IncidentDirection direction);

    cvector getTransmissionVector(const cvector& incident, IncidentDirection side);

    cvector getInterfaceVector();

    cvector getFieldVectorE(double z, int n);

    cvector getFieldVectorH(double z, int n);

    ReflectionTransfer(SlabBase* solver, Expansion& expansion);

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
     * \param emitting should the reflection matrix in the first layer be 0?
     */
    void findReflection(int start, int end, bool emitting);

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
     * \param emitting is the field emitting?
     */
    DataVector<Vec<3,dcomplex>> computeFieldE(const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method, bool emitting);

    /**
     * Compute magnetic field at the given mesh.
     * \param dst_mesh target mesh
     * \param method interpolation method
     * \param emitting is the field emitting?
     */
    DataVector<Vec<3,dcomplex>> computeFieldH(const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method, bool emitting);

    /**
     * Compute light magnitude.
     * \param power mode power
     * \param dst_mesh destination mesh
     * \param method interpolation method
     * \param emitting is the field emitting?
     */
    DataVector<double> computeFieldMagnitude(double power, const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method, bool emitting);

    DataVector<Vec<3,dcomplex>> getFieldE(const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method) {
        determineFields();
        return computeFieldE(dst_mesh, method, false);
    }

    DataVector<Vec<3,dcomplex>> getFieldH(const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method) {
        determineFields();
        return computeFieldH(dst_mesh, method, false);
    }

    DataVector<double> getFieldMagnitude(double power, const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method) {
        determineFields();
        return computeFieldMagnitude(power, dst_mesh, method, false);
    }

    DataVector<Vec<3,dcomplex>> getReflectedFieldE(const cvector& incident, IncidentDirection side, const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method) {
        determineReflectedFields(incident, side);
        return computeFieldE(dst_mesh, method, true);
    }

    DataVector<Vec<3,dcomplex>> getReflectedFieldH(const cvector& incident, IncidentDirection side, const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method) {
        determineReflectedFields(incident, side);
        return computeFieldH(dst_mesh, method, true);
    }

    DataVector<double> getReflectedFieldMagnitude(const cvector& incident, IncidentDirection side, const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method) {
        determineReflectedFields(incident, side);
        return computeFieldMagnitude(1., dst_mesh, method, true);
    }
};


}}} // namespace plask::solvers::slab

#endif // PLASK__SOLVER_SLAB_REFLECTIONBASE_H
