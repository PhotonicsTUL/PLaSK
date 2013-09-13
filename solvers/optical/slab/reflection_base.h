#ifndef PLASK__SOLVER_SLAB_REFLECTIONBASE_H
#define PLASK__SOLVER_SLAB_REFLECTIONBASE_H

#include "matrices.h"
#include "diagonalizer.h"
#include "slab_base.h"

namespace plask { namespace solvers { namespace slab {

template <typename GeometryT>
class ReflectionSolver: public SlabSolver<GeometryT> {

  protected:

    enum Variable { K_0, K_TRAN, K_LONG };      ///< Possible variables to dig for

    std::unique_ptr<Diagonalizer> diagonalizer; ///< Diagonalizer used to compute matrix of eigenvalues and eigenvectors

    cmatrix interface_field_matrix;             ///< Determined field at the interface
    dcomplex* interface_field;                  ///< Pointer to the interface field data

    cmatrix A;                                  ///< The (diagonalized field) admittance matrix for the interface
    cmatrix M;                                  ///< The final matrix0 which must fulfill M * E = 0

    dcomplex* evals;                            ///< Found eigenvalues of matrix M
    double* rwork;                              ///< temporary space
    int lwork;                                  ///< temporary space
    dcomplex* work;                             ///< temporary space

    dcomplex k0,                                ///< Normalized frequency [1/µm]
             klong,                             ///< Longitudinal wavevector [1/µm]
             ktran;                             ///< Transverse wavevector [1/µm]

    Variable variable;                          ///< Which variable to dig for

    cmatrix P;                                  ///< current reflection matrix
    bool allP;                                  ///< do we need to keep all the P matrices?

    bool fields_determined;                     ///< Are the diagonalized fields determined for all layers?

  private:

    cdiagonal phas;                             ///< current phase shift matrix
    int* ipiv;                                  ///< pivot vector
    std::vector<cmatrix> memP;                  ///< reflection matrices for each layer

  public:

    bool emitting;                              ///< \c True if the structure is emitting vertically.

    ~ReflectionSolver();

  protected:

    ReflectionSolver(const std::string& name): SlabSolver<GeometryT>(name),
        k0(NAN), klong(0.), ktran(0.), variable(K_0) {}

    void init();

};


}}} // namespace plask::solvers::slab

#endif // PLASK__SOLVER_SLAB_REFLECTIONBASE_H
