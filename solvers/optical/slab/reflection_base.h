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
    cmatrix M;                                  ///< The final matrix which must fulfill M * E = 0

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

    Data2DLog<dcomplex,dcomplex> detlog;        ///< Determinant logger

  private:

    cdiagonal phas;                             ///< current phase shift matrix
    int* ipiv;                                  ///< pivot vector
    std::vector<cmatrix> memP;                  ///< reflection matrices for each layer

  public:

    bool emitting;                              ///< \c True if the structure is emitting vertically.

    ~ReflectionSolver();

    /// Get currently searched variable
    Variable getVariable() const { return variable; }
    /// Set new variable to searched
    /// \param var new variable
    void setVariable(Variable var) {
        variable = var;
        detlog.resetCounter();
        switch (var) {
            case K_0: detlog.axis_arg_name = "k0"; return;
            case K_LONG: detlog.axis_arg_name = "beta"; return;
            case K_TRAN: detlog.axis_arg_name = "k"; return;
        }
    }

    /// Get current wavelength
    dcomplex getWavelength() const { return 2e3*M_PI / k0; }
    /// Set current wavelength
    void setWavelength(dcomplex lambda) {
        k0 = 2e3*M_PI / lambda;
        // this->invalidate();
    }

    /// Get current k0
    dcomplex getK0() const { return k0; }
    /// Set current k0
    void setK0(dcomplex k) {
        k0 = k;
        // this->invalidate();
    }

    /// Get longitudinal wavevector
    dcomplex getKlong() const { return klong; }
    /// Set longitudinal wavevector
    void setKlong(dcomplex k)  {
        klong = k; 
        // this->invalidate();
    }

    /// Get transverse wavevector
    dcomplex getKtran() const { return ktran; }
    /// Set transverse wavevector
    void setKtran(dcomplex k)  {
        ktran = k; 
        // this->invalidate();
    }

    /// Get discontinuity matrix determinant for the current parameters
    dcomplex getDeterminant() {
        this->initCalculation();
        return determinant();
    }
    
  protected:

    /// Solver constructor
    ReflectionSolver(const std::string& name): SlabSolver<GeometryT>(name),
        interface_field(nullptr), evals(nullptr), rwork(nullptr), work(nullptr),
        k0(NAN), klong(0.), ktran(0.), variable(K_0), detlog("", "modal", "k0", "det"),
        ipiv(nullptr) {}

    /// Initialize memory for calculations
    void init();

    /// Compute discontinuity matrix determinant for the current parameters
    dcomplex determinant();

    /// Get admittance (A) and discontinuity (M) matrices for the whole structure
    void getFinalMatrix() {
        getAM(this->stack.size()-1, this->interface+1, false);
        getAM(0, this->interface, true);
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
    void findReflection(size_t start, size_t end);

    /**
     * Store P matrix if we want it for field compuation
     * \param n layer number
     */
    void storeP(size_t n);


};


}}} // namespace plask::solvers::slab

#endif // PLASK__SOLVER_SLAB_REFLECTIONBASE_H
