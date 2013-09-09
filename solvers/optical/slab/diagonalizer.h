/**
 *  \file   diagonalizer.h  Module responsible for calculating and holding diagonalized matrices
 */
#ifndef PLASK__SOLVER_SLAB_DIAGONALIZER_H
#define PLASK__SOLVER_SLAB_DIAGONALIZER_H

#include <utility>

#include <plask/plask.hpp>

#include "matrices.h"
#include "expansion.h"

namespace plask { namespace  solvers { namespace slab {

/** 
 * Base for the class determining and holding the necessary matrices
 * This is the abstract base class for all diagonalizers (multi-threaded,
 * disk-storage, MPI-using etc.
 * This class should compute gamma, Te and Th matrices for each layer and
 * store it until the next initDiagonalization(...) is called or the
 * object is destroyed (this is necessary for computation of the fields
 * after the convergence).
 */
class Diagonalizer
{
  protected:
    Expansion& src;                 ///< Information about the matrices to diagonalize
    std::vector<bool> diagonalized; ///< True if the given layer was diagonalized

  public:
    const int lcount;                // number of layers

    Diagonalizer(Expansion& src) :
        src(src), diagonalized(src.lcount(), false), lcount(src.lcount()) {}

    virtual ~Diagonalizer() {}

    /// Return the overall matrix size
    virtual int matrixSize() const = 0;

    /// Return the reference to the source object
    inline const Expansion& source() const { return src; }

    /// Return the reference to the source object
    inline Expansion& source() { return src; }

    /// Initiate the diagonalization
    virtual void initDiagonalization(dcomplex ko, dcomplex kx, dcomplex ky) = 0;

    /// Calculate the diagonalization of given layer
    virtual void diagonalizeLayer(size_t layer) = 0;

    /// Return diagnoal matrix of eignevalues
    virtual const cdiagonal& Gamma(size_t layer) const = 0;
    
    /// Return matrix of eigenvectors of QE
    virtual const cmatrix& TE(size_t layer) const = 0;
    
    /// Return matrix of eigenvectors of QH
    virtual const cmatrix& TH(size_t layer) const = 0;
    
    /// Return inverse matrix of eigenvectors of QE
    virtual const cmatrix& invTE(size_t layer) const = 0;
    
    /// Return inverse matrix of eigenvectors of QH
    virtual const cmatrix& invTH(size_t layer) const = 0;

    // Diagonalization function to compute the smallest eigenvalue of the provided matrix
//     virtual dcomplex smallest_eigevalue(const cmatrix& M) = 0;
};


/** 
 * Simple diagonalizer
 * This class is a simple diagonalizer. It calculates all its results
 * immediately and stores them in the memory.
 */
class SimpleDiagonalizer : public Diagonalizer
{
  protected:
    dcomplex k0;                        // The frequency for which we compute the diagonalization for all layers
    dcomplex Kx, Ky;                    // The wavevector for which we compute the diagonalization for all layers

    std::vector<cdiagonal> gamma;       // diagonal matrices Gamma
    std::vector<cmatrix> Te, Th;        // matrices EE and EH
    std::vector<cmatrix> Te1, Th1;      // matrices TE^-1 anf TE^-1

    cmatrix QE;                         // temporary matrix to store QH = RE * RH
    cmatrix tmp;                        // some temporary matrix

  public:
    SimpleDiagonalizer(Expansion& g);
    ~SimpleDiagonalizer();

    virtual int matrixSize() const { return src.matrixSize(); }

    virtual void initDiagonalization(dcomplex ko, dcomplex kx, dcomplex ky) {
        k0 = ko; Kx = kx, Ky = ky;
        for (int i = 0; i < lcount; i++) diagonalized[i] = false;
    };

    virtual void diagonalizeLayer(size_t layer);

    // Functions returning references to calculated matrices
    virtual const cdiagonal& Gamma(size_t layer) const { return gamma[layer]; }
    virtual const cmatrix& TE(size_t layer) const { return Te[layer]; }
    virtual const cmatrix& TH(size_t layer) const { return Th[layer]; }
    virtual const cmatrix& invTE(size_t layer) const { return Te1[layer]; }
    virtual const cmatrix& invTH(size_t layer) const { return Th1[layer]; }
};

}}} // namespace plask::solvers::slab
#endif // PLASK__SOLVER_SLAB_DIAGONALIZER_H
