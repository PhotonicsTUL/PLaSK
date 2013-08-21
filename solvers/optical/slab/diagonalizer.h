/**
 *  \file   diagonalizer.h  Module responsible for calculating and holding diagonalized matrices
 */
#ifndef PLASK__SOLVER_VSLAB_DIAGONALIZER_H
#define PLASK__SOLVER_VSLAB_DIAGONALIZER_H

#include <utility>

#include <plask/plask.hpp>

#include "matrices.h"

namespace plask { namespace  solvers { namespace slab {

//TODO
struct GridBase {
    size_t lcount;
    bool diagonalQE(size_t i) {}
    size_t matrixSize() {}
    template <typename... Cls> cmatrix getRE(Cls...) {}
    template <typename... Cls> cmatrix getRH(Cls...) {}
};

/** 
 * Base for the class determining and holding the necessary matrices
 * This is the abstract base class for all diagonalizers (multi-threaded,
 * disk-storage, MPI-using etc.
 * This class should compute gamma, Te and Th matrices for each layer and
 * store it until the next initDiagonalization(...) is called or the
 * object is destroyed (this is necessary for computation of the fields
 * after the convergence).
 */
class DiagonalizerBase
{
  protected:
    GridBase& grid;                  // information about the grid
    std::vector<bool> diagonalized;  // true if the given layer was diagonalized

  public:
    const int lcount;                // number of layers

    DiagonalizerBase(GridBase& g) :
        grid(g), lcount(g.lcount), diagonalized(g.lcount, false) {}

    virtual ~DiagonalizerBase() {}

    /// Return the overall matrix size
    virtual int matrixSize() const = 0;

    /// Return the reference to the grid object
    inline GridBase& getGrid() { return (GridBase&)(grid); }

    /// Initiate the diagonalization
    virtual void initDiagonalization(dcomplex ko, dcomplex kx, dcomplex ky, double mgain) = 0;

    // Calculate the diagonalization of given layer
    virtual void diagonalizeLayer(int layer) = 0;

    // Functions returning references to calculated matrices
    virtual const cdiagonal& Gamma(int layer) const = 0;
    virtual const cmatrix& TE(int layer) const = 0;
    virtual const cmatrix& TH(int layer) const = 0;
    virtual const cmatrix& invTE(int layer) const = 0;
    virtual const cmatrix& invTH(int layer) const = 0;

    // Diagonalization function to compute the smallest eigenvalue of the provided matrix
//     virtual dcomplex smallest_eigevalue(const cmatrix& M) = 0;
};


/** 
 * Simple diagonalizer
 * This class is a simple diagonalizer. It calculates all its results
 * immidiatelly and stores them in the memory.
 */
class SimpleDiagonalizer : public DiagonalizerBase
{
  protected:
    dcomplex k0;                        // The frequency for which we compute the diagonalization for all layers
    dcomplex Kx, Ky;                    // The wavevector for which we compute the diagonalization for all layers
    double matgain;                     // Material gain when threshold is searched

    std::vector<cdiagonal> gamma;       // diagonal matrices Gamma
    std::vector<cmatrix> Te, Th;        // matrices EE and EH
    std::vector<cmatrix> Te1, Th1;      // matrices TE^-1 anf TE^-1

    cmatrix QE;                         // temporary matrix to store QH = RE * RH
    cmatrix tmp;                        // some temporary matrix

  public:
    SimpleDiagonalizer(GridBase& g);
    ~SimpleDiagonalizer();

    /// Return the overall matrix size
    virtual int matrixSize() const { return grid.matrixSize(); }

    /// Initiate the diagonalization
    virtual void initDiagonalization(dcomplex ko, dcomplex kx, dcomplex ky, double mgain) {
        k0 = ko; Kx = kx, Ky = ky;
        matgain = mgain;
        for (int i = 0; i < lcount; i++) diagonalized[i] = false;
    };

    // Calculate the diagonalization of given layer
    virtual void diagonalizeLayer(int layer);

    // Functions returning references to calculated matrices
    virtual const cdiagonal& Gamma(int layer) const { return gamma[layer]; }

    virtual const cmatrix& TE(int layer) const { return Te[layer]; }
    virtual const cmatrix& TH(int layer) const { return Th[layer]; }

    virtual const cmatrix& invTE(int layer) const { return Te1[layer]; }
    virtual const cmatrix& invTH(int layer) const { return Th1[layer]; }
};

}}} // namespace plask::solvers::slab
#endif // PLASK__SOLVER_VSLAB_DIAGONALIZER_H
