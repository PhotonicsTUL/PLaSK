#ifndef PLASK__SOLVER_SLAB_EXPANSION_PW2D_H
#define PLASK__SOLVER_SLAB_EXPANSION_PW2D_H

#include <plask/plask.hpp>

#include "expansion.h"
#include "reflection_solver_2d.h"
#include "fft.h"

namespace plask { namespace solvers { namespace slab {

struct ExpansionPW2D: public Expansion {

    FourierReflection2D* solver;        ///< Solver which performs calculations (and is the interface to the outside world)

    RegularMesh1D xmesh;                ///< Horizontal axis for structure sampling

    size_t N;                           ///< Number of expansion coefficients
    size_t nN;                          ///< Number of of required coefficients for material parameters
    double left;                        ///< Left side of the sampled area
    double right;                       ///< Right side of the sampled area
    bool symmetric;                     ///< Indicates if the expansion is a symmetric one
    bool periodic;                      ///< Indicates if the geometry is periodic (otherwise use PMLs)

    size_t pil,                         ///< Index of the beginnig of the left PML
           pir;                         ///< Index of the beginnig of the right PML

    DataVector<Tensor2<dcomplex>> mag;  ///< Magnetic permeability coefficients (used with for PMLs)

    FFT fft;                            ///< FFT object

    /**
     * Create new expansion
     * \param solver solver which performs calculations
     */
    ExpansionPW2D(FourierReflection2D* solver);

    virtual size_t lcount() const;

    virtual bool diagonalQE(size_t l) const;

    virtual size_t matrixSize() const;

    virtual cmatrix getRE(size_t l, dcomplex k0, dcomplex kx, dcomplex ky);

    virtual cmatrix getRH(size_t l, dcomplex k0, dcomplex kx, dcomplex ky);

  protected:

    /**
     * Compute expansion coefficients for material parameters
     * \param l layer number
     */
    DataVector<const Tensor3<dcomplex>> getMaterialParameters(size_t l);

};

}}} // namespace plask

#endif // PLASK__SOLVER_SLAB_EXPANSION_PW2D_H
