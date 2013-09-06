#ifndef PLASK__SOLVER_SLAB_EXPANSION_PW2D_H
#define PLASK__SOLVER_SLAB_EXPANSION_PW2D_H

#include <plask/plask.hpp>

#include "expansion.h"
#include "fft.h"

namespace plask { namespace solvers { namespace slab {

struct FourierReflection2D;

struct ExpansionPW2D: public Expansion {

    FourierReflection2D* solver;        ///< Solver which performs calculations (and is the interface to the outside world)

    RegularAxis xmesh;                  ///< Horizontal axis for structure sampling
    RegularAxis xpoints;                ///< Horizontal points in which fields will be computed by the inverse FFT

    size_t N;                           ///< Number of expansion coefficients
    size_t nN;                          ///< Number of of required coefficients for material parameters
    double left;                        ///< Left side of the sampled area
    double right;                       ///< Right side of the sampled area
    bool symmetric;                     ///< Indicates if the expansion is a symmetric one
    bool periodic;                      ///< Indicates if the geometry is periodic (otherwise use PMLs)

    size_t pil,                         ///< Index of the beginnig of the left PML
           pir;                         ///< Index of the beginnig of the right PML

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

    /**
     * Get refractive index back from expansion
     * \param l layer number
     * \param mesh mesh to get parameters to
     * \param interp interpolation method
     * \return computed refractive indices
     */
    DataVector<const Tensor3<dcomplex>> getMaterialNR(size_t l, const RectilinearAxis mesh,
                                                      InterpolationMethod interp=INTERPOLATION_DEFAULT);

  protected:

    DataVector<Tensor2<dcomplex>> mag;      ///< Magnetic permeability coefficients (used with for PMLs)
    DataVector<Tensor3<dcomplex>> coeffs;   ///< Material coefficients

    FFT::Forward1D matFFT;                  ///< FFT object for material coeffictiens

    /**
     * Compute expansion coefficients for material parameters
     * \param l layer number
     */
    void getMaterialCoefficients(size_t l);
};

}}} // namespace plask

#endif // PLASK__SOLVER_SLAB_EXPANSION_PW2D_H
