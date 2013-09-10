#ifndef PLASK__SOLVER_SLAB_EXPANSION_PW2D_H
#define PLASK__SOLVER_SLAB_EXPANSION_PW2D_H

#include <plask/plask.hpp>

#include "expansion.h"
#include "fft.h"

namespace plask { namespace solvers { namespace slab {

struct FourierReflection2D;

struct ExpansionPW2D: public Expansion {

    /// Type of the mode symmetry
    enum Symmetry {
        SYMMETRIC_E_TRAN,               ///< E_tran and H_long are symmetric and E_long and H_tran anti-symmetric
        SYMMETRIC_E_LONG                ///< E_long and H_tran are symmetric and E_tran and H_long anti-symmetric
    };
    
    FourierReflection2D* solver;        ///< Solver which performs calculations (and is the interface to the outside world)

    RegularAxis xmesh;                  ///< Horizontal axis for structure sampling
    RegularAxis xpoints;                ///< Horizontal points in which fields will be computed by the inverse FFT

    size_t N;                           ///< Number of expansion coefficients
    size_t nN;                          ///< Number of of required coefficients for material parameters
    double left;                        ///< Left side of the sampled area
    double right;                       ///< Right side of the sampled area
    bool symmetric;                     ///< Indicates if the expansion is a symmetric one
    bool periodic;                      ///< Indicates if the geometry is periodic (otherwise use PMLs)

    size_t pil,                         ///< Index of the beginning of the left PML
           pir;                         ///< Index of the beginning of the right PML

    /**
     * Create new expansion
     * \param solver solver which performs calculations
     * \param allow_symmetry \c true if expansion may be symmetric (i.e. \f$ k_x = 0 \f$)
     */
    ExpansionPW2D(FourierReflection2D* solver, bool allow_symmetry=true);

    virtual size_t lcount() const;

    virtual bool diagonalQE(size_t l) const;

    virtual size_t matrixSize() const;

    virtual cmatrix getRE(size_t l, dcomplex k0, dcomplex beta, dcomplex kx);

    virtual cmatrix getRH(size_t l, dcomplex k0, dcomplex beta, dcomplex kx);

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
    
    /// Get \f$ \varepsilon_{zz} \f$
    dcomplex epszz(int i) { return coeffs[(i>=0)? i : i+nN].c00; }
    
    /// Get \f$ \varepsilon_{xx} \f$
    dcomplex epsxx(int i) { return coeffs[(i>=0)? i : i+nN].c11; }
    
    /// Get \f$ \varepsilon_{yy}^{-1} \f$
    dcomplex iepsyy(int i) { return coeffs[(i>=0)? i : i+nN].c22; }
    
    /// Get \f$ \varepsilon_{zx} \f$
    dcomplex epszx(int i) { return coeffs[(i>=0)? i : i+nN].c01; }
    
    /// Get \f$ \varepsilon_{xz} \f$
    dcomplex epsxz(int i) { return coeffs[(i>=0)? i : i+nN].c10; }
    
    /// Get \f$ \mu_{xx} \f$
    dcomplex muzz(int i) { return mag[(i>=0)? i : i+nN].c00; }
    
    /// Get \f$ \mu_{xx} \f$
    dcomplex muxx(int i) { return mag[(i>=0)? i : i+nN].c00; }
    
    /// Get \f$ \mu_{xx} \f$
    dcomplex imuyy(int i) { return mag[(i>=0)? i : i+nN].c11; }
};

}}} // namespace plask

#endif // PLASK__SOLVER_SLAB_EXPANSION_PW2D_H
