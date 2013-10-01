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

    /// Polarization of separated modes
    enum Polarization {
        TE,                             ///< E_z and H_x exist
        TM                              ///< H_z and E_x exist
    };

    RegularAxis xmesh;                  ///< Horizontal axis for structure sampling
    RegularAxis xpoints;                ///< Horizontal points in which fields will be computed by the inverse FFT

    size_t N;                           ///< Number of expansion coefficients
    size_t nN;                          ///< Number of of required coefficients for material parameters
    double left;                        ///< Left side of the sampled area
    double right;                       ///< Right side of the sampled area
    bool symmetric;                     ///< Indicates if the expansion is a symmetric one
    bool periodic;                      ///< Indicates if the geometry is periodic (otherwise use PMLs)
    bool separated;                     ///< Indicates whether TE and TM modes can be separated

    Symmetry symmetry;                  ///< Indicates symmetry if `symmetric`
    Polarization polarization;          ///< Indicates polarization if `separated`

    size_t pil,                         ///< Index of the beginning of the left PML
           pir;                         ///< Index of the beginning of the right PML

    /**
     * Create new expansion
     * \param solver solver which performs calculations
     */
    ExpansionPW2D(FourierReflection2D* solver);

    /**
     * Init expansion
     * \param long_zero \c true if \f$ k_z = 0 \f$)
     * \param tran_zero \c true if \f$ k_x = 0 \f$)
     */
    void init(bool long_zero, bool tran_zero);

    virtual size_t lcount() const;

    virtual bool diagonalQE(size_t l) const;

    virtual size_t matrixSize() const { return separated? N : 2*N; }

    virtual void getMatrices(size_t l, dcomplex k0, dcomplex beta, dcomplex kx, cmatrix& RE, cmatrix& RH);

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

    FFT::Forward1D matFFT;                  ///< FFT object for material coeffictiens

    /**
     * Compute expansion coefficients for material parameters
     * \param l layer number
     * \return material coeffcients
     */
    DataVector<Tensor3<dcomplex>> getMaterialCoefficients(size_t l);

    /// Get \f$ \varepsilon_{zz} \f$
    dcomplex epszz(const DataVector<Tensor3<dcomplex>>& coeffs, int i) { return coeffs[(i>=0)?i:i+nN].c00; }

    /// Get \f$ \varepsilon_{xx} \f$
    dcomplex epsxx(const DataVector<Tensor3<dcomplex>>& coeffs, int i) { return coeffs[(i>=0)?i:i+nN].c11; }

    /// Get \f$ \varepsilon_{yy}^{-1} \f$
    dcomplex iepsyy(const DataVector<Tensor3<dcomplex>>& coeffs, int i) { return coeffs[(i>=0)?i:i+nN].c22; }

    /// Get \f$ \varepsilon_{zx} \f$
    dcomplex epszx(const DataVector<Tensor3<dcomplex>>& coeffs, int i) { return coeffs[(i>=0)?i:i+nN].c01; }

    /// Get \f$ \varepsilon_{xz} \f$
    dcomplex epsxz(const DataVector<Tensor3<dcomplex>>& coeffs, int i) { return coeffs[(i>=0)?i:i+nN].c10; }

    /// Get \f$ \mu_{xx} \f$
    dcomplex muzz(const DataVector<Tensor3<dcomplex>>& coeffs, int i) { return mag[(i>=0)?i:i+nN].c00; }

    /// Get \f$ \mu_{xx} \f$
    dcomplex muxx(const DataVector<Tensor3<dcomplex>>& coeffs, int i) { return mag[(i>=0)?i:i+nN].c00; }

    /// Get \f$ \mu_{xx} \f$
    dcomplex imuyy(const DataVector<Tensor3<dcomplex>>& coeffs, int i) { return mag[(i>=0)?i:i+nN].c11; }

    /// Get \f$ E_x \f$ index
    size_t iEx(int i) { return 2 * ((i>=0)?i:i+N); }

    /// Get \f$ E_x \f$ index
    size_t iEz(int i) { return 2 * ((i>=0)?i:i+N) + 1; }

    /// Get \f$ E_x \f$ index
    size_t iHx(int i) { return 2 * ((i>=0)?i:i+N) + 1; }

    /// Get \f$ E_x \f$ index
    size_t iHz(int i) { return 2 * ((i>=0)?i:i+N); }

    /// Get \f$ E_x \f$ index
    size_t iE(int i) { return (i>=0)?i:i+N; }

    /// Get \f$ E_x \f$ index
    size_t iH(int i) { return (i>=0)?i:i+N; }
};

}}} // namespace plask

#endif // PLASK__SOLVER_SLAB_EXPANSION_PW2D_H
