#ifndef PLASK__SOLVER_SLAB_EXPANSION_PW2D_H
#define PLASK__SOLVER_SLAB_EXPANSION_PW2D_H

#include <plask/plask.hpp>

#include "../expansion.h"
#include "../meshadapter.h"
#include "fft.h"

namespace plask { namespace solvers { namespace slab {

struct FourierSolver2D;

struct PLASK_SOLVER_API ExpansionPW2D: public Expansion {

    RegularAxis xmesh;                  ///< Horizontal axis for structure sampling

    size_t N;                           ///< Number of expansion coefficients
    size_t nN;                          ///< Number of of required coefficients for material parameters
    size_t nM;                          ///< Number of FFT coefficients
    double left;                        ///< Left side of the sampled area
    double right;                       ///< Right side of the sampled area
    bool periodic;                      ///< Indicates if the geometry is periodic (otherwise use PMLs)
    bool initialized;                   ///< Expansion is initialized

    Component symmetry;                 ///< Indicates symmetry if `symmetric`
    Component polarization;             ///< Indicates polarization if `separated`

    size_t pil,                         ///< Index of the beginning of the left PML
           pir;                         ///< Index of the beginning of the right PML

    /// Cached permittivity expansion coefficients
    std::vector<DataVector<Tensor3<dcomplex>>> coeffs;

    /// Information if the layer is diagonal
    std::vector<bool> diagonals;

    /**
     * Create new expansion
     * \param solver solver which performs calculations
     */
    ExpansionPW2D(FourierSolver2D* solver);

    /// Indicates if the expansion is a symmetric one
    bool symmetric() const { return symmetry != E_UNSPECIFIED; }

    /// Indicates whether TE and TM modes can be separated
    bool separated() const { return polarization != E_UNSPECIFIED; }

    /**
     * Init expansion
     * \param compute_coeffs compute material coefficients
     */
    void init();

    /// Free allocated memory
    void reset();

    /// Compute all expansion coefficients
    void computeMaterialCoefficients() {
        size_t nlayers = lcount();
        assert(coeffs.size() == nlayers);
        for (size_t l = 0; l < nlayers; ++l)
            layerMaterialCoefficients(l);
    }

    virtual size_t lcount() const override;

    virtual bool diagonalQE(size_t l) const override {
        return diagonals[l];
    }

    size_t matrixSize() const override { return separated()? N : 2*N; }

    void getMatrices(size_t l, cmatrix& RE, cmatrix& RH) override;

    void prepareField() override;

    void cleanupField() override;

    DataVector<const Vec<3,dcomplex>> getField(size_t l,
                                               const shared_ptr<const typename LevelsAdapter::Level>& level,
                                               const cvector& E, const cvector& H) override;

    LazyData<Tensor3<dcomplex>> getMaterialNR(size_t l,
                                              const shared_ptr<const typename LevelsAdapter::Level>& level,
                                              InterpolationMethod interp=INTERPOLATION_DEFAULT) override;

  private:

    DataVector<Vec<3,dcomplex>> field;
    FFT::Backward1D fft_x, fft_yz;

  protected:

    DataVector<Tensor2<dcomplex>> mag;      ///< Magnetic permeability coefficients (used with for PMLs)

    FFT::Forward1D matFFT;                  ///< FFT object for material coefficients

    /**
     * Compute expansion coefficients for material parameters
     * \param l layer number
     */
    void layerMaterialCoefficients(size_t l);

  public:

    /// Get \f$ \varepsilon_{zz} \f$
    dcomplex epszz(size_t l, int i) { return coeffs[l][(i>=0)?i:i+nN].c00; }

    /// Get \f$ \varepsilon_{xx} \f$
    dcomplex epsxx(size_t l, int i) { return coeffs[l][(i>=0)?i:i+nN].c11; }

    /// Get \f$ \varepsilon_{yy}^{-1} \f$
    dcomplex iepsyy(size_t l, int i) { return coeffs[l][(i>=0)?i:i+nN].c22; }

    /// Get \f$ \varepsilon_{zx} \f$
    dcomplex epszx(size_t l, int i) { return coeffs[l][(i>=0)?i:i+nN].c01; }

    /// Get \f$ \varepsilon_{xz} \f$
    dcomplex epsxz(size_t l, int i) { return conj(coeffs[l][(i>=0)?i:i+nN].c01); }

    /// Get \f$ \mu_{zz} \f$
    dcomplex muzz(size_t l, int i) { return mag[(i>=0)?i:i+nN].c00; }

    /// Get \f$ \mu_{xx} \f$
    dcomplex muxx(size_t l, int i) { return mag[(i>=0)?i:i+nN].c11; }

    /// Get \f$ \mu_{yy}^{-1} \f$
    dcomplex imuyy(size_t l, int i) { return mag[(i>=0)?i:i+nN].c11; }

    /// Get \f$ E_x \f$ index
    size_t iEx(int i) { return 2 * ((i>=0)?i:i+N); }

    /// Get \f$ E_z \f$ index
    size_t iEz(int i) { return 2 * ((i>=0)?i:i+N) + 1; }

    /// Get \f$ H_x \f$ index
    size_t iHx(int i) { return 2 * ((i>=0)?i:i+N) + 1; }

    /// Get \f$ H_z \f$ index
    size_t iHz(int i) { return 2 * ((i>=0)?i:i+N); }

    /// Get \f$ E \f$ index for separated equations
    size_t iE(int i) { return (i>=0)?i:i+N; }

    /// Get \f$ H \f$ index for separated equations
    size_t iH(int i) { return (i>=0)?i:i+N; }
};

}}} // namespace plask

#endif // PLASK__SOLVER_SLAB_EXPANSION_PW2D_H
