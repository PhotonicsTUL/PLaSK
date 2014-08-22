#ifndef PLASK__SOLVER_SLAB_EXPANSION_PW3D_H
#define PLASK__SOLVER_SLAB_EXPANSION_PW3D_H

#include <plask/plask.hpp>

#include "../expansion.h"
#include "fft.h"

namespace plask { namespace solvers { namespace slab {

struct FourierSolver3D;

struct PLASK_SOLVER_API ExpansionPW3D: public Expansion {

    RegularAxis long_mesh,              ///< Horizontal axis for structure sampling in longitudinal direction
                tran_mesh;              ///< Horizontal axis for structure sampling in transverse direction

    size_t Nl,                          ///< Number of expansion coefficients in longitudinal direction
           Nt;                          ///< Number of expansion coefficients in transverse direction
    size_t nNl,                         ///< Number of of required coefficients for material parameters in longitudinal direction
           nNt;                         ///< Number of of required coefficients for material parameters in transverse direction

    double left;                        ///< Left side of the sampled area
    double right;                       ///< Right side of the sampled area
    double back;                        ///< Back side of the sampled area
    double front;                       ///< Front side of the sampled area
    bool periodic_long,                 ///< Indicates if the geometry is periodic (otherwise use PMLs) in longitudinal direction
         periodic_tran;                 ///< Indicates if the geometry is periodic (otherwise use PMLs) in transverse direction
    bool initialized;                   ///< Expansion is initialized

    Component symmetry_long,            ///< Indicates symmetry if `symmetric` in longitudinal direction
              symmetry_tran;            ///< Indicates symmetry if `symmetric` in transverse direction

    size_t pil,                         ///< Index of the beginning of the left PML
           pir,                         ///< Index of the beginning of the right PML
           pif,                         ///< Index of the beginning of the front PML
           pib;                         ///< Index of the beginning of the back PML

    /// Cached permittivity expansion coefficients
    std::vector<DataVector<Tensor3<dcomplex>>> coeffs;

    /// Information if the layer is diagonal
    std::vector<bool> diagonals;

    /**
     * Create new expansion
     * \param solver solver which performs calculations
     */
    ExpansionPW3D(FourierSolver3D* solver);

    /// Indicates if the expansion is a symmetric one in longitudinal direction
    bool symmetric_long() const { return symmetry_long != E_UNSPECIFIED; }

    /// Indicates if the expansion is a symmetric one in transverse direction
    bool symmetric_tran() const { return symmetry_tran != E_UNSPECIFIED; }

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
        std::exception_ptr error;
        #pragma omp parallel for
        for (size_t l = 0; l < nlayers; ++l) {
            if (error) continue;
            try {
                layerMaterialCoefficients(l);
            } catch(...) {
                #pragma omp critical
                error = std::current_exception();
            }
        }
        if (error) std::rethrow_exception(error);
    }

    virtual size_t lcount() const;

    virtual bool diagonalQE(size_t l) const {
        return diagonals[l];
    }

    size_t matrixSize() const override { return 2*Nl*Nt; }

    void getMatrices(size_t l, dcomplex k0, dcomplex klong, dcomplex ktran, cmatrix& RE, cmatrix& RH) override;

    void prepareField() override;

    void cleanupField() override;

    DataVector<const Vec<3,dcomplex>> getField(size_t l,
                                               const shared_ptr<const typename LevelsAdapter::Level>& level,
                                               const cvector& E, const cvector& H) override;

    LazyData<Tensor3<dcomplex>> getMaterialNR(size_t lay,
                                              const shared_ptr<const typename LevelsAdapter::Level>& level,
                                              InterpolationMethod interp) override;

  private:

    DataVector<Vec<3,dcomplex>> field;
    FFT::Backward2D fft_x, fft_y, fft_z;

  protected:

    DataVector<Tensor2<dcomplex>> mag_long; ///< Magnetic permeability coefficients in longitudinal direction (used with for PMLs)
    DataVector<Tensor2<dcomplex>> mag_tran; ///< Magnetic permeability coefficients in transverse direction (used with for PMLs)

    FFT::Forward2D matFFT;                  ///< FFT object for material coefficients

    /**
     * Compute expansion coefficients for material parameters
     * \param l layer number
     */
    void layerMaterialCoefficients(size_t l);

  public:

    /// Get \f$ \varepsilon_{xx} \f$
    dcomplex epsxx(size_t lay, int l, int t) {
        if (l < 0) l += nNl; if (t < 0) t += nNt;
        return coeffs[lay][nNl * t + l].c00;
    }

    /// Get \f$ \varepsilon_{yy} \f$
    dcomplex epsyy(size_t lay, int l, int t) {
        if (l < 0) l += nNl; if (t < 0) t += nNt;
        return coeffs[lay][nNl * t + l].c11;
    }

    /// Get \f$ \varepsilon_{zz}^{-1} \f$
    dcomplex iepszz(size_t lay, int l, int t) {
        if (l < 0) l += nNl; if (t < 0) t += nNt;
        return coeffs[lay][nNl * t + l].c22;
    }

    /// Get \f$ \varepsilon_{xy} \f$
    dcomplex epsxy(size_t lay, int l, int t) {
        if (l < 0) l += nNl; if (t < 0) t += nNt;
        return coeffs[lay][nNl * t + l].c01;
    }

    /// Get \f$ \varepsilon_{yx} \f$
    dcomplex epsyx(size_t lay, int l, int t) { return conj(epsxy(lay, l, t)); }

    /// Get \f$ \mu_{xx} \f$
    dcomplex muxx(size_t lay, int l, int t) { return mag_long[(l>=0)?l:l+nNl].c11 * mag_tran[(t>=0)?t:t+nNt].c00; }

    /// Get \f$ \mu_{yy} \f$
    dcomplex muyy(size_t lay, int l, int t) { return mag_long[(l>=0)?l:l+nNl].c00 * mag_tran[(t>=0)?t:t+nNt].c11; }

    /// Get \f$ \mu_{zz}^{-1} \f$
    dcomplex imuzz(size_t lay, int l, int t) { return mag_long[(l>=0)?l:l+nNl].c11 * mag_tran[(t>=0)?t:t+nNt].c11; }

    /// Get \f$ E_x \f$ index
    size_t iEx(int l, int t) {
        if (l < 0) { if (symmetric_long()) l = -l; else l += Nl; }
        if (t < 0) { if (symmetric_tran()) t = -t; else t += Nt; }
        assert(0 <= l && l < Nl);
        assert(0 <= t && t < Nt);
        return 2 * (Nl*t + l);
    }

    /// Get \f$ E_y \f$ index
    size_t iEy(int l, int t) {
        if (l < 0) { if (symmetric_long()) l = -l; else l += Nl; }
        if (t < 0) { if (symmetric_tran()) t = -t; else t += Nt; }
        assert(0 <= l && l < Nl);
        assert(0 <= t && t < Nt);
        return 2 * (Nl*t + l) + 1;
    }

    /// Get \f$ H_x \f$ index
    size_t iHx(int l, int t) {
        if (l < 0) { if (symmetric_long()) l = -l; else l += Nl; }
        if (t < 0) { if (symmetric_tran()) t = -t; else t += Nt; }
        assert(0 <= l && l < Nl);
        assert(0 <= t && t < Nt);
        return 2 * (Nl*t + l) + 1;
    }

    /// Get \f$ H_y \f$ index
    size_t iHy(int l, int t) {
        if (l < 0) { if (symmetric_long()) l = -l; else l += Nl; }
        if (t < 0) { if (symmetric_tran()) t = -t; else t += Nt; }
        assert(0 <= l && l < Nl);
        assert(0 <= t && t < Nt);
        return 2 * (Nl*t + l);
    }
};

}}} // namespace plask

#endif // PLASK__SOLVER_SLAB_EXPANSION_PW3D_H
