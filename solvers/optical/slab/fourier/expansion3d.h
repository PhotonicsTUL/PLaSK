#ifndef PLASK__SOLVER_SLAB_EXPANSION_PW3D_H
#define PLASK__SOLVER_SLAB_EXPANSION_PW3D_H

#include <plask/plask.hpp>

#include "../expansion.h"
#include "fft.h"

namespace plask { namespace solvers { namespace slab {

struct FourierSolver3D;

struct PLASK_SOLVER_API ExpansionPW3D: public Expansion {

    dcomplex klong,                     ///< Longitudinal wavevector
             ktran;                     ///< Transverse wavevector

    size_t N0,                          ///< Number of expansion coefficients along the first basis vector (longitudinal direction)
           N1;                          ///< Number of expansion coefficients along the second basis vector (transverse direction)
    size_t nN0,                         ///< Number of of required coefficients for material parameters along the first basis vector (longitudinal direction)
           nN1;                         ///< Number of of required coefficients for material parameters along the second basis vector (transverse direction)
    size_t nM0,                         ///< Number of FFT coefficients along the first basis vector (longitudinal direction)
           nM1;                         ///< Number of FFT coefficients along the second basis vector (transverse direction)

    Vec<3,double> vec0,                 ///< First basis vector
                  vec1;                 ///< Second basis vector
           
    double lo0;                         ///< Back side of the sampled area
    double hi0;                         ///< Front side of the sampled area
    double lo1;                         ///< Left side of the sampled area
    double hi1;                         ///< Right side of the sampled area

    bool periodic_long,                 ///< Indicates if the geometry is periodic (otherwise use PMLs) in longitudinal direction
         periodic_tran;                 ///< Indicates if the geometry is periodic (otherwise use PMLs) in transverse direction

    Component symmetry_long,            ///< Indicates symmetry if `symmetric` in longitudinal direction
              symmetry_tran;            ///< Indicates symmetry if `symmetric` in transverse direction

    size_t pil,                         ///< Index of the beginning of the left PML
           pir,                         ///< Index of the beginning of the right PML
           pif,                         ///< Index of the beginning of the front PML
           pib;                         ///< Index of the beginning of the back PML

    bool initialized;                   ///< Expansion is initialized

    /// Cached permittivity expansion coefficients
    std::vector<DataVector<Tensor3<dcomplex>>> coeffs;

    /// Information if the layer is diagonal
    std::vector<bool> diagonals;

    /// Mesh for getting material data
    shared_ptr<EquilateralMesh3D> mesh;

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

    bool diagonalQE(size_t l) const override {
        return diagonals[l];
    }

    size_t matrixSize() const override { return 2*N0*N1; }

    void getMatrices(size_t l, cmatrix& RE, cmatrix& RH) override;

    void prepareField() override;

    void cleanupField() override;

    LazyData<Vec<3,dcomplex>> getField(size_t l,
                                       const shared_ptr<const typename LevelsAdapter::Level>& level,
                                       const cvector& E, const cvector& H) override;

    LazyData<Tensor3<dcomplex>> getMaterialNR(size_t lay,
                                              const shared_ptr<const typename LevelsAdapter::Level>& level,
                                              InterpolationMethod interp) override;

    double integratePoyntingVert(const cvector& E, const cvector& H) override;

  private:

    DataVector<Vec<3,dcomplex>> field;
    FFT::Backward2D fft_x, fft_y, fft_z;

    void copy_coeffs_long(size_t l, const DataVector<Tensor3<dcomplex>>& work, size_t tw, size_t tc) {
        if (symmetric_long()) {
            std::copy_n(work.begin()+tw*nM0, nN0, coeffs[l].begin()+tc*nN0);
        } else {
            size_t nn = nN0/2;
            std::copy_n(work.begin()+tw*nM0, nn+1, coeffs[l].begin()+tc*nN0);
            std::copy_n(work.begin()+(tw+1)*nM0-nn, nn, coeffs[l].begin()+tc*nN0+nn+1);
        }
    }

  protected:

    DataVector<Tensor2<dcomplex>> mag_long; ///< Magnetic permeability coefficients in longitudinal direction (used for PMLs)
    DataVector<Tensor2<dcomplex>> mag_tran; ///< Magnetic permeability coefficients in transverse direction (used for PMLs)

    FFT::Forward2D matFFT;                  ///< FFT object for material coefficients

    /// Obtained temperature
    LazyData<double> temperature;

    /// Flag indicating if the gain is connected
    bool gain_connected;

    /// Obtained gain
    LazyData<double> gain;

    void prepareIntegrals(double lam, double glam) override;

    void cleanupIntegrals(double lam, double glam) override;

    void layerIntegrals(size_t layer, double lam, double glam) override;

  public:

    void setKlong(dcomplex k) {
        if (k != klong) {
            klong = k;
            solver->clearFields();
        }
    }

    void setKtran(dcomplex k) {
        if (k != ktran) {
            ktran = k;
            solver->clearFields();
        }
    }

    void setSymmetryLong(Component sym) {
        if (sym != symmetry_long) {
            symmetry_long = sym;
            solver->clearFields();
        }
    }

    void setSymmetryTran(Component sym) {
        if (sym != symmetry_tran) {
            symmetry_tran = sym;
            solver->clearFields();
        }
    }

    /// Get \f$ \varepsilon_{xx} \f$
    dcomplex epsxx(size_t lay, int l, int t) {
        if (l < 0) l += nN0; if (t < 0) t += nN1;
        return coeffs[lay][nN0 * t + l].c00;
    }

    /// Get \f$ \varepsilon_{yy} \f$
    dcomplex epsyy(size_t lay, int l, int t) {
        if (l < 0) l += nN0; if (t < 0) t += nN1;
        return coeffs[lay][nN0 * t + l].c11;
    }

    /// Get \f$ \varepsilon_{zz}^{-1} \f$
    dcomplex iepszz(size_t lay, int l, int t) {
        if (l < 0) l += nN0; if (t < 0) t += nN1;
        return coeffs[lay][nN0 * t + l].c22;
    }

    /// Get \f$ \varepsilon_{xy} \f$
    dcomplex epsxy(size_t lay, int l, int t) {
        if (l < 0) l += nN0; if (t < 0) t += nN1;
        return coeffs[lay][nN0 * t + l].c01;
    }

    /// Get \f$ \varepsilon_{yx} \f$
    dcomplex epsyx(size_t lay, int l, int t) { return conj(epsxy(lay, l, t)); }

    /// Get \f$ \mu_{xx} \f$
    dcomplex muxx(size_t lay, int l, int t) { return mag_long[(l>=0)?l:l+nN0].c11 * mag_tran[(t>=0)?t:t+nN1].c00; }

    /// Get \f$ \mu_{yy} \f$
    dcomplex muyy(size_t lay, int l, int t) { return mag_long[(l>=0)?l:l+nN0].c00 * mag_tran[(t>=0)?t:t+nN1].c11; }

    /// Get \f$ \mu_{zz}^{-1} \f$
    dcomplex imuzz(size_t lay, int l, int t) { return mag_long[(l>=0)?l:l+nN0].c11 * mag_tran[(t>=0)?t:t+nN1].c11; }

    /// Get \f$ E_x \f$ index
    size_t iEx(int l, int t) {
        if (l < 0) { if (symmetric_long()) l = -l; else l += N0; }
        if (t < 0) { if (symmetric_tran()) t = -t; else t += N1; }
        assert(0 <= l && l < N0);
        assert(0 <= t && t < N1);
        return 2 * (N0*t + l);
    }

    /// Get \f$ E_y \f$ index
    size_t iEy(int l, int t) {
        if (l < 0) { if (symmetric_long()) l = -l; else l += N0; }
        if (t < 0) { if (symmetric_tran()) t = -t; else t += N1; }
        assert(0 <= l && l < N0);
        assert(0 <= t && t < N1);
        return 2 * (N0*t + l) + 1;
    }

    /// Get \f$ H_x \f$ index
    size_t iHx(int l, int t) {
        if (l < 0) { if (symmetric_long()) l = -l; else l += N0; }
        if (t < 0) { if (symmetric_tran()) t = -t; else t += N1; }
        assert(0 <= l && l < N0);
        assert(0 <= t && t < N1);
        return 2 * (N0*t + l) + 1;
    }

    /// Get \f$ H_y \f$ index
    size_t iHy(int l, int t) {
        if (l < 0) { if (symmetric_long()) l = -l; else l += N0; }
        if (t < 0) { if (symmetric_tran()) t = -t; else t += N1; }
        assert(0 <= l && l < N0);
        assert(0 <= t && t < N1);
        return 2 * (N0*t + l);
    }
};

}}} // namespace plask

#endif // PLASK__SOLVER_SLAB_EXPANSION_PW3D_H
