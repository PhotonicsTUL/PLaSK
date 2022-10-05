#ifndef PLASK__SOLVER_SLAB_EXPANSION_PW3D_H
#define PLASK__SOLVER_SLAB_EXPANSION_PW3D_H

#include <plask/plask.hpp>

#include "../expansion.hpp"
#include "fft.hpp"

namespace plask { namespace optical { namespace slab {

struct FourierSolver3D;

struct PLASK_SOLVER_API GradientFunctions: public MultiFieldProperty<double> {
    enum EnumType {
            COS2 = 0,
            COSSIN = 1
        };
    static constexpr size_t NUM_VALS = 2;
    static constexpr const char* NAME = "refractive index gradient functions cos² and cos·sin";
    static constexpr const char* UNIT = "-";
};

struct PLASK_SOLVER_API ExpansionPW3D: public Expansion {

    dcomplex klong,                     ///< Longitudinal wavevector
             ktran;                     ///< Transverse wavevector

    size_t Nl,                          ///< Number of expansion coefficients in longitudinal direction
           Nt;                          ///< Number of expansion coefficients in transverse direction
    size_t nNl,                         ///< Number of of required coefficients for material parameters in longitudinal direction
           nNt;                         ///< Number of of required coefficients for material parameters in transverse direction
    size_t eNl,                         ///< Number of expansion coefficients in longitudinal direction ignoring symmetry
           eNt;                         ///< Number of expansion coefficients in transverse direction ignoring symmetry

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

    ///< Refractive index expansion coefficients
    struct Coeff {
        dcomplex c22, c00, ic00, c11, ic11, c01;
        Coeff() {}
        Coeff(const Coeff&) = default;
        Coeff(const dcomplex& val): c22(val),
            c00(val), ic00((val != 0.)? 1. / val : 0.),
            c11(val), ic11((val != 0.)? 1. / val : 0.),
            c01(val) {}
        Coeff(const Tensor3<dcomplex>& eps): c22(eps.c22),
            c00(eps.c00), ic00((eps.c00 != 0.)? 1. / eps.c00 : 0.),
            c11(eps.c11), ic11((eps.c11 != 0.)? 1. / eps.c11 : 0.),
            c01(eps.c01) {}
        Coeff& operator*=(dcomplex a) {
            c22 *= a; c00 *= a; ic00 *= a; c11 *= a; ic11 *= a; c01 *= a;
            return *this;
        }
        Coeff& operator=(const Tensor3<dcomplex>& eps) {
            c22 = eps.c22;
            c00 = eps.c00; ic00 = (eps.c00 != 0.)? 1. / eps.c00 : 0.;
            c11 = eps.c11; ic11 = (eps.c11 != 0.)? 1. / eps.c11 : 0.;
            c01 = eps.c01;
            return *this;
        }
        bool differs(const Coeff& other) const {
            return !(is_zero(other.c22-c22) && is_zero(other.c00-c00) && is_zero(other.c11-c11) && is_zero(other.c01-c01));
        }
        operator Tensor3<dcomplex>() const { return Tensor3<dcomplex>(c00, c11, c22, c01); }
    };

    /// Cached permittivity expansion coefficients
    std::vector<DataVector<Coeff>> coeffs;

    /// Gradient data structure (cos² and cos·sin)
    struct Gradient {
        struct Vertex;
        dcomplex c2, cs;
        Gradient(const Gradient&) = default;
        Gradient(double c2, double cs): c2(c2), cs(cs) {}
        Gradient(const Vec<2>& norm): c2(norm.c0 * norm.c0), cs(norm.c0 * norm.c1) {}
        Gradient& operator=(const Gradient& norm) = default;
        Gradient& operator=(const Vec<2>& norm) {
            // double f = 1. / (norm.c0*norm.c0 + norm.c1*norm.c1);
            c2 = /* f *  */norm.c0 * norm.c0;
            cs = /* f *  */norm.c0 * norm.c1;
            return *this;
        }
        Gradient operator*(double f) const { return Gradient(c2.real() * f, cs.real() * f); }
        Gradient operator/(double f) const { return Gradient(c2.real() / f, cs.real() / f); }
        Gradient& operator+=(const Gradient& norm) { c2 += norm.c2; cs += norm.cs; return *this; }
        Gradient& operator*=(double f) { c2 *= f; cs *= f; return *this; }
        Gradient& operator/=(double f) { c2 /= f; cs /= f; return *this; }
        // Gradient operator/(size_t n) const { double f = 1. / double(n); return Gradient(c2.real() * f, cs.real() * f); }
        bool isnan() const { return ::isnan(c2.real()); }
    };

    /// Cached gradients data
    std::vector<DataVector<Gradient>> gradients;

    /// Cached ε_zz, Δε_xx, and Δε_yy matrices
    std::vector<cmatrix> coeffs_ezz, coeffs_dexx, coeffs_deyy;

    /// Information if the layer is diagonal
    std::vector<bool> diagonals;

    /// Mesh for getting material data
    shared_ptr<RectangularMesh<3>> mesh;

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

    size_t matrixSize() const override { return 2*Nl*Nt; }

    void getMatrices(size_t l, cmatrix& RE, cmatrix& RH) override;

    void prepareField() override;

    void cleanupField() override;

    LazyData<Vec<3,dcomplex>> getField(size_t l,
                                       const shared_ptr<const typename LevelsAdapter::Level>& level,
                                       const cvector& E, const cvector& H) override;

    LazyData<Tensor3<dcomplex>> getMaterialNR(size_t lay,
                                              const shared_ptr<const typename LevelsAdapter::Level>& level,
                                              InterpolationMethod interp) override;

    LazyData<double> getGradients(GradientFunctions::EnumType what,
                                  const shared_ptr<const typename LevelsAdapter::Level>& level,
                                  InterpolationMethod interp);

    double integrateField(WhichField field, size_t layer, const cmatrix& TE, const cmatrix& TH,
                          const std::function<std::pair<dcomplex,dcomplex>(size_t, size_t)>& vertical) override;

    double integratePoyntingVert(const cvector& E, const cvector& H) override;

    void getDiagonalEigenvectors(cmatrix& Te, cmatrix Te1, const cmatrix& RE, const cdiagonal& gamma) override;

  private:

    DataVector<Vec<3,dcomplex>> field;
    FFT::Backward2D fft_x, fft_y, fft_z;

    void addToeplitzMatrix(cmatrix& work, int ordl, int ordt, size_t lay, int c, char syml, char symt, double a = 1.) {
        for (int it = (symt ? 0 : -ordt); it <= ordt; ++it) {
            size_t It = (it >= 0)? it : it + Nt;
            for (int il = (syml ? 0 : -ordl); il <= ordl; ++il) {
                size_t Il = (il >= 0)? il : il + Nl;
                for (int jt = -ordt; jt <= ordt; ++jt) {
                    size_t Jt = (jt >= 0)? jt : (symt)? -jt : jt + Nt;
                    int ijt = it - jt; if (symt && ijt < 0) ijt = -ijt;
                    for (int jl = -ordl; jl <= ordl; ++jl) {
                        size_t Jl = (jl >= 0)? jl : (syml)? -jl : jl + Nl;
                        double f = 1.;
                        if (syml && jl < 0) { f *= syml; }
                        if (symt && jt < 0) { f *= symt; }
                        int ijl = il - jl; if (syml && ijl < 0) ijl = -ijl;
                        work(Nl * It + Il, Nl * Jt + Jl) += a * f * eps(lay, ijl, ijt, c);
                    }
                }
            }
        }
    }

    void makeToeplitzMatrix(cmatrix& work, int ordl, int ordt, size_t lay, int c, char syml, char symt, double a = 1.) {
        zero_matrix(work);
        addToeplitzMatrix(work, ordl, ordt, lay, c, syml, symt, a);
    }

    void makeToeplitzMatrix(cmatrix& workc2, cmatrix& workcs, const DataVector<Gradient>& norms,
                            int ordl, int ordt, char syml, char symt) {
        zero_matrix(workc2);
        zero_matrix(workcs);
        for (int it = (symt ? 0 : -ordt); it <= ordt; ++it) {
            size_t It = (it >= 0)? it : it + Nt;
            for (int il = (syml ? 0 : -ordl); il <= ordl; ++il) {
                size_t Il = (il >= 0)? il : il + Nl;
                size_t I = Nl * It + Il;
                for (int jt = -ordt; jt <= ordt; ++jt) {
                    size_t Jt = (jt >= 0)? jt : (symt)? -jt : jt + Nt;
                    int ijt = it - jt; if (ijt < 0) ijt = symt? -ijt : ijt + nNt;
                    for (int jl = -ordl; jl <= ordl; ++jl) {
                        size_t Jl = (jl >= 0)? jl : (syml)? -jl : jl + Nl;
                        double fc = 1., fs = 1.;
                        if (syml && jl < 0) { fc *= syml; fs *= -syml; }
                        if (symt && jt < 0) { fc *= symt; fs *= -symt; }
                        int ijl = il - jl; if (ijl < 0) ijl = syml? -ijl : ijl + nNl;
                        size_t J = Nl * Jt + Jl, ij = nNl * ijt + ijl;
                        workc2(I,J) += fc * norms[ij].c2;
                        workcs(I,J) += fs * norms[ij].cs;
                    }
                }
            }
        }
    }

  protected:

    DataVector<Tensor2<dcomplex>> mag_long; ///< Magnetic permeability coefficients in longitudinal direction (used with for PMLs)
    DataVector<Tensor2<dcomplex>> mag_tran; ///< Magnetic permeability coefficients in transverse direction (used with for PMLs)

    FFT::Forward2D matFFT;                  ///< FFT object for material coefficients
    FFT::Forward2D cos2FFT, cssnFFT;        ///< FFT object for gradients

    void beforeLayersIntegrals(double lam, double glam) override;

    void layerIntegrals(size_t layer, double lam, double glam) override;

  public:

    dcomplex getKlong() const { return klong; }
    void setKlong(dcomplex k) {
        if (k != klong) {
            klong = k;
            solver->clearFields();
        }
    }

    dcomplex getKtran() const { return ktran; }
    void setKtran(dcomplex k) {
        if (k != ktran) {
            ktran = k;
            solver->clearFields();
        }
    }

    Component getSymmetryLong() const { return symmetry_long; }
    void setSymmetryLong(Component sym) {
        if (sym != symmetry_long) {
            symmetry_long = sym;
            solver->clearFields();
        }
    }

    Component getSymmetryTran() const { return symmetry_tran; }
    void setSymmetryTran(Component sym) {
        if (sym != symmetry_tran) {
            symmetry_tran = sym;
            solver->clearFields();
        }
    }

    /// Get single index for x-y
    size_t idx(int l, int t) {
        if (l < 0) { if (symmetric_long()) l = -l; else l += int(Nl); }
        if (t < 0) { if (symmetric_tran()) t = -t; else t += int(Nt); }
        assert(0 <= l && std::size_t(l) < Nl);
        assert(0 <= t && std::size_t(t) < Nt);
        return Nl * t + l;
    }

    /// Get single index for x-y ignoring symmetry
    size_t idxe(int l, int t) {
        if (l < 0) l += int(eNl);
        if (t < 0) t += int(eNt);
        assert(0 <= l && std::size_t(l) < eNl);
        assert(0 <= t && std::size_t(t) < eNt);
        return eNl * t + l;
    }

    /// Get \f$ \varepsilon_{xx} \f$
    dcomplex epsxx(size_t lay, int l, int t) {
        if (l < 0) l += int(nNl);
        if (t < 0) t += int(nNt);
        return coeffs[lay][nNl * t + l].c00;
    }

    /// Get \f$ \varepsilon_{yy} \f$
    dcomplex epsyy(size_t lay, int l, int t) {
        if (l < 0) l += int(nNl);
        if (t < 0) t += int(nNt);
        return coeffs[lay][nNl * t + l].c11;
    }

    /// Get \f$ \varepsilon_{zz}^{-1} \f$
    dcomplex iepszz(size_t lay, int l, int t) {
        if (l < 0) l += int(nNl);
        if (t < 0) t += int(nNt);
        return coeffs[lay][nNl * t + l].c22;
    }

    /// Get \f$ \varepsilon_{zz}^{-1} \f$
    dcomplex iepszz(size_t lay, int il, int jl, int it, int jt) {
        if (jl < 0) { if (symmetric_long()) return 0.; else jl += int(Nl); }
        if (jt < 0) { if (symmetric_tran()) return 0.; else jt += int(Nt); }
        if (il < 0) il += int(Nl);
        if (it < 0) it += int(Nt);
        return coeffs_ezz[lay](Nl * it + il, Nl * jt + jl);
    }

    /// Get \f$ \varepsilon_{xy} \f$
    dcomplex epsxy(size_t lay, int l, int t) {
        if (l < 0) l += int(nNl);
        if (t < 0) t += int(nNt);
        return coeffs[lay][nNl * t + l].c01;
    }

    /// Get \f$ \varepsilon_c \f$
    dcomplex eps(size_t lay, int l, int t, int c) {
        if (l < 0) l += int(nNl);
        if (t < 0) t += int(nNt);
        return *(reinterpret_cast<dcomplex*>(coeffs[lay].data() + (nNl * t + l)) + c);
    }

    /// Get \f$ \varepsilon_{yx} \f$
    dcomplex epsyx(size_t lay, int l, int t) { return conj(epsxy(lay, l, t)); }

    /// Get \f$ \mu_{xx} \f$
    dcomplex muxx(size_t PLASK_UNUSED(lay), int l, int t) { return mag_long[(l>=0)?l:l+nNl].c11 * mag_tran[(t>=0)?t:t+nNt].c00; }

    /// Get \f$ \mu_{yy} \f$
    dcomplex muyy(size_t PLASK_UNUSED(lay), int l, int t) { return mag_long[(l>=0)?l:l+nNl].c00 * mag_tran[(t>=0)?t:t+nNt].c11; }

    /// Get \f$ \mu_{zz}^{-1} \f$
    dcomplex imuzz(size_t PLASK_UNUSED(lay), int l, int t) { return mag_long[(l>=0)?l:l+nNl].c11 * mag_tran[(t>=0)?t:t+nNt].c11; }

    /// Get \f$ E_x \f$ index
    size_t iEx(int l, int t) {
        return 2 * idx(l, t);
    }

    /// Get \f$ E_y \f$ index
    size_t iEy(int l, int t) {
        return 2 * idx(l, t) + 1;
    }

    /// Get \f$ H_x \f$ index
    size_t iHx(int l, int t) {
        return 2 * idx(l, t) + 1;
    }

    /// Get \f$ H_y \f$ index
    size_t iHy(int l, int t) {
        return 2 * idx(l, t);
    }
};

struct ExpansionPW3D::Gradient::Vertex {
    int l, t;
    Gradient val;
    Vertex(int l, int t, const Gradient& src): l(l), t(t), val(src) {}
};

}}} // namespace plask

#endif // PLASK__SOLVER_SLAB_EXPANSION_PW3D_H
