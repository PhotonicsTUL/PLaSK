#ifndef PLASK__SOLVER__SLAB_EXPANSIONCYL_FINI_H
#define PLASK__SOLVER__SLAB_EXPANSIONCYL_FINI_H

#include <plask/plask.hpp>

#include "expansioncyl.h"
#include "../patterson.h"
#include "../meshadapter.h"


namespace plask { namespace solvers { namespace slab {

struct PLASK_SOLVER_API ExpansionBesselFini: public ExpansionBessel {

    /**
     * Create new expansion
     * \param solver solver which performs calculations
     */
    ExpansionBesselFini(BesselSolverCyl* solver);

    /// Fill factors with Bessel zeros
    void computeBesselZeros();

    /// Perform m-specific initialization
    void init2() override;

    /// Free allocated memory
    void reset() override;

    void getMatrices(size_t layer, cmatrix& RE, cmatrix& RH) override;

  protected:

    /// Matrices with computed integrals necessary to construct RE and RH matrices
    struct Integrals {

        struct Data {
            dcomplex im;   ///< J_{m-1}(gr) eps_{zz}^{-1}(r) J_{m-1}(kr) r dr
            dcomplex ip;   ///< J_{m+1}(gr) eps_{zz}^{-1}(r) J_{m+1}(kr) r dr
            dcomplex mm;    ///< J_{m-1}(gr) ½ [eps_{rr}(r) + eps_{pp}(r)] J_{m-1}(kr) r dr
            dcomplex pp;    ///< J_{m+1}(gr) ½ [eps_{rr}(r) + eps_{pp}(r)] J_{m+1}(kr) r dr
            dcomplex mp;    ///< J_{m-1}(gr) ½ [eps_{rr}(r) - eps_{pp}(r)] J_{m+1}(kr) r dr
            dcomplex pm;    ///< J_{m+1}(gr) ½ [eps_{rr}(r) - eps_{pp}(r)] J_{m-1}(kr) r dr
            dcomplex dm;   ///< J_{m-1}(gr) deps_{zz}^{-1}/dr J_{m}(kr) r dr
            dcomplex dp;   ///< J_{m+1}(gr) deps_{zz}^{-1}/dr J_{m}(kr) r dr
            dcomplex bm;   ///< J_{m-1}(kr) deps_{zz}^{-1}/dr J_{m}(gr) r dr
            dcomplex bp;   ///< J_{m+1}(kr) deps_{zz}^{-1}/dr J_{m}(gr) r dr
            Data() {}
          private:
            friend struct Integrals;
            Data(std::nullptr_t):
                im(0.), ip(0.), mm(0.), pp(0.), mp(0.), pm(0.), dm(0.), dp(0.), bm(0.), bp(0.) {}
        };

      private:
        DataVector<Data> data;

        inline size_t idx(size_t i, size_t j) const { return (i<=j)? j*(j+1)/2 + i: i*(i+1)/2 + j; }

      public:
        Integrals() {}

        Integrals(size_t N) { reset(N); }

        void reset(size_t N) {
            size_t len = N*(N+1)/2;
            data.reset(len, Data(nullptr));
        }

        void reset() {
            data.reset();
        }

        void zero() {
            std::fill(data.begin(), data.end(), Data(nullptr));
        }

        Data& operator()(size_t i, size_t j) { return data[idx(i,j)]; }
        const Data& operator()(size_t i, size_t j) const { return data[idx(i,j)]; }

        dcomplex& Vmm(size_t i, size_t j) { return data[idx(i,j)].im; }
        const dcomplex& Vmm(size_t i, size_t j) const { return data[idx(i,j)].im; }

        dcomplex& Vpp(size_t i, size_t j) { return data[idx(i,j)].ip; }
        const dcomplex& Vpp(size_t i, size_t j) const { return data[idx(i,j)].ip; }

        dcomplex& Tmm(size_t i, size_t j) { return data[idx(i,j)].mm; }
        const dcomplex& Tmm(size_t i, size_t j) const { return data[idx(i,j)].mm; }

        dcomplex& Tpp(size_t i, size_t j) { return data[idx(i,j)].pp; }
        const dcomplex& Tpp(size_t i, size_t j) const { return data[idx(i,j)].pp; }

        dcomplex& Tmp(size_t i, size_t j) {
            if (i <= j) return data[j*(j+1)/2+i].mp;
            else return data[i*(i+1)/2+j].pm;
        }
        const dcomplex& Tmp(size_t i, size_t j) const {
            if (i <= j) return data[j*(j+1)/2+i].mp;
            else return data[i*(i+1)/2+j].pm;
        }

        dcomplex& Tpm(size_t i, size_t j) {
            if (i <= j) return data[j*(j+1)/2+i].pm;
            else return data[i*(i+1)/2+j].mp;
        }
        const dcomplex& Tpm(size_t i, size_t j) const {
            if (i <= j) return data[j*(j+1)/2+i].pm;
            else return data[i*(i+1)/2+j].mp;
        }

        dcomplex& Dm(size_t i, size_t j) {
            if (i <= j) return data[j*(j+1)/2+i].dm;
            else return data[i*(i+1)/2+j].bm;
        }
        const dcomplex& Dm(size_t i, size_t j) const {
            if (i <= j) return data[j*(j+1)/2+i].dm;
            else return data[i*(i+1)/2+j].bm;
        }

        dcomplex& Dp(size_t i, size_t j) {
            if (i <= j) return data[j*(j+1)/2+i].dp;
            else return data[i*(i+1)/2+j].bp;
        }
        const dcomplex& Dp(size_t i, size_t j) const {
            if (i <= j) return data[j*(j+1)/2+i].dp;
            else return data[i*(i+1)/2+j].bp;
        }
    };

    /// Computed integrals
    std::vector<Integrals> layers_integrals;

    /// Integrals for magnetic permeability
    Integrals mu_integrals;

    void layerIntegrals(size_t layer, double lam, double glam) override;

  public:

#ifndef NDEBUG
    cmatrix epsVmm(size_t layer);
    cmatrix epsVpp(size_t layer);
    cmatrix epsTmm(size_t layer);
    cmatrix epsTpp(size_t layer);
    cmatrix epsTmp(size_t layer);
    cmatrix epsTpm(size_t layer);
    cmatrix epsDm(size_t layer);
    cmatrix epsDp(size_t layer);

    cmatrix muVmm();
    cmatrix muVpp();
    cmatrix muTmm();
    cmatrix muTpp();
    cmatrix muTmp();
    cmatrix muTpm();
    cmatrix muDm();
    cmatrix muDp();
#endif

};

}}} // # namespace plask::solvers::slab

#endif // PLASK__SOLVER__SLAB_EXPANSIONCYL_FINI_H
