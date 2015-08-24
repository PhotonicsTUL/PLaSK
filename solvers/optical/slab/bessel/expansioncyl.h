#ifndef PLASK__SOLVER__SLAB_EXPANSIONCYL_H
#define PLASK__SOLVER__SLAB_EXPANSIONCYL_H

#include <plask/plask.hpp>

#include "../expansion.h"
#include "../patterson.h"
#include "../meshadapter.h"

namespace plask { namespace solvers { namespace slab {

struct BesselSolverCyl;

struct PLASK_SOLVER_API ExpansionBessel: public Expansion {

    size_t N;                           ///< Number of expansion coefficients
    bool initialized;                   ///< Expansion is initialized

//     size_t pil,                         ///< Index of the beginning of the left PML
//            pir;                         ///< Index of the beginning of the right PML

    /// Horizontal axis with separate integration intervals.
    /// material functions contain discontinuities at these points
    shared_ptr<OrderedAxis> rbounds;
    
    ///  Argument coefficients for Bessel expansion base
    std::vector<double> factors;
    
    /**
     * Create new expansion
     * \param solver solver which performs calculations
     */
    ExpansionBessel(BesselSolverCyl* solver);

    /**
     * fill factors with Bessel zeros
     */
    void computeBesselZeros();
    
    /**
     * Init expansion
     * \param compute_coeffs compute material coefficients
     */
    void init();

    /// Free allocated memory
    void reset();

    /// Compute itegrals for RE and RH matrices
    void computeIntegrals() {
        size_t nlayers = lcount();
        assert(layers_integrals.size() == nlayers);
        #pragma omp parallel for
        for (size_t l = 0; l < nlayers; ++l)
            layerIntegrals(l);
    }

    size_t lcount() const override;

    bool diagonalQE(size_t l) const override {
        return diagonals[l];
    }

    size_t matrixSize() const override { return 2*N; } // TODO should be N for m = 0?

    void getMatrices(size_t l, cmatrix& RE, cmatrix& RH) override;

    void prepareField() override;

    void cleanupField() override;

    DataVector<const Vec<3,dcomplex>> getField(size_t layer,
                                               const shared_ptr<const typename LevelsAdapter::Level>& level,
                                               const cvector& E, const cvector& H) override;

    LazyData<Tensor3<dcomplex>> getMaterialNR(size_t layer,
                                              const shared_ptr<const typename LevelsAdapter::Level>& level,
                                              InterpolationMethod interp=INTERPOLATION_DEFAULT) override;

  protected:

    /// Integration segment data
    struct Segment {
        double Z;               ///< Center of the segment
        double D;               ///< Width of the segment divideb by 2.
        unsigned n;             ///< Patterson integration order for segment
    };
    
    /// Integration segments
    std::vector<Segment> segments;
    
    /// Axis for obtaining material parameters
    shared_ptr<OrderedAxis> raxis;
    
    /// Matrices with computed integrals necessary to construct RE and RH matrices
    struct Integrals {
        struct Data {
            dcomplex iem;   ///< J_{m-1}(gr) eps^{-1}(r) J_{m-1}(kr) r dr
            dcomplex iep;   ///< J_{m+1}(gr) eps^{-1}(r) J_{m+1}(kr) r dr
            dcomplex em;    ///< J_{m-1}(gr) eps(r) J_{m-1}(kr) r dr
            dcomplex ep;    ///< J_{m+1}(gr) eps(r) J_{m+1}(kr) r dr
            dcomplex dem;   ///< J_{m-1}(gr) deps/dr J_{m}(kr) r dr
            dcomplex dep;   ///< J_{m+1}(gr) deps/dr J_{m}(kr) r dr
            dcomplex bem;   ///< J_{m-1}(kr) deps/dr J_{m}(gr) r dr
            dcomplex bep;   ///< J_{m+1}(kr) deps/dr J_{m}(gr) r dr
            Data() {}
            Data(dcomplex val): iem(val), iep(val), em(val), ep(val), dem(val), dep(val), bem(val), bep(val) {}
        };
      private:
        DataVector<Data> data;
        inline size_t idx(size_t i, size_t j) const { return (i<=j)? j*(j+1)/2 + i: i*(i+1)/2 + j; }
      public:
        Integrals() {}
        Integrals(size_t N) { reset(N); }
        void reset(size_t N) {
            size_t len = N*(N+1)/2;
            data.reset(len, Data(0.));
        }
        void zero() {
            std::fill(data.begin(), data.end(), Data(0.));
        }
        dcomplex& ieps_minus(size_t i, size_t j) { return data[idx(i,j)].iem; }
        const dcomplex& ieps_minus(size_t i, size_t j) const { return data[idx(i,j)].iem; }
        dcomplex& ieps_plus(size_t i, size_t j) { return data[idx(i,j)].iep; }
        const dcomplex& ieps_plus(size_t i, size_t j) const { return data[idx(i,j)].iep; }
        dcomplex& eps_minus(size_t i, size_t j) { return data[idx(i,j)].em; }
        const dcomplex& eps_minus(size_t i, size_t j) const { return data[idx(i,j)].em; }
        dcomplex& eps_plus(size_t i, size_t j) { return data[idx(i,j)].ep; }
        const dcomplex& eps_plus(size_t i, size_t j) const { return data[idx(i,j)].ep; }

        dcomplex& deps_minus(size_t i, size_t j) {
            if (i <= j) return data[j*(j+1)/2+i].dem;
            else return data[i*(i+1)/2+j].bem;
        }
        const dcomplex& deps_minus(size_t i, size_t j) const {
            if (i <= j) return data[j*(j+1)/2+i].dem;
            else return data[i*(i+1)/2+j].bem;
        }
        dcomplex& deps_plus(size_t i, size_t j) {
            if (i <= j) return data[j*(j+1)/2+i].dep;
            else return data[i*(i+1)/2+j].bep;
        }
        const dcomplex& deps_plus(size_t i, size_t j) const {
            if (i <= j) return data[j*(j+1)/2+i].dep;
            else return data[i*(i+1)/2+j].bep;
        }
    };

    /// Computed integrals
    std::vector<Integrals> layers_integrals;
    
    /// Information if the layer is diagonal
    std::vector<bool> diagonals;

    /**
     * Compute itegrals for RE and RH matrices
     * \param l layer number
     */
    void layerIntegrals(size_t l);

  public:

};

}}} // # namespace plask::solvers::slab

#endif // PLASK__SOLVER__SLAB_EXPANSIONCYL_H