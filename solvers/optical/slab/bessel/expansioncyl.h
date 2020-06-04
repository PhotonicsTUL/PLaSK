#ifndef PLASK__SOLVER__SLAB_EXPANSIONCYL_H
#define PLASK__SOLVER__SLAB_EXPANSIONCYL_H

#include <plask/plask.hpp>

#include "../expansion.h"
#include "../meshadapter.h"
#include "../patterson.h"

namespace plask { namespace optical { namespace slab {

struct BesselSolverCyl;

struct PLASK_SOLVER_API ExpansionBessel : public Expansion {
    int m;  ///< Angular dependency index

    bool initialized;  ///< Expansion is initialized

    bool m_changed;  ///< m has changed and init2 must be called

    /// Horizontal axis with separate integration intervals.
    /// material functions contain discontinuities at these points
    OrderedAxis rbounds;

    /// Argument coefficients for Bessel expansion base (zeros of Bessel function for finite domain)
    std::vector<double> kpts;

    /// Mesh for getting material data
    shared_ptr<RectangularMesh<2>> mesh;

    /**
     * Create new expansion
     * \param solver solver which performs calculations
     */
    ExpansionBessel(BesselSolverCyl* solver);

    virtual ~ExpansionBessel() {}

    /// Init expansion
    void init1();

    /// Perform m-specific initialization
    virtual void init2() = 0;

    /// Estimate required integration order
    void init3();

    /// Free allocated memory
    virtual void reset();

    bool diagonalQE(size_t l) const override { return diagonals[l]; }

    size_t matrixSize() const override;

    void prepareField() override;

    void cleanupField() override;

    LazyData<Vec<3, dcomplex>> getField(size_t layer,
                                        const shared_ptr<const typename LevelsAdapter::Level>& level,
                                        const cvector& E,
                                        const cvector& H) override;

    LazyData<Tensor3<dcomplex>> getMaterialNR(size_t layer,
                                              const shared_ptr<const typename LevelsAdapter::Level>& level,
                                              InterpolationMethod interp = INTERPOLATION_DEFAULT) override;

  private:
    inline double getT(size_t layer, size_t ri) {
        double T = 0., W = 0.;
        for (size_t k = 0, v = ri * solver->verts->size(); k != mesh->vert()->size(); ++v, ++k) {
            if (solver->stack[k] == layer) {
                double w = (k == 0 || k == mesh->vert()->size() - 1) ? 1e-6 : solver->vbounds->at(k) - solver->vbounds->at(k - 1);
                T += w * temperature[v];
                W += w;
            }
        }
        T /= W;
        return T;
    }

  protected:
    /// Integration segment data
    struct Segment {
        double Z;                    ///< Center of the segment
        double D;                    ///< Width of the segment divided by 2
        DataVector<double> weights;  ///< Cached integration weights for segment
    };

    /// Integration segments
    std::vector<Segment> segments;

    /// Information if the layer is diagonal
    std::vector<bool> diagonals;

    /// Obtained temperature
    LazyData<double> temperature;

    /// Flag indicating if the gain is connected
    bool gain_connected;

    /// Obtained gain
    LazyData<Tensor2<double>> gain;

    /// Matrices with computed integrals necessary to construct RE and RH matrices
    struct Integrals {
        cmatrix Vzz;  ///< [ J_{m}(gr) eps_{zz} J_{m}(kr) r dr ]^{-1}
        cmatrix Tss;  ///< [ J_{m-1}(gr) eps_{rr}^{-1} J_{m-1}(kr) r dr ]^{-1} + [ J_{m-1}(gr) eps_{pp} J_{m-1}(kr) r dr ]
        cmatrix Tsp;  ///< [ J_{m-1}(gr) eps_{rr}^{-1} J_{m-1}(kr) r dr ]^{-1} [ J_{m-1}(hr) J_{m+1}(kr) r dr ]
                      ///<  - [ J_{m-1}(gr)eps_{pp} J_{m-1}(kr) r dr ]
        cmatrix Tps;  ///< [ J_{m+1}(gr) eps_{rr}^{-1} J_{m+1}(hr) r dr ]^{-1} [ J_{m+1}(hr) J_{m-1}(kr) r dr ]
                      ///<  - [ J_{m+1}(gr)eps_{pp} J_{m+1}(kr) r dr ]
        cmatrix Tpp;  ///< [ J_{m+1}(gr) eps_{rr}^{-1} J_{m+1}(hr) r dr ]^{-1} + [ J_{m+1}(gr) eps_{pp} J_{m+1}(kr) r dr ]

      public:
        void reset() {
            Vzz.reset();
            Tss.reset();
            Tsp.reset();
            Tps.reset();
            Tpp.reset();
        }

        void reset(size_t N) {
            Vzz.reset(N, N);
            Tss.reset(N, N);
            Tsp.reset(N, N);
            Tps.reset(N, N);
            Tpp.reset(N, N);
        }
    };

    /// Computed integrals
    std::vector<Integrals> layers_integrals;

    void prepareIntegrals(double lam, double glam) override;

    void cleanupIntegrals(double, double) override;

    Tensor3<dcomplex> getEps(size_t layer, size_t ri, double r, double matz, double lam, double glam);

    std::pair<dcomplex, dcomplex> integrateLayer(size_t layer, double lam, double glam, bool finite);

    void integrateParams(Integrals& integrals,
                         const dcomplex* epsp_data, const dcomplex* iepsr_data, const dcomplex* epsz_data);

  public:
    unsigned getM() const { return m; }
    void setM(unsigned n) {
        if (int(n) != m) {
            write_debug("{0}: m changed from {1} to {2}", solver->getId(), m, n);
            m = n;
            solver->recompute_integrals = true;
            solver->clearFields();
        }
    }

    /// Get \f$ X_s \f$ index
    size_t idxs(size_t i) { return 2 * i; }

    /// Get \f$ X_p \f$ index
    size_t idxp(size_t i) { return 2 * i + 1; }

    // #ifndef NDEBUG
    //     cmatrix epsVmm(size_t layer);
    //     cmatrix epsVpp(size_t layer);
    //     cmatrix epsTmm(size_t layer);
    //     cmatrix epsTpp(size_t layer);
    //     cmatrix epsTmp(size_t layer);
    //     cmatrix epsTpm(size_t layer);
    //     cmatrix epsDm(size_t layer);
    //     cmatrix epsDp(size_t layer);
    //     dmatrix epsVV(size_t layer);
    // #endif
};

}}}  // namespace plask::optical::slab

#endif  // PLASK__SOLVER__SLAB_EXPANSIONCYL_H
