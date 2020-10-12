#ifndef PLASK__SOLVER__SLAB_EXPANSIONCYL_H
#define PLASK__SOLVER__SLAB_EXPANSIONCYL_H

#include <plask/plask.hpp>

#include "../expansion.hpp"
#include "../meshadapter.hpp"
#include "../patterson.hpp"

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

    double integratePoyntingVert(const cvector& E, const cvector& H) override;

    double integrateField(WhichField field, size_t layer, const cvector& E, const cvector& H) override;

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
        cmatrix V_k;
        cmatrix TT;
        cmatrix Tss;
        cmatrix Tsp;
        cmatrix Tps;
        cmatrix Tpp;

        void reset() {
            V_k.reset();
            TT.reset();
            Tss.reset();
            Tsp.reset();
            Tps.reset();
            Tpp.reset();
        }

        void reset(size_t N) {
            V_k.reset(N, N);
            TT.reset(2*N, 2*N);
            size_t NN = N * N;
            Tss.reset(N, N, TT.data());
            Tsp.reset(N, N, TT.data() + NN);
            Tps.reset(N, N, TT.data() + 2*NN);
            Tpp.reset(N, N, TT.data() + 3*NN);
        }
    };

    /// Computed integrals
    std::vector<Integrals> layers_integrals;

    void prepareIntegrals(double lam, double glam) override;

    void cleanupIntegrals(double, double) override;

    Tensor3<dcomplex> getEps(size_t layer, size_t ri, double r, double matz, double lam, double glam);

    void layerIntegrals(size_t layer, double lam, double glam) override;

    virtual void integrateParams(Integrals& integrals,
                                 const dcomplex* datap, const dcomplex* datar, const dcomplex* dataz,
                                 dcomplex datap0, dcomplex datar0, dcomplex dataz0) = 0;

    virtual double fieldFactor(size_t i) = 0;

    virtual cvector getHz(const cvector& Bz) = 0;

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

    #ifndef NDEBUG
        cmatrix epsV_k(size_t layer);
        cmatrix epsTss(size_t layer);
        cmatrix epsTsp(size_t layer);
        cmatrix epsTps(size_t layer);
        cmatrix epsTpp(size_t layer);
    #endif

    /// Return expansion wavevectors
    virtual std::vector<double> getKpts() {
        std::vector<double> res;
        res.reserve(kpts.size());
        double ib = 1. / rbounds[rbounds.size()-1];
        for (double k: kpts) res.push_back(k * ib);
        return res;
    }
};

}}}  // namespace plask::optical::slab

#endif  // PLASK__SOLVER__SLAB_EXPANSIONCYL_H
