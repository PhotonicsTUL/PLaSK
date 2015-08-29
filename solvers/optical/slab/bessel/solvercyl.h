#ifndef PLASK__SOLVER__SLAB_SOLVERCYL_H
#define PLASK__SOLVER__SLAB_SOLVERCYL_H

#include <plask/plask.hpp>

#include "../solver.h"
#include "../reflection.h"
#include "expansioncyl.h"


namespace plask { namespace solvers { namespace slab {

/**
 * Reflection transformation solver in Cartesian 2D geometry.
 */
struct PLASK_SOLVER_API BesselSolverCyl: public SlabSolver<Geometry2DCylindrical> {

    friend struct ExpansionBessel;
    
    std::string getClassName() const override { return "optical.BesselCyl"; }

    struct Mode {
        BesselSolverCyl* solver;        ///< Solver this mode belongs to
        dcomplex k0;                    ///< Stored mode frequency
        int m;                          ///< 
        double power;                   ///< Mode power [mW]

        Mode(BesselSolverCyl* solver): solver(solver), power(1e-9) {}

        bool operator==(const Mode& other) const {
            return m == other.m && is_zero(k0 - other.k0);
        }
    };

    struct ParamGuard {
        BesselSolverCyl* solver;
        dcomplex k0;
        int m;
        bool recomp;
        ParamGuard(BesselSolverCyl* solver, bool recomp=false): solver(solver),
            k0(solver->k0), m(solver->m), recomp(recomp) {}
        ~ParamGuard() {
            solver->setM(m);
            solver->setK0(k0, recomp);
        }
    };

  protected:

    /// Angular dependency index
    unsigned m;

    /// Maximum order of the orthogonal base
    size_t size;

    /// Class responsible for computing expansion coefficients
    ExpansionBessel expansion;

    void onInitialize() override;

    void onInvalidate() override;

    void computeIntegrals() override {
        expansion.computeIntegrals();
    }

    /// Type of discrete cosine transform. Can be only 1 or two
    int dct;

  public:

    /// Computed modes
    std::vector<Mode> modes;

    /// Expected integration estimate error
    double integral_error;
    
    /// Provider for computed resonant wavelength
    typename ProviderFor<Wavelength>::Delegate outWavelength;

    /// Provider for computed modal extinction
    typename ProviderFor<ModalLoss>::Delegate outLoss;

    BesselSolverCyl(const std::string& name="");

    void loadConfiguration(XMLReader& reader, Manager& manager) override;

    /**
     * Find the mode around the specified effective index.
     * This method remembers the determined mode, for retrieval of the field profiles.
     * \param start initial wavelength value to search the mode around
     * \return determined effective index
     */
    size_t findMode(dcomplex start, int m=0);

    /// Get order of the orthogonal base
    size_t getSize() const { return size; }
    /// Set order of the orthogonal base
    void setSize(size_t n) {
        size = n;
        invalidate();
    }

    /// Get order of the orthogonal base
    unsigned getM() const { return m; }
    /// Set order of the orthogonal base
    void setM(unsigned n) {
        if (n != m) {
            m = n;
            recompute_integrals = true;
        }
    }

    Expansion& getExpansion() override { return expansion; }

  protected:

    /// Insert mode to the list or return the index of the exiting one
    size_t insertMode() {
        Mode mode(this);
        mode.k0 = k0;
        for (size_t i = 0; i != modes.size(); ++i)
            if (modes[i] == mode) return i;
        modes.push_back(mode);
        outWavelength.fireChanged();
        outLoss.fireChanged();
        outLightMagnitude.fireChanged();
        outElectricField.fireChanged();
        outMagneticField.fireChanged();
        return modes.size()-1;
    }

    size_t nummodes() const override { return modes.size(); }

    /**
     * Return mode effective index
     * \param n mode number
     */
    double getWavelength(size_t n) {
        if (n >= modes.size()) throw NoValue(Wavelength::NAME);
        return modes[n].k0.real();
    }

    /**
     * Return mode modal loss
     * \param n mode number
     */
    double getModalLoss(size_t n) {
        if (n >= modes.size()) throw NoValue(ModalLoss::NAME);
        return 2e4 * modes[n].k0.imag();  // 2e4  2/µm -> 2/cm
    }

    /**
     * Compute electric field
     * \param num mode number
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    LazyData<Vec<3,dcomplex>> getE(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method) override;

    /**
     * Compute magnetic field
     * \param num mode number
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    LazyData<Vec<3,dcomplex>> getH(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method) override;

    /**
     * Compute light intensity
     * \param num mode number
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    LazyData<double> getMagnitude(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method) override;

#ifndef NDEBUG
  public:
    cmatrix ieps_minus(size_t layer);
    cmatrix ieps_plus(size_t layer);
    cmatrix eps_minus(size_t layer);
    cmatrix eps_plus(size_t layer);
    cmatrix deps_minus(size_t layer);
    cmatrix deps_plus(size_t layer);
#endif
                                              
};



}}} // # namespace plask::solvers::slab

#endif // PLASK__SOLVER__SSLAB_SOLVERCYL_H