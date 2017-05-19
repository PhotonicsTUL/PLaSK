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
struct PLASK_SOLVER_API BesselSolverCyl: public SlabSolver<SolverWithMesh<Geometry2DCylindrical,OrderedAxis>> {

    friend struct ExpansionBessel;

    std::string getClassName() const override { return "optical.BesselCyl"; }

    struct Mode {
        double lam0;                    ///< Wavelength for which integrals are computed
        dcomplex k0;                    ///< Stored mode frequency
        int m;                          ///< Stored angular parameter
        double power;                   ///< Mode power [mW]
        double tolx;                            ///< Tolerance for mode comparison

        Mode(const ExpansionBessel& expansion, double tolx): 
            lam0(expansion.lam0), k0(expansion.k0), m(expansion.m), power(1e-9), tolx(tolx) {}

        bool operator==(const Mode& other) const {
            return m == other.m && is_equal(k0, other.k0) && is_equal(lam0, other.lam0) &&
                   ((isnan(lam0) && isnan(other.lam0)) || lam0 == other.lam0);
        }

        bool operator==(const ExpansionBessel& other) const {
            return m == other.m && is_equal(k0, other.k0) && is_equal(lam0, other.lam0) &&
                   ((isnan(lam0) && isnan(other.lam0)) || lam0 == other.lam0);
        }

        template <typename T>
        bool operator!=(const T& other) const {
            return !(*this == other);
        }
        
      private:
    
        /// Compare mode arguments
        template <typename T>
        bool is_equal(T a, T b) const {
            return abs(a-b) <= tolx;
        }
    };

  protected:

    /// Angular dependency index
    int m;

    /// Maximum order of the orthogonal base
    size_t size;

    void onInitialize() override;

    void onInvalidate() override;

    void computeIntegrals() override {
        expansion.computeIntegrals();
    }

    /// Type of discrete cosine transform. Can be only 1 or two
    int dct;

  public:

    /// Class responsible for computing expansion coefficients
    ExpansionBessel expansion;

    /// Computed modes
    std::vector<Mode> modes;

    void clearModes() override {
        modes.clear();
    }

    void setExpansionDefaults(bool with_k0=true) override {
        expansion.setLam0(getLam0());
        if (with_k0) {
            expansion.setK0(getK0());
            expansion.setM(getM());
        }
    }

    /// Expected integration estimate error
    double integral_error;

    /// Maximum number of integration points in a single segemnt
    size_t max_itegration_points;

    /// Lateral PMLs
    PML pml;

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
    size_t findMode(dcomplex start, int m=1);

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
    void setM(unsigned n) { m = n; }

    /**
     * Return mode wavelength
     * \param n mode number
     */
    double getWavelength(size_t n) {
        if (n >= modes.size()) throw NoValue(Wavelength::NAME);
        return (2e3*M_PI / modes[n].k0).real();
    }

    Expansion& getExpansion() override { return expansion; }

    /**
     * Compute electric field coefficients for given \a z
     * \param num mode number
     * \param z position within the layer
     * \return electric field coefficients
     */
    cvector getFieldVectorE(size_t num, double z) {
        applyMode(modes[num]);
        return transfer->getFieldVectorE(z);
    }

    /**
     * Compute magnetic field coefficients for given \a z
     * \param num mode number
     * \param z position within the layer
     * \return magnetic field coefficients
     */
    cvector getFieldVectorH(size_t num, double z) {
        applyMode(modes[num]);
        return transfer->getFieldVectorH(z);
    }

    /// Check if the current parameters correspond to some mode and insert it
    size_t setMode() {
        if (abs2(this->getDeterminant()) > root.tolf_max*root.tolf_max)
            throw BadInput(this->getId(), "Cannot set the mode, determinant too large");
        return insertMode();
    }

  protected:

    /// Insert mode to the list or return the index of the exiting one
    size_t insertMode() {
        Mode mode(expansion, root.tolx);
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

    void applyMode(const Mode& mode) {
        writelog(LOG_DEBUG, "Current mode <m: {:d}, lam: {}nm>", mode.m, str(2e3*M_PI/mode.k0, "({:.3f}{:+.3g}j)"));
        expansion.setLam0(mode.lam0);
        expansion.setK0(mode.k0);
        expansion.setM(mode.m);
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

#endif // PLASK__SOLVER__SSLAB_SOLVERCYL_H
