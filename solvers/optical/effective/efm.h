#ifndef PLASK__MODULE_OPTICAL_EFM_HPP
#define PLASK__MODULE_OPTICAL_EFM_HPP

#include <limits>

#include <plask/plask.hpp>
#include <camos/camos.h>

#include "broyden.h"
#include "bisection.h"

namespace plask { namespace solvers { namespace effective {

static constexpr int MH = 2; // Hankel function type (1 or 2)

/**
 * Solver performing calculations in 2D Cartesian space using effective index method
 */
struct EffectiveFrequencyCylSolver: public SolverWithMesh<Geometry2DCylindrical, RectilinearMesh2D> {

    struct FieldZ {
        dcomplex F, B;
        FieldZ() = default;
        FieldZ(dcomplex f, dcomplex b): F(f), B(b) {}
        FieldZ operator*(dcomplex a) const { return FieldZ(F*a, B*a); }
        FieldZ operator/(dcomplex a) const { return FieldZ(F/a, B/a); }
        FieldZ operator*=(dcomplex a) { F *= a; B *= a; return *this; }
        FieldZ operator/=(dcomplex a) { F /= a; B /= a; return *this; }
    };

    struct FieldR {
        dcomplex J, H;
        FieldR() = default;
        FieldR(dcomplex j, dcomplex h): J(j), H(h) {}
        FieldR operator*(dcomplex a) const { return FieldR(a*J, a*H); }
        FieldR operator/(dcomplex a) const { return FieldR(J/a, H/a); }
        FieldR operator*=(dcomplex a) { J *= a; H *= a; return *this; }
        FieldR operator/=(dcomplex a) { J /= a; H /= a; return *this; }
    };

    struct MatrixR {
        dcomplex JJ, JH, HJ, HH;
        MatrixR(dcomplex jj, dcomplex jh, dcomplex hj, dcomplex hh): JJ(jj), JH(jh), HJ(hj), HH(hh) {}
        FieldR operator*(const FieldR& v) { return FieldR(JJ*v.J + JH*v.H, HJ*v.J + HH*v.H); }
        FieldR solve(const FieldR& v) {
            return FieldR(HH*v.J - JH*v.H, -HJ*v.J + JJ*v.H) / (JJ*HH - JH*HJ);
        }
    };

    /// Direction of the possible emission
    enum Emission {
        TOP,        ///< Top emission
        BOTTOM      ///< Bottom emission
    };

    /// Details of the computed mode
    struct Mode {
        EffectiveFrequencyCylSolver* solver;///< Solver this mode belongs to
        int m;                              ///< Number of the LP_mn mode describing angular dependence
        bool have_fields;                   ///< Did we compute fields for current state?
        std::vector<FieldR> rfields;        ///< Computed horizontal fields
        dcomplex freqv;                     ///< Stored frequency parameter
        double power;                       ///< Mode power [mW]
        
        Mode(EffectiveFrequencyCylSolver* solver): solver(solver), m(0), have_fields(false), rfields(solver->rsize), power(1.) {}
        
        Mode(EffectiveFrequencyCylSolver* solver, int m): solver(solver), m(m), have_fields(false), rfields(solver->rsize), power(1.) {}
        
        bool operator==(const Mode& other) {
            return m == other.m && is_zero(freqv - other.freqv);
        }
        
        /// Compute horizontal part of the field
        dcomplex rField(double r) const {
            double Jr, Ji, Hr, Hi;
            long nz, ierr;
            size_t ir = solver->mesh->axis0.findIndex(r); if (ir > 0) --ir; if (ir >= solver->veffs.size()) ir = solver->veffs.size()-1;
            dcomplex x = r * solver->k0 * sqrt(solver->nng[ir] * (solver->veffs[ir]-freqv));
            if (real(x) < 0.) x = -x;
            zbesj(x.real(), x.imag(), m, 1, 1, &Jr, &Ji, nz, ierr);
            if (ierr != 0)
                throw ComputationError(solver->getId(), "Could not compute J(%1%, %2%)", m, str(x));
            if (ir == 0) {
                Hr = Hi = 0.;
            } else {
                zbesh(x.real(), x.imag(), m, 1, MH, 1, &Hr, &Hi, nz, ierr);
                if (ierr != 0)
                    throw ComputationError(solver->getId(), "Could not compute H(%1%, %2%)", m, str(x));
            }
            return rfields[ir].J * dcomplex(Jr, Ji) + rfields[ir].H * dcomplex(Hr, Hi);
        }
    };
    
  protected:

    friend struct RootDigger;

    /// Logger for char_val
    Data2DLog<dcomplex,dcomplex> log_value;

    size_t rsize,   ///< Last element of horizontal mesh to consider
           zbegin,  ///<First element of vertical mesh to consider
           zsize;   ///< Last element of vertical mesh to consider

    /// Cached refractive indices
    std::vector<std::vector<dcomplex,aligned_allocator<dcomplex>>> nrCache;

    /// Cached group indices
    std::vector<std::vector<dcomplex,aligned_allocator<dcomplex>>> ngCache;

    /// Computed vertical fields
    std::vector<FieldZ> zfields;

    /// Stripe to take vertical fields from
    size_t stripe;

    /// Computed effective frequencies for each stripe
    std::vector<dcomplex,aligned_allocator<dcomplex>> veffs;

    /// Computed weighted indices for each stripe
    std::vector<dcomplex,aligned_allocator<dcomplex>> nng;

    /// Old value of k0 to detect changes
    dcomplex old_k0;

    /// Direction of laser emission
    Emission emission;

  public:

    /// Distance outside outer borders where material is sampled
    double outdist;

    // Parameters for rootdigger
    RootDigger::Params root;        ///< Parameters for horizontal root digger
    RootDigger::Params stripe_root; ///< Parameters for vertical root diggers

    /// Allowed relative power integral precision
    double perr;

    /// Current value of reference normalized frequency [1/µm]
    dcomplex k0;

    /// Computed modes
    std::vector<Mode> modes;

    EffectiveFrequencyCylSolver(const std::string& name="");

    virtual std::string getClassName() const { return "optical.EffectiveFrequencyCyl"; }

    virtual std::string getClassDescription() const {
        return "Calculate optical modes and optical field distribution using the effective index method "
               "in Cartesian two-dimensional space.";
    }

    virtual void loadConfiguration(plask::XMLReader& reader, plask::Manager& manager);

    /// Get emission direction
    ///\return emission direction
    Emission getEmission() const { return emission; }

    /// Set emission direction
    /// \param emis new emissjon direction
    void setEmission(Emission emis) {
        emission = emis;
        for (auto& mode: modes) 
            mode.have_fields = false;
    }

    /**
     * Set the simple mesh based on the geometry bounding boxes.
     **/
    void setSimpleMesh() {
        writelog(LOG_INFO, "Creating simple mesh");
        setMesh(make_shared<RectilinearMesh2DSimpleGenerator>(true)); // set generator forcing line at r = 0
    }

    /**
     * Set up the horizontal mesh. Horizontal division is provided while vertical one is created basing on the geometry bounding boxes.
     *
     * \param meshx horizontal mesh
     **/
    void setHorizontalMesh(const RectilinearMesh1D& meshx) {
        writelog(LOG_INFO, "Setting horizontal mesh");
        if (!geometry) throw NoChildException();
        auto meshxy = make_shared<RectilinearMesh2D>(*RectilinearMesh2DSimpleGenerator()(geometry->getChild()));
        meshxy->tran() = meshx;
        setMesh(meshxy);
    }

    /**
     * Find the mode around the specified effective index.
     *
     * This method remembers the determined mode, for retrieval of the field profiles.
     *
     * \param lambda0 initial wavelength close to the solution
     * \param m number of the LP_mn mode describing angular dependence
     * \return index of the found mode
     */
    size_t findMode(dcomplex lambda0, int m=0);

    /**
     * Find the modes within the specified range
     *
     * This method \b does \b not remember the determined modes!
     *
     * \param lambda1 one corner of the range to browse
     * \param lambda2 another corner of the range to browse
     * \param m number of the LP_mn mode describing angular dependence
     * \param resteps minimum number of steps to check function value on real contour
     * \param imsteps minimum number of steps to check function value on imaginary contour
     * \param eps approximate error for integrals
     * \return vector of indices of determined modes
     */
    std::vector<size_t> findModes(plask::dcomplex lambda1=0., plask::dcomplex lambda2=0., int m=0, size_t resteps=256, size_t imsteps=64, dcomplex eps=dcomplex(1e-6,1e-9));

    /**
     * Compute modal determinant for the whole matrix
     * \param v frequency parameter
     * \param m number of the LP_mn mode describing angular dependence
     */
    dcomplex getDeterminantV(dcomplex v, int m=0) {
        stageOne();
        Mode mode(this, m);
        return detS(v, mode);
    }

    /**
     * Compute modal determinant for the whole matrix
     * \param lambda wavelength
     * \param m number of the LP_mn mode describing angular dependence
     */
    dcomplex getDeterminant(dcomplex lambda, int m=0) {
        if (isnan(k0.real())) k0 = 2e3*M_PI / lambda;
        dcomplex v =  2. - 4e3*M_PI / lambda / k0;
        stageOne();
        Mode mode(this,m);
        dcomplex det = detS(v, mode);
        // log_value(v, det);
        return det;
    }

    /**
     * Set particular value of the effective index, e.g. to one of the values returned by findModes.
     * If it is not proper mode, exception is throw.
     * \param clambda complex wavelength of the mode
     * \return index of the set mode
     */
    size_t setMode(plask::dcomplex clambda, int m = 0);
    
    /**
     * Set particular value of the effective index, e.g. to one of the values returned by findModes.
     * If it is not proper mode, exception is throw.
     * \param lambda wavelength of the mode
     * \param loss modal loss (as returned by outLoss)
     * \param m number of the LP_mn mode describing angular dependence
     * \return index of the set mode
     */
    inline size_t setMode(double lambda, double loss, int m=0) {
        return setMode(dcomplex(lambda, -lambda*lambda / (2e7*M_PI) * loss));
    }
    
    /// Receiver for the temperature
    ReceiverFor<Temperature, Geometry2DCylindrical> inTemperature;

    /// Receiver for the gain
    ReceiverFor<Gain, Geometry2DCylindrical> inGain;

    /// Provider for computed resonant wavelength
    ProviderFor<Wavelength>::Delegate outWavelength;

    /// Provider for computed modal extinction
    ProviderFor<ModalLoss>::Delegate outLoss;

    /// Provider of optical field
    ProviderFor<LightIntensity, Geometry2DCylindrical>::Delegate outIntensity;

  protected:

    /// Do we need to compute gain
    bool need_gain;
    
    /// Initialize the solver
    virtual void onInitialize();

    /// Invalidate the data
    virtual void onInvalidate();

    /**
     * Fist stage of computations
     * Perform vertical computations
     */
    void stageOne();

    /// Return S matrix determinant for one stripe
    dcomplex detS1(const dcomplex& x, const std::vector<dcomplex,aligned_allocator<dcomplex>>& NR,
                   const std::vector<dcomplex,aligned_allocator<dcomplex>>& NG, bool save=false);

    /**
     * Compute field weights basing on solution for given stripe. Also compute data for determining vertical fields
     * \param stripe main stripe number
     */
    std::vector<double,aligned_allocator<double>> computeWeights(size_t stripe);

    /// Return S matrix determinant for one stripe
    void computeStripeNNg(size_t stripe);

    /** Integrate horizontal field
     * \param mode mode to integrate
     */
    double integrateBessel(const Mode& mode) const;

    /// Return S matrix determinant for the whole structure
    dcomplex detS(const plask::dcomplex& v, Mode& mode, bool save=false);

    /// Insert mode to the list or return the index of the exiting one
    size_t insertMode(const Mode& mode) {
        for (size_t i = 0; i != modes.size(); ++i)
            if (modes[i] == mode) return i;
        modes.push_back(mode);
        outWavelength.fireChanged();
        outLoss.fireChanged();
        outIntensity.fireChanged();
        return modes.size()-1;
    }

    /// Return number of found modes
    size_t nmodes() const {
        return modes.size();
    }
    
    /**
     * Return mode wavelength
     * \param n mode number
     */
    double getWavelength(size_t n) {
        if (n >= modes.size()) throw NoValue(Wavelength::NAME);
        return real(2e3*M_PI / (k0 * (1. - modes[n].freqv/2.)));
    }

    /**
     * Return mode modal loss
     * \param n mode number
     */
    double getModalLoss(size_t n) {
        if (n >= modes.size()) throw NoValue(ModalLoss::NAME);
        return imag(2e4 * k0 * (1. - modes[n].freqv/2.));  // 2e4  2/µm -> 2/cm
    }
    
    /// Method computing the distribution of light intensity
    DataVector<const double> getLightIntenisty(int num, const MeshD<2>& dst_mesh, InterpolationMethod=DEFAULT_INTERPOLATION);
    
  private:
    template <typename MeshT>
    bool getLightIntenisty_Efficient(size_t num, const MeshD<2>& dst_mesh, DataVector<double>& results);
};


}}} // namespace plask::solvers::effective

#endif // PLASK__MODULE_OPTICAL_EFM_HPP

