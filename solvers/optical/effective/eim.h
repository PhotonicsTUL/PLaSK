#ifndef PLASK__MODULE_OPTICAL_EIM_HPP
#define PLASK__MODULE_OPTICAL_EIM_HPP

#include <limits>

#include <plask/plask.hpp>

#include "muller.h"
#include "bisection.h"

namespace plask { namespace solvers { namespace effective {

/**
 * Solver performing calculations in 2D Cartesian space using effective index method
 */
struct EffectiveIndex2DSolver: public SolverWithMesh<Geometry2DCartesian, RectilinearMesh2D> {

    /// Mode symmetry in horizontal axis
    enum Symmetry {
        SYMMETRY_DEFAULT,
        SYMMETRY_POSITIVE,
        SYMMETRY_NEGATIVE,
        SYMMETRY_NONE
    };

    /// Mode polarization
    enum Polarization {
        TE,
        TM,
    };

    /// Direction of the possible emission
    enum Emission {
        FRONT,
        BACK
    };

    struct Field {
        dcomplex F, B;
        Field() = default;
        Field(dcomplex f, dcomplex b): F(f), B(b) {}
        Field operator*(dcomplex a) const { return Field(F*a, B*a); }
        Field operator/(dcomplex a) const { return Field(F/a, B/a); }
        Field operator*=(dcomplex a) { F *= a; B *= a; return *this; }
        Field operator/=(dcomplex a) { F /= a; B /= a; return *this; }
    };

    struct Matrix {
        dcomplex ff, fb, bf, bb;
        Matrix() = default;
        Matrix(dcomplex t1, dcomplex t2, dcomplex t3, dcomplex t4): ff(t1), fb(t2), bf(t3), bb(t4) {}
        static Matrix eye() { return Matrix(1.,0.,0.,1.); }
        static Matrix diag(dcomplex f, dcomplex b) { return Matrix(f,0.,0.,b); }
        Matrix operator*(const Matrix& T) {
            return Matrix( ff*T.ff + fb*T.bf,   ff*T.fb + fb*T.bb,
                           bf*T.ff + bb*T.bf,   bf*T.fb + bb*T.bb );
        }
        Field solve(const Field& v) {
            return Field(bb*v.F - fb*v.B, -bf*v.F + ff*v.B) / (ff*bb - fb*bf);
        }
    };

    /// Details of the computed mode
    struct Mode {
        EffectiveIndex2DSolver* solver; ///< Solver this mode belongs to
        Symmetry symmetry;              ///< Horizontal symmetry of the modes
        dcomplex neff;                  ///< Stored mode effective index
        bool have_fields;               ///< Did we compute fields for current state?
        std::vector<Field,aligned_allocator<Field>> xfields; ///< Computed horizontal fields
        double power;                   ///< Mode power [mW]

        Mode(EffectiveIndex2DSolver* solver, Symmetry sym): solver(solver), have_fields(false), xfields(solver->xend), power(1.) {
            setSymmetry(sym);
        }

        void setSymmetry(Symmetry sym) {
            if (solver->geometry->isSymmetric(Geometry::DIRECTION_TRAN)) {
                if (sym == SYMMETRY_DEFAULT)
                    sym = SYMMETRY_POSITIVE;
                else if (sym == SYMMETRY_NONE)
                    throw BadInput(solver->getId(), "For symmetric geometry specify positive or negative symmetry");
            } else {
                if (sym == SYMMETRY_DEFAULT)
                    sym = SYMMETRY_NONE;
                else if (sym != SYMMETRY_NONE)
                    throw BadInput(solver->getId(), "For non-symmetric geometry no symmetry may be specified");
            }
            symmetry = sym;
        }

        bool operator==(const Mode& other) const {
            return symmetry == other.symmetry && is_zero( neff - other.neff );
        }
    };


  protected:

    friend struct RootMuller;

    size_t xbegin,  ///< First element of horizontal mesh to consider
           xend,    ///< Last element of horizontal mesh to consider
           ybegin,  ///< First element of vertical mesh to consider
           yend;    ///< Last element of vertical mesh to consider

    /// Logger for determinant
    Data2DLog<dcomplex,dcomplex> log_value;

    /// Cached refractive indices
    std::vector<std::vector<dcomplex,aligned_allocator<dcomplex>>> nrCache;

    /// Computed horizontal and vertical fields
    std::vector<Field,aligned_allocator<Field>> yfields;

    /// Field confinement weights in stripes
    std::vector<double,aligned_allocator<double>> weights;

    /// Computed effective epsilons for each stripe
    std::vector<dcomplex,aligned_allocator<dcomplex>> epsilons;

    /// Should stripe indices be recomputed
    bool recompute_neffs;

    double stripex;             ///< Position of the main stripe

    Polarization polarization;  ///< Chosen light polarization

  public:

    Emission emission;          ///< Direction of laser emission

    dcomplex vneff;             ///< Vertical effective index of the main stripe

    double outdist;             ///< Distance outside outer borders where material is sampled

    /// Mirror reflectivities
    boost::optional<std::pair<double,double>> mirrors;

    /// Parameters for main rootdigger
    RootMuller::Params root;

    /// Parameters for sripe rootdigger
    RootMuller::Params stripe_root;

    /// Computed modes
    std::vector<Mode> modes;

    EffectiveIndex2DSolver(const std::string& name="");

    virtual ~EffectiveIndex2DSolver() {
        inTemperature.changedDisconnectMethod(this, &EffectiveIndex2DSolver::onInputChange);
        inGain.changedDisconnectMethod(this, &EffectiveIndex2DSolver::onInputChange);
    }

    virtual std::string getClassName() const { return "optical.EffectiveIndex2D"; }

    virtual std::string getClassDescription() const {
        return "Calculate optical modes and optical field distribution using the effective index method "
               "in Cartesian two-dimensional space.";
    }

    virtual void loadConfiguration(plask::XMLReader& reader, plask::Manager& manager);

    /// \return position of the main stripe
    double getStripeX() const { return stripex; }

    /**
     * Set position of the main stripe
     * \param x horizontal position of the main stripe
     */
    void setStripeX(double x) {
        stripex = x;
        recompute_neffs = true;
        vneff = 0.;
        if (!modes.empty()) writelog(LOG_DETAIL, "Clearing the computed modes");
        modes.clear();
    }

    /// \return current polarization
    Polarization getPolarization() const { return polarization; }

    /**
     * Set new polarization
     * \param polar new polarization
     */
    void setPolarization(Polarization polar) {
        polarization = polar;
        recompute_neffs = true;
        vneff = 0.;
        if (!modes.empty()) writelog(LOG_DETAIL, "Clearing the computed modes");
        modes.clear();
    }

    /// \return current wavelength
    dcomplex getWavelength() const { return 2e3*M_PI / k0; }

    /**
     * Set new polarization
     * \param polar new polarization
     */
    void setWavelength(dcomplex wavelength) {
        k0 = 2e3*M_PI / wavelength;
        recompute_neffs = true;
        vneff = 0.;
        if (!modes.empty()) writelog(LOG_DETAIL, "Clearing the computed modes");
        modes.clear();
    }

    /**
     * Set the simple mesh based on the geometry bounding boxes.
     **/
    void setSimpleMesh() {
        writelog(LOG_INFO, "Creating simple mesh");
        setMesh(make_shared<RectilinearMesh2DSimpleGenerator>());
    }

    /**
     * Set up the horizontal mesh. Horizontal division is provided while vertical one is created basing on the geometry bounding boxes.
     *
     * \param meshx horizontal mesh
     **/
    void setHorizontalMesh(const RectilinearAxis& meshx) {
        writelog(LOG_INFO, "Setting horizontal mesh");
        if (!geometry) throw NoChildException();
        auto meshxy = make_shared<RectilinearMesh2D>(*RectilinearMesh2DSimpleGenerator()(geometry->getChild()));
        meshxy->axis0 = meshx;
        setMesh(meshxy);
    }

    /**
     * Look for the vertical effective indices within the specified range
     * This method \b does \b not remember the determined modes!
     * \param neff1 one corner of the range to browse
     * \param neff2 another corner of the range to browse
     * \param resteps minimum number of steps to check function value on real contour
     * \param imsteps minimum number of steps to check function value on imaginary contour
     * \param eps approximate error for integrals
     * \return vector of determined effective indices
     */
    std::vector<dcomplex> searchVNeffs(plask::dcomplex neff1=0., plask::dcomplex neff2=0., size_t resteps=256, size_t imsteps=64, dcomplex eps=dcomplex(1e-6,1e-9));

    /**
     * Find the mode around the specified effective index.
     * \param neff initial effective index to search the mode around
     * \param symmetry mode symmetry
     * \return index of found mode
     */
    size_t findMode(dcomplex neff, Symmetry symmetry=SYMMETRY_DEFAULT);

    /**
     * Find the mode between the specified effective indices.
     * \param neff1 initial effective index to search the mode from
     * \param neff1 initial effective index to search the mode to
     * \param symmetry mode symmetry
     * \return index of found mode
     */
    size_t findMode(dcomplex neff1, dcomplex neff2, Symmetry symmetry=SYMMETRY_DEFAULT);

    /**
     * Find the modes within the specified range
     * \param neff1 one corner of the range to browse
     * \param neff2 another corner of the range to browse
     * \param symmetry mode symmetry
     * \param resteps minimum number of steps to check function value on real contour
     * \param imsteps minimum number of steps to check function value on imaginary contour
     * \param eps approximate error for integrals
     * \return vector of indices of found modes
     */
    std::vector<size_t> findModes(dcomplex neff1=0., dcomplex neff2=0., Symmetry symmetry=SYMMETRY_DEFAULT, size_t resteps=256, size_t imsteps=64, dcomplex eps=dcomplex(1e-6,1e-9));

    /**
     * Compute determinant for a single stripe
     * \param neff effective index to use
     */
    dcomplex getVertDeterminant(dcomplex neff) {
        initCalculation();
        return detS1(neff, nrCache[mesh->tran().findIndex(stripex)]);
    }

    /**
     * Compute modal determinant for the whole matrix
     * \param neff effective index to use
     * \param symmetry mode symmetry
     */
    dcomplex getDeterminant(dcomplex neff, Symmetry sym=SYMMETRY_DEFAULT) {
        stageOne();
        Mode mode(this,sym);
        dcomplex det = detS(neff, mode);
        return det;
    }

    /**
     * Set particular value of the effective index, e.g. to one of the values returned by findModes.
     * If it is not proper mode, exception is throw
     * \param neff effective index of the mode
     * \param symmetry mode symmetry
     * \return index of set mode
     */
    size_t setMode(dcomplex neff, Symmetry sym=SYMMETRY_DEFAULT);

    /// Receiver for the temperature
    ReceiverFor<Temperature, Geometry2DCartesian> inTemperature;

    /// Receiver for the gain
    ReceiverFor<Gain, Geometry2DCartesian> inGain;

    /// Provider for computed effective index
    ProviderFor<EffectiveIndex>::Delegate outNeff;

    /// Provider of optical field
    ProviderFor<LightIntensity, Geometry2DCartesian>::Delegate outLightIntensity;

    /// Provider for refractive index
    ProviderFor<RefractiveIndex, Geometry2DCartesian>::Delegate outRefractiveIndex;

  protected:

    /// Slot called when gain has changed
    void onInputChange(ReceiverBase&, ReceiverBase::ChangeReason) {
        recompute_neffs = true;
    }

    /// Initialize the solver
    virtual void onInitialize();

    /// Invalidate the data
    virtual void onInvalidate();

    /// Cache of the normalized frequency [1/µm]
    dcomplex k0;

    /// Do we need to have gain
    bool need_gain;

    /// Compute mirror losses for specified effective mode
    double getMirrorLosses(dcomplex n) {
        const double lambda = real(2e3*M_PI / k0);
        double R1, R2;
        if (mirrors) {
            std::tie(R1,R2) = *mirrors;
        } else {
            const double n1 = real(geometry->getFrontMaterial()->Nr(lambda, 300.)),
                         n2 = real(geometry->getBackMaterial()->Nr(lambda, 300.));
            R1 = abs((n-n1) / (n+n1));
            R2 = abs((n-n2) / (n+n2));
        }
        return lambda * std::log(R1*R2) / (4e3 * M_PI * geometry->getExtrusion()->getLength());
    }

    /**
     * Update refractive index cache
     */
    void updateCache();

    /**
     * Fist stage of computations
     * Perform vertical computations
     */
    void stageOne();

    /**
     * Compute field weights basing on solution for given stripe. Also compute data for determining vertical fields
     * \param stripe main stripe number
     */
    void computeWeights(size_t stripe);

    /**
     * Normalize horizontal fields, so multiplying LightIntensity by power gives proper LightIntensity in (V/m)²
     * \param kx computed horizontal propagation constants
     */
    void normalizeFields(Mode& mode, const std::vector<dcomplex,aligned_allocator<dcomplex>>& kx);

    /**
     * Compute S matrix determinant for one stripe
     * \param x vertical effective index
     * \param NR refractive indices
     * \param save if \c true, the fields are saved to yfields
     */
    dcomplex detS1(const dcomplex& x, const std::vector<dcomplex,aligned_allocator<dcomplex>>& NR, bool save=false);

    /**
     * Return S matrix determinant for the whole structure
     * \param x effective index
     * \param save if \c true, the fields are saved to xfields
     */
    dcomplex detS(const dcomplex& x, Mode& mode, bool save=false);

    /// Insert mode to the list or return the index of the exiting one
    size_t insertMode(const Mode& mode) {
        for (size_t i = 0; i != modes.size(); ++i)
            if (modes[i] == mode) return i;
        modes.push_back(mode);
        outNeff.fireChanged();
        outLightIntensity.fireChanged();
        return modes.size()-1;
    }

    /// Return number of found modes
    size_t nmodes() const {
        return modes.size();
    }

    /**
     * Return mode effective index
     * \param n mode number
     */
    dcomplex getEffectiveIndex(size_t n) {
        if (n >= modes.size()) throw NoValue(EffectiveIndex::NAME);
        return modes[n].neff;
    }

    /// Method computing the distribution of light intensity
    DataVector<const double> getLightIntenisty(int num, const plask::MeshD<2>& dst_mesh, plask::InterpolationMethod=INTERPOLATION_DEFAULT);

    /// Get used refractive index
    DataVector<const Tensor3<dcomplex>> getRefractiveIndex(const MeshD<2>& dst_mesh, double, InterpolationMethod);

  private:
    template <typename MeshT>
    bool getLightIntenisty_Efficient(size_t num, const plask::MeshD<2>& dst_mesh, DataVector<double>& result,
                                     const std::vector<dcomplex,aligned_allocator<dcomplex>>& kx,
                                     const std::vector<dcomplex,aligned_allocator<dcomplex>>& ky);

};


}}} // namespace plask::solvers::effective

#endif // PLASK__MODULE_OPTICAL_EIM_HPP
