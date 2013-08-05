#ifndef PLASK__MODULE_OPTICAL_EIM_HPP
#define PLASK__MODULE_OPTICAL_EIM_HPP

#include <limits>

#include <plask/plask.hpp>

#include "broyden.h"
#include "bisection.h"

namespace plask { namespace solvers { namespace effective {

/**
 * Solver performing calculations in 2D Cartesian space using effective index method
 */
struct EffectiveIndex2DSolver: public SolverWithMesh<Geometry2DCartesian, RectilinearMesh2D> {

    /// Mode symmetry in horizontal axis
    enum Symmetry {
        NO_SYMMETRY,
        SYMMETRY_POSITIVE,
        SYMMETRY_NEGATIVE
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


  protected:

    friend struct RootDigger;

    size_t xbegin,  ///< First element of horizontal mesh to consider
           xend,    ///< Last element of horizontal mesh to consider
           ybegin,  ///<First element of vertical mesh to consider
           yend;    ///< Last element of vertical mesh to consider

    /// Logger for determinant
    Data2DLog<dcomplex,dcomplex> log_value;

    /// Cached refractive indices
    std::vector<std::vector<dcomplex,aligned_allocator<dcomplex>>> nrCache;

    /// Computed horizontal and vertical fields
    std::vector<Field,aligned_allocator<Field>> xfields, yfields;

    /// Field confinement weights in stripes
    std::vector<double,aligned_allocator<double>> weights;

    /// Did we compute fields for current Neff?
    bool have_fields;

    /// Computed effective indices for each stripe
    std::vector<dcomplex,aligned_allocator<dcomplex>> epsilons;

    /// Should stripe indices be recomputed
    bool recompute_neffs;

    double stripex;             ///< Position of the main stripe

    Polarization polarization;  ///< Chosen light polarization

    Symmetry symmetry;          ///< Symmetry of the searched modes

    Emission emission;          ///< Direction of laser emission

  public:

    dcomplex vneff;             ///< Vertical effective index of the main stripe

    double outdist;             ///< Distance outside outer borders where material is sampled

    /// Mirror reflectivities
    boost::optional<std::pair<double,double>> mirrors;

    /// Parameters for main rootdigger
    RootDigger::Params root;

    /// Parameters for sripe rootdigger
    RootDigger::Params stripe_root;

    EffectiveIndex2DSolver(const std::string& name="");

    virtual std::string getClassName() const { return "optical.EffectiveIndex2D"; }

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
        have_fields = false;
    }

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
    }

    /// \return current polarization
    Symmetry getSymmetry() const { return symmetry; }

    /**
     * Set new symmetry
     * \param sym new symmetry
     */
    void setSymmetry(Symmetry sym) {
        symmetry = sym;
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
    void setHorizontalMesh(const RectilinearMesh1D& meshx) {
        writelog(LOG_INFO, "Setting horizontal mesh");
        if (!geometry) throw NoChildException();
        auto meshxy = make_shared<RectilinearMesh2D>(*RectilinearMesh2DSimpleGenerator()(geometry->getChild()));
        meshxy->axis0 = meshx;
        setMesh(meshxy);
    }

    /**
     * Find the vertical effective indices within the specified range
     *
     * This method \b does \b not remember the determined modes!
     *
     * \param neff1 one corner of the range to browse
     * \param neff2 another corner of the range to browse
     * \param resteps minimum number of steps to check function value on real contour
     * \param imsteps minimum number of steps to check function value on imaginary contour
     * \param eps approximate error for integrals
     * \return vector of determined effective indices
     */
    std::vector<dcomplex> findVeffs(plask::dcomplex neff1=0., plask::dcomplex neff2=0., size_t resteps=256, size_t imsteps=64, dcomplex eps=dcomplex(1e-6,1e-9));

    /**
     * Find the mode around the specified effective index.
     *
     * This method remembers the determined mode, for retrieval of the field profiles.
     *
     * \param neff initial effective index to search the mode around
     * \return determined effective index
     */
    dcomplex computeMode(dcomplex neff);

    /**
     * Find the modes within the specified range
     *
     * This method \b does \b not remember the determined modes!
     *
     * \param neff1 one corner of the range to browse
     * \param neff2 another corner of the range to browse
     * \param resteps minimum number of steps to check function value on real contour
     * \param imsteps minimum number of steps to check function value on imaginary contour
     * \param eps approximate error for integrals
     * \return vector of determined effective indices
     */
    std::vector<dcomplex> findModes(plask::dcomplex neff1=0., plask::dcomplex neff2=0., size_t resteps=256, size_t imsteps=64, dcomplex eps=dcomplex(1e-6,1e-9));

    /**
     * Set particular value of the effective index, e.g. to one of the values returned by findModes.
     * If it is not proper mode, exception is throw
     *
     * \param neff effective index of the mode
     */
    void setMode(dcomplex neff);

    /**
     * Compute determinant for a single stripe
     * \param stripe index of stripe
     * \param neff effective index to use
     */
    dcomplex getStripeDeterminant(size_t stripe, dcomplex neff) { initCalculation(); return detS1(neff, nrCache[stripe]); }

    /**
     * Compute modal determinant for the whole matrix
     * \param neff effective index to use
     */
    dcomplex getDeterminant(dcomplex neff) {
        stageOne();
        dcomplex det = detS(neff);
        // log_value(neff, det);
        return det;
    }


    /// Receiver of the wavelength
    ReceiverFor<Wavelength> inWavelength;

    /// Receiver for the temperature
    ReceiverFor<Temperature, Geometry2DCartesian> inTemperature;

    /// Receiver for the gain
    ReceiverFor<Gain, Geometry2DCartesian> inGain;

    /// Provider for computed effective index
    ProviderFor<EffectiveIndex>::WithValue outNeff;

    /// Provider of optical field
    ProviderFor<OpticalIntensity, Geometry2DCartesian>::Delegate outIntensity;

  protected:

    /// Initialize the solver
    virtual void onInitialize();

    /// Invalidate the data
    virtual void onInvalidate();

    dcomplex k0;        ///< Cache of the normalized frequency [1/µm]

    /// Compute mirror losses for specified effective index
    double getMirrorLosses(double n) {
        const double lambda = inWavelength();
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
     * Normalize horizontal fields, so multiplying OpticalIntensity by power gives proper LightIntensity in (V/m)²
     * \param kx computed horizontal propagation constants
     */
    void normalizeFields(const std::vector<dcomplex,aligned_allocator<dcomplex>>& kx);

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
    dcomplex detS(const dcomplex& x, bool save=false);

    /// Method computing the distribution of light intensity
    DataVector<const double> getLightIntenisty(const plask::MeshD<2>& dst_mesh, plask::InterpolationMethod=DEFAULT_INTERPOLATION);

  private:
    template <typename MeshT>
    bool getLightIntenisty_Efficient(const plask::MeshD<2>& dst_mesh, DataVector<double>& result,
                                     const std::vector<dcomplex,aligned_allocator<dcomplex>>& kx,
                                     const std::vector<dcomplex,aligned_allocator<dcomplex>>& ky);

};


}}} // namespace plask::solvers::effective

#endif // PLASK__MODULE_OPTICAL_EIM_HPP
