#ifndef PLASK__MODULE_OPTICAL_EFM_HPP
#define PLASK__MODULE_OPTICAL_EFM_HPP

#include <limits>

#include <Eigen/Dense>

#include<Eigen/StdVector> // This is needed to ensure the proper alignment of Eigen::Vector2cd in std::vector
#ifndef PLASK_EIGEN_STL_VECTOR_SPECIALIZATION_DEFINED
#   define PLASK_EIGEN_STL_VECTOR_SPECIALIZATION_DEFINED
    EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector2cd)
#endif

#include <plask/plask.hpp>

#include "broyden.h"

namespace plask { namespace solvers { namespace effective {

/**
 * Solver performing calculations in 2D Cartesian space using effective index method
 */
struct EffectiveFrequencyCylSolver: public SolverWithMesh<Geometry2DCylindrical, RectilinearMesh2D> {

  protected:

    friend class RootDigger;

    /// Logger for char_val
    Data2DLog<dcomplex,dcomplex> log_value;

    /// Cached refractive indices
    std::vector<std::vector<dcomplex>> nrCache;

    /// Cached group indices
    std::vector<std::vector<dcomplex>> ngCache;

    /// Computed horizontal fields
    std::vector<Eigen::Vector2cd> fieldR, fieldZ;

    /// Did we compute fields for current state?
    bool have_fields;

    /// Computed effective frequencies for each stripe
    std::vector<dcomplex> veffs;

    /// Computed weighted indices for each stripe
    std::vector<dcomplex> nng;

    /// Old value of the l number (to detect changes)
    int old_l;

    /// Old value of k0 to detect changes
    dcomplex old_k0;

    /// Stored frequency parameter for field calculations
    dcomplex v;

    /// Stored vertical propagation constants for field calculations
    std::vector<dcomplex> betaz;

  public:

    /// Number of the LP_lm mode describing angular dependence
    int l;

    /// Current value of reference normalized frequency
    dcomplex k0;

    double outer_distance; ///< Distance outside outer borders where material is sampled

    // Parameters for rootdigger
    RootDigger::Params root;
    RootDigger::Params striperoot;

    EffectiveFrequencyCylSolver(const std::string& name="");

    virtual std::string getClassName() const { return "EffectiveFrequencyCyl"; }

    virtual std::string getClassDescription() const {
        return "Calculate optical modes and optical field distribution using the effective index method "
               "in Cartesian two-dimensional space.";
    }

    virtual void loadConfiguration(plask::XMLReader& reader, plask::Manager& manager);

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
        shared_ptr<RectilinearMesh2D> meshxy = RectilinearMesh2DSimpleGenerator().generate(geometry->getChild());
        meshxy->tran() = meshx;
        setMesh(meshxy);
    }

    /**
     * Find the mode around the specified effective index.
     *
     * This method remembers the determined mode, for retrieval of the field profiles.
     *
     * \param lambda0 initial wavelength close to the solution
     * \return determined effective index
     */
    dcomplex computeMode(dcomplex lambda0);


    /**
     * Find the modes within the specified range
     *
     * This method \b does \b not remember the determined modes!
     *
     * \param lambda1 one end of the range to browse
     * \param lambda2 another end of the range to browse
     * \param steps number of steps for range browsing
     * \param nummodes maximum number of modes to find
     * \return vector of determined effective indices
     */
    std::vector<dcomplex> findModes(dcomplex lambda1, dcomplex lambda2, unsigned steps=100, unsigned nummodes=std::numeric_limits<unsigned>::max());


    /**
     * Find approximate modes by scanning the desired range
     *
     * \param lambda1 one end of the range to browse
     * \param lambda2 another end of the range to browse
     * \param steps number of steps for range browsing
     * \return vector of determined potential effective indices
     */
    std::vector<dcomplex> findModesMap(dcomplex lambda1, dcomplex lambda2, unsigned steps=100);

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
     * \param veff stripe effective frequency to use
     */
    dcomplex getStripeDeterminantV(size_t stripe, dcomplex veff) {
        bool invalid = !initCalculation();
        dcomplex result = detS1(veff, nrCache[stripe], ngCache[stripe]);
        if (invalid) invalidate();
        return result;
    }

    /**
     * Compute modal determinant for the whole matrix
     * \param v frequency parameter
     */
    dcomplex getDeterminantV(dcomplex v) {
        stageOne();
        return detS(v);
    }

    /**
     * Compute modal determinant for the whole matrix
     * \param lambda wavelength
     */
    dcomplex getDeterminant(dcomplex lambda) {
        if (isnan(k0.real())) k0 = 2e3*M_PI / lambda;
        v =  2. - 4e3*M_PI / lambda / k0;
        stageOne();
        dcomplex det = detS(v);
        log_value(v, det);
        return det;
    }


    /// Receiver for the temperature
    ReceiverFor<Temperature, Geometry2DCylindrical> inTemperature;

    /// Receiver for the gain
    ReceiverFor<Gain, Geometry2DCylindrical> inGain;

    /// Provider for computed resonant wavelength
    ProviderFor<Wavelength>::WithValue outWavelength;

    /// Provider of optical field
    ProviderFor<OpticalIntensity, Geometry2DCylindrical>::Delegate outIntensity;

  protected:

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
    dcomplex detS1(const dcomplex& x, const std::vector<dcomplex>& NR, const std::vector<dcomplex>& NG);

    /// Return S matrix determinant for one stripe
    void computeStripeNNg(std::size_t stripe);

    /// Get matrix transforming amplitudes of J and Y from (i-1)-th layer to i-th one
    Eigen::Matrix2cd getMatrix(plask::dcomplex v, std::size_t i);

    /// Return S matrix determinant for the whole structure
    dcomplex detS(const plask::dcomplex& v);

    /// Method computing the distribution of light intensity
    const DataVector<double> getLightIntenisty(const plask::MeshD<2>& dst_mesh, plask::InterpolationMethod=DEFAULT_INTERPOLATION);

  private:
    template <typename MeshT>
    bool getLightIntenisty_Efficient(const MeshD<2>& dst_mesh, DataVector<double>& results);
};


}}} // namespace plask::solvers::effective

#endif // PLASK__MODULE_OPTICAL_EFM_HPP
