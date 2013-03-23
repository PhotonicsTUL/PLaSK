#ifndef PLASK__MODULE_OPTICAL_EIM_HPP
#define PLASK__MODULE_OPTICAL_EIM_HPP

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
struct EffectiveIndex2DSolver: public SolverWithMesh<Geometry2DCartesian, RectilinearMesh2D> {

    enum Symmetry {
        NO_SYMMETRY,
        SYMMETRY_POSITIVE,
        SYMMETRY_NEGATIVE
    };

    enum Polarization {
        TE,
        TM,
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
    std::vector<std::vector<dcomplex>> nrCache;

    /// Computed horizontal and vertical fields
    std::vector<Eigen::Vector2cd> fieldX, fieldY;

    /// Field confinement weights in stripes
    std::vector<double> fieldWeights;

    /// Did we compute fields for current Neff?
    bool have_fields;

    /// Computed effective indices for each stripe
    std::vector<dcomplex> stripeNeffs;

    /// Old polarization
    Polarization old_polarization;

  public:

    Polarization polarization;  ///< Chosen light polarization

    Symmetry symmetry;  ///< Symmetry of the searched modes

    double outdist; ///< Distance outside outer borders where material is sampled

    // Parameters for rootdigger
    RootDigger::Params root;
    RootDigger::Params stripe_root;

    EffectiveIndex2DSolver(const std::string& name="");

    virtual std::string getClassName() const { return "optical.EffectiveIndex2D"; }

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
        shared_ptr<RectilinearMesh2D> meshxy = RectilinearMesh2DSimpleGenerator().generate(geometry->getChild());
        meshxy->axis0 = meshx;
        setMesh(meshxy);
    }

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
     * \param neff1 one end of the range to browse
     * \param neff2 another end of the range to browse
     * \param steps number of steps for range browsing
     * \param nummodes maximum number of modes to find
     * \return vector of determined effective indices
     */
    std::vector<dcomplex> findModes(dcomplex neff1, dcomplex neff2, unsigned steps=100, unsigned nummodes=std::numeric_limits<unsigned>::max());


    /**
     * Find approximate modes by scanning the desired range
     *
     * \param neff1 one end of the range to browse
     * \param neff2 another end of the range to browse
     * \param steps number of steps for range browsing
     * \return vector of determined potential effective indices
     */
    std::vector<dcomplex> findModesMap(dcomplex neff1, dcomplex neff2, unsigned steps=100);

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
        log_value(neff, det);
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

    dcomplex k0;        ///< Cache of the normalized frequency

    /// Compute mirror losses for specified effective index
    double getMirrorLosses(const dcomplex& neff) {
        const double lambda = inWavelength();
        const double n = real(neff); // TODO is this really correct?
        const double n1 = real(geometry->getFrontMaterial()->Nr(lambda, 300.)),
                     n2 = real(geometry->getBackMaterial()->Nr(lambda, 300.));
        const double R1 = abs((n-n1) / (n+n1)),
                     R2 = abs((n-n2) / (n+n2));
        return lambda * std::log(R1*R2) / (4e3 * M_PI * geometry->getExtrusion()->getLength());
    }

    /**
     * Fist stage of computations
     * Perform vertical computations
     */
    void stageOne();

    /// Return S matrix determinant for one stripe
    dcomplex detS1(const dcomplex& x, const std::vector<dcomplex>& NR);

    /// Return S matrix determinant for the whole structure
    dcomplex detS(const dcomplex& x);

    /// Method computing the distribution of light intensity
    DataVector<const double> getLightIntenisty(const plask::MeshD<2>& dst_mesh, plask::InterpolationMethod=DEFAULT_INTERPOLATION);

  private:
    template <typename MeshT>
    bool getLightIntenisty_Efficient(const plask::MeshD<2>& dst_mesh, DataVector<double>& result,
                                     const std::vector<dcomplex>& betax, const std::vector<dcomplex>& betay);

};


}}} // namespace plask::solvers::effective

#endif // PLASK__MODULE_OPTICAL_EIM_HPP
