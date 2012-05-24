#ifndef PLASK__MODULE_OPTICAL_EIM_HPP
#define PLASK__MODULE_OPTICAL_EIM_HPP

#include <limits>

#include <Eigen/Dense>

#include <plask/plask.hpp>

#include "rootdigger.h"

namespace plask { namespace modules { namespace eim {

/**
 * Module performing calculations in 2D Cartesian space using effective index method
 */
struct EffectiveIndex2dModule: public ModuleWithMesh<Space2dCartesian, RectilinearMesh2d> {

    enum Symmetry {
        NO_SYMMETRY,
        SYMMETRY_POSITIVE,
        SYMMETRY_NEGATIVE
    };

    enum Polarization {
        TE,
        TM,
    };

  private:

    friend class RootDigger;

    /// Logger for char_val
    Data2dLog<dcomplex,double> log_value;

    /// Cached refractive indices
    std::vector<std::vector<dcomplex>> nrCache;

    /// Computed effective indices for each stripe
    std::vector<dcomplex> stripeNeffs;

    /// Number of stripe to start from
    size_t xbegin;

    /// Old polarization
    Polarization old_polarization;

  public:

    Polarization polarization;  ///< Chosen light polarization

    Symmetry symmetry;  ///< Symmetry of the searched modes

    double outer_distance; ///< Distance outside outer borders where material is sampled

    // Parameters for rootdigger
    double tolx,        ///< Absolute tolerance on the argument
           tolf_min,    ///< Sufficient tolerance on the function value
           tolf_max,    ///< Required tolerance on the function value
           maxstep;     ///< Maximum step in one iteration
    int maxiterations;  ///< Maximum number of iterations

    EffectiveIndex2dModule();

    virtual std::string getName() const { return "Effective Index Method 2D"; }

    virtual std::string getDescription() const {
        return "Calculate optical modes and optical field distribution using the effective index method "
               "in Cartesian two-dimensional space.";
    }

    /**
     * Set the simple mesh based on the geometry bounding boxes.
     **/
    void setSimpleMesh() {
        log(LOG_INFO, "Creating simple mesh");
        if (!geometry) throw NoChildException();
        auto child = geometry->getChild();
        if (!child) throw NoChildException();
        setMesh(make_shared<RectilinearMesh2d>(child));
    }

    /**
     * Set up the horizontal mesh. Horizontal division is provided while vertical one is created basing on the geometry bounding boxes.
     *
     * \param meshx horizontal mesh
     **/
    void setHorizontalMesh(const RectilinearMesh1d& meshx) {
        log(LOG_INFO, "Setting horizontal mesh");
        if (!geometry) throw NoChildException();
        auto child = geometry->getChild();
        if (!child) throw NoChildException();
        shared_ptr<RectilinearMesh2d> meshxy = make_shared<RectilinearMesh2d>(child);
        meshxy->tran() = meshx;
        setMesh(meshxy);
    }

    // /**
    //  * Get the position of the matching interface.
    //  *
    //  * \return index of the vertical mesh, where interface is set
    //  */
    // inline size_t getInterface() { return interface; }
    //
    // /**
    //  * Set the position of the matching interface.
    //  *
    //  * \param index index of the vertical mesh, where interface is set
    //  */
    // inline void setInterface(size_t index) {
    //     if (!mesh) setSimpleMesh();
    //     if (index < 0 || index >= mesh->up().size())
    //         throw BadInput(getId(), "wrong interface position");
    //     log(LOG_DEBUG, "Setting interface at postion %g (mesh index: %d)",  mesh->up()[index], index);
    //     interface = index;
    // }
    //
    // /**
    //  * Set the position of the matching interface at the top of the provided geometry element
    //  *
    //  * \param path path to the element in the geometry
    //  */
    // void setInterfaceOn(const PathHints& path) {
    //     if (!mesh) setSimpleMesh();
    //     auto boxes = geometry->getLeafsBoundingBoxes(path);
    //     if (boxes.size() != 1) throw NotUniqueElementException();
    //     interface = std::lower_bound(mesh->up().begin(), mesh->up().end(), boxes[0].upper.up) - mesh->up().begin();
    //     if (interface >= mesh->up().size()) interface = mesh->up().size() - 1;
    //     log(LOG_DEBUG, "Setting interface at postion %g (mesh index: %d)",  mesh->up()[interface], interface);
    // }

    /**
     * Find the mode around the specified effective index.
     *
     * This method remembers the determined mode, for retrieval of the field profiles.
     *
     * \param neff initial effective index to search the mode around
     * \return determined effective index
     **/
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


    /// Receiver of the wavelength
    ReceiverFor<Wavelength> inWavelength;

    /// Receiver for temperature
    ReceiverFor<Temperature, Space2dCartesian> inTemperature;

    /// Provider for computed effective index
    ProviderFor<EffectiveIndex>::WithValue outNeff;

    /// Provider of optical field
    ProviderFor<OpticalIntensity, Space2dCartesian>::Delegate outIntensity;

  protected:

    /// Initialize the module
    virtual void onInitialize();

    /// Invalidate the data
    virtual void onInvalidate();

  private:

    dcomplex k0;        ///< Cache of the normalized frequency

    /**
     * Fist stage of computations
     * Update the refractive indices cache and perform vertical computations
     */
    void stageOne();

    /// Return the effective index of a single vertical stripe, optionally also computing fields
    Eigen::Matrix2cd getMatrix1(const dcomplex& neff, const std::vector<dcomplex>& NR);

    /// Return S matrix determinant for one stripe
    dcomplex detS1(const dcomplex& x, const std::vector<dcomplex>& NR);

    /// Return the  effective index of the whole structure, optionally also computing fields
    Eigen::Matrix2cd getMatrix(const dcomplex& neff);

    /// Return S matrix determinant for the whole structure
    dcomplex detS(const dcomplex& x);

    /// Method computing the distribution of light intensity
    const DataVector<double> getLightIntenisty(const Mesh<2>& dst_mesh, InterpolationMethod method=DEFAULT_INTERPOLATION);

};


}}} // namespace plask::modules::eim

#endif // PLASK__MODULE_OPTICAL_EIM_HPP
