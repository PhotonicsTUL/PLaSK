#ifndef PLASK__MODULE_OPTICAL_EIM_HPP
#define PLASK__MODULE_OPTICAL_EIM_HPP

#include <limits>

#include <plask/plask.hpp>

#include "rootdigger.h"

namespace plask { namespace modules { namespace eim {

/**
 * Module performing calculations in 2D Cartesian space using effective index method
 */
class EffectiveIndex2dModule: public ModuleOver<Space2dCartesian> {

    friend class RootDigger;

    RootDigger rootdigger;

    /// The mesh used for cutting the structure into one-dimensional stripes
    shared_ptr<RectilinearMesh2d> mesh;

    /// Logger for char_val
    Data2dLog<dcomplex,double> log_value;

    /// Information if the geometry or mesh has changed since last cache
    bool changed;

    /// Cached sizes of mesh cells
    std::vector<double> dTran, dUp;

    /// Cached refractive indices
    std::vector<std::vector<dcomplex>> nrCache;

  public:

    enum Symmetry {
        NO_SYMMETRY,
        SYMMETRY_POSITIVE,
        SYMMETRY_NEGATIVE
    };

    Symmetry symmetry;  ///< Structure symmetry

    // Parameters for rootdigger
    double tolx,        ///< Absolute tolerance on the argument
           tolf_min,    ///< Sufficient tolerance on the function value
           tolf_max,    ///< Required tolerance on the function value
           maxstep;     ///< Maximum step in one iteration
    int maxiterations;  ///< Maximum number of iterations

    EffectiveIndex2dModule();

    virtual std::string getName() const { return "Optical: Effective Index Method 2D"; }

    virtual std::string getDescription() const {
        return "Calculate optical modes and optical field distribution using the effective index method "
               "in Cartesian two-dimensional space.";
    }

    /**
     * Set new geometry for the module
     *
     * @param new_geometry new geometry space
     */
    virtual void setGeometry(const shared_ptr<Space2dCartesian>& new_geometry) {
        ModuleOver<Space2dCartesian>::setGeometry(new_geometry);
        symmetry = geometry->isSymmetric(CalculationSpace::DIRECTION_TRAN)? SYMMETRY_POSITIVE : NO_SYMMETRY;
        auto child = new_geometry->getChild();
        if (!child) throw NoChildException();
        setMesh(RectilinearMesh2d(child));
    }

    /**
     * Set up the mesh. Both horizontal and vertical divisions are provided
     *
     * \param meshxy the new mesh
     **/
    void setMesh(const RectilinearMesh2d& meshxy);

    /**
     * Set up the mesh. Both horizontal and vertical divisions are provided
     *
     * \param meshxy the new mesh
     **/
    void setMesh(const shared_ptr<RectilinearMesh2d>& meshxy) {
        setMesh(*meshxy);
    }

    /**
     * Set the simple mesh based on the geometry bounding boxes.
     **/
    void setSimpleMesh() {
        log(LOG_INFO, "Creating simple mesh");
        auto child = geometry->getChild();
        if (!child) throw NoChildException();
        setMesh(RectilinearMesh2d(child));
    }

    /**
     * Set up the mesh. Horizontal division is provided while vertical one is created basing on the geometry bounding boxes.
     *
     * \param meshx horizontal mesh
     **/
    void setMesh(const RectilinearMesh1d& meshx) {
        log(LOG_INFO, "Setting horizontal mesh");
        auto child = geometry->getChild();
        if (!child) throw NoChildException();
        RectilinearMesh2d meshxy(child);
        meshxy.tran() = meshx;
        setMesh(meshxy);
    }

    /**
     * Set up the mesh. Horizontal division is provided while vertical one is created basing on the geometry bounding boxes.
     *
     * \param meshx horizontal mesh
     **/
    void setMesh(const shared_ptr<RectilinearMesh1d>& meshx) {
        setMesh(*meshx);
    }



    /**
     * Find the mode around the specified propagation constant.
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


  private:

    /// Cache the effective indices
    void updateCache();

    /// Return function value for root digger
    dcomplex char_val(dcomplex x);

    /// Method computing the distribution of light intensity
    shared_ptr<const std::vector<double>> getLightIntenisty(const Mesh<2>& dst_mesh, InterpolationMethod method=DEFAULT_INTERPOLATION);

};


}}} // namespace plask::modules::eim

#endif // PLASK__MODULE_OPTICAL_EIM_HPP
