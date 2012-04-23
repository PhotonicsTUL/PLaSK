#ifndef PLASK__MODULE_OPTICAL_EIM_HPP
#define PLASK__MODULE_OPTICAL_EIM_HPP

#include <limits>

#include <plask/plask.hpp>

#include "rootdigger.h"

namespace plask { namespace eim {

/**
 * Module performing calculations in 2D Cartesian space using effective index method
 */
class EffectiveIndex2dModule: public ModuleCartesian2d {

    friend class RootDigger;

    RootDigger rootdigger;

    /// The mesh used for cutting the structure into one-dimentional stripes
    shared_ptr<RectilinearMesh2d> mesh;

    Data2dLog<dcomplex,double> log_value;

  public:

    // Parameters for rootdigger
    double tolx,        ///< absolute tolerance on the argument
           tolf_min,    ///< sufficient tolerance on the function value
           tolf_max,    ///< required tolerance on the function value
           maxstep;     ///< maximum step in one iteration
    int maxiterations;  ///< maximum number of iterations

    EffectiveIndex2dModule();

    virtual std::string getName() const { return "Optical: Effective Index Method 2D"; }

    virtual std::string getDescription() const {
        return "Calculate optical modes and optical field distribution using the effective index method "
               "in Cartesian two-dimentional space.";
    }

    /**
     * Set new geometry for the module
     *
     * @param new_geometry new geometry space
     */
    void setGeometry(const shared_ptr<Space2dCartesian>& new_geometry) {
        ModuleCartesian2d::setGeometry(new_geometry);
        auto child = new_geometry->getChild();
        if (!child) throw NoChildException();
        mesh = make_shared<RectilinearMesh2d>(child);
    }

    /**
     * Set the simple mesh based on the geometry bounding boxes.
     **/
    void setSimpleMesh() {
        log(LOG_INFO, "Creating automatic simple mesh");
        auto child = geometry->getChild();
        if (!child) throw NoChildException();
        mesh = make_shared<RectilinearMesh2d>(child);
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
        mesh = make_shared<RectilinearMesh2d>(meshx, meshxy.c1);
    }

    /**
     * Set up the mesh. Horizontal division is provided while vertical one is created basing on the geometry bounding boxes.
     *
     * \param meshx horizontal mesh
     **/
    void setMesh(const shared_ptr<RectilinearMesh1d>& meshx) {
        log(LOG_INFO, "Setting mesh");
        setMesh(*meshx);
    }

    /**
     * Set up the mesh. Both horizontal and vertical divisions are provided
     *
     * \param meshxy the new mesh
     **/
    void setMesh(const shared_ptr<RectilinearMesh2d>& meshxy) {
        mesh = meshxy;
    }



    /**
     * Find the mode around the specified propagation constant.
     *
     * This method remembers the determined mode, for rietrieval of the field profiles.
     *
     * \param neff initial effective index to search the mode around
     * \return determined propagation constant
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
     * \return vector of determined propagation constants
     */
    std::vector<dcomplex> findModes(dcomplex neff1, dcomplex neff2, int steps=100, int nummodes=std::numeric_limits<int>::max());


    /**
     * Find approximate modes by scanning the desired range
     *
     * \param neff1 one end of the range to browse
     * \param neff2 another end of the range to browse
     * \param steps number of steps for range browsing
     * \return vector of determined potential propagation constants
     */
    std::vector<dcomplex> findModesMap(dcomplex neff1, dcomplex neff2, int steps=100);


    /// Receiver of the wavelength
    ReceiverFor<Wavelength> inWavelength;

    /// Receiver for temperature
    ReceiverFor<Temperature, Space2dCartesian> inTemperature;

    /// Provider for computed effective index
    ProviderFor<PropagationConstant>::WithValue outNeff;

    /// Provider of optical field
    ProviderFor<OpticalIntensity, Space2dCartesian>::Delegate outIntensity;


  private:

    /// Return function value for root digger
    dcomplex char_val(dcomplex x) { return 0.; /* TODO */ }

    /// Method computing the distribution of light intensity
    shared_ptr<const std::vector<double>> getLightIntenisty(const Mesh<2>& dst_mesh, InterpolationMethod method=DEFAULT_INTERPOLATION);

};


}} // namespace plask::eim

#endif // PLASK__MODULE_OPTICAL_EIM_HPP
