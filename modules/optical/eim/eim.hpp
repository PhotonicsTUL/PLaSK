#ifndef PLASK__MODULE_OPTICAL_EIM_HPP
#define PLASK__MODULE_OPTICAL_EIM_HPP

#include <plask/plask.hpp>

namespace plask { namespace eim {


/**
 * Module performing calculations in 2D Cartesian space using effective index method
 */
struct EffectiveIndex2dModule: public Module {

    /// Default constructor creates default mesh based on geometry
    EffectiveIndex2dModule() : outIntensity(this, &EffectiveIndex2dModule::getLightIntenisty) {
        setSimpleMesh();
    }

    /// Constructor with 1D mesh provided
    EffectiveIndex2dModule(const RectilinearMesh1d& mesh) : outIntensity(this, &EffectiveIndex2dModule::getLightIntenisty) {
        setMesh(mesh);
    }

    /// Constructor with 2D mesh provided
    EffectiveIndex2dModule(const RectilinearMesh2d& mesh) : outIntensity(this, &EffectiveIndex2dModule::getLightIntenisty) {
        setMesh(mesh);
    }

    /**
     * Set the simple mesh based on the geometry bounding boxes.
     **/
    void setSimpleMesh();


    /**
     * Set up the mesh. Horizontal division is provided while vertical one is created basing on the geometry bounding boxes.
     *
     * @param meshx Horizontal mesh
     **/
    void setMesh(const RectilinearMesh1d& meshx);


    /**
     * Set up the mesh. Both horizontal and vertical divisions are provided
     *
     * @param meshxy The mesh
     **/
    void setMesh(const RectilinearMesh2d& meshxy);


    /// Provider of optical field
    ProviderFor<OpticalIntensity, space::Cartesian2d>::Delegate outIntensity;

    /// Receiver for temperature
    plask::ReceiverFor<Temperature, space::Cartesian2d> inTemperature;

    /// Method computing the distribution of light intensity
    shared_ptr<const std::vector<double>> getLightIntenisty(const Mesh<space::Cartesian2d>& dst_mesh, InterpolationMethod method=DEFAULT_INTERPOLATION);

  private:

    /// The mesh used for cutting the structure into one-dimentional stripes
    RectilinearMesh2d mesh;

};


}} // namespace plask::eim

#endif // PLASK__MODULE_OPTICAL_EIM_HPP
