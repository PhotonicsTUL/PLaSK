#ifndef PLASK__MODULE_OPTICAL_EIM_HPP
#define PLASK__MODULE_OPTICAL_EIM_HPP

#include <plask/plask.hpp>

namespace plask { namespace eim {


/**
 * Module performing calculations in 2D Cartesian space using effective index method
 */
class EffectiveIndex2dModule: public Module {

    /// Geometry in which the calculations are performed
    shared_ptr<const CartesianExtend> geometry;

    /// The mesh used for cutting the structure into one-dimentional stripes
    RectilinearMesh2d mesh;

  public:

    /**
     * Default constructor creates default mesh based on geometry
     *
     * \param geometry geometry in which the calculations are done
     */
    EffectiveIndex2dModule(shared_ptr<const CartesianExtend> geometry) :
        geometry(geometry), outBeta(NAN), outIntensity(this, &EffectiveIndex2dModule::getLightIntenisty) {
        inTemperature = 300.;
        setSimpleMesh();
    }

    /**
     * Constructor with 1D mesh provided
     *
     * \param geometry geometry in which the calculations are done
     * \param mesh horizontal mesh for dividing geometry
     */
    EffectiveIndex2dModule(shared_ptr<const CartesianExtend> geometry, const RectilinearMesh1d& mesh) :
        geometry(geometry), outBeta(NAN), outIntensity(this, &EffectiveIndex2dModule::getLightIntenisty) {
        inTemperature = 300.;
        setMesh(mesh);
    }

    /**
     * Constructor with 2D mesh provided
     *
     * \param geometry geometry in which the calculations are done
     * \param mesh mesh for dividing geometry
     */
    EffectiveIndex2dModule(shared_ptr<const CartesianExtend> geometry, const RectilinearMesh2d& mesh) :
        geometry(geometry), mesh(mesh), outBeta(NAN), outIntensity(this, &EffectiveIndex2dModule::getLightIntenisty) {
        inTemperature = 300.;
    }

    /**
     * Set the simple mesh based on the geometry bounding boxes.
     **/
    void setSimpleMesh() {
        mesh = RectilinearMesh2d(geometry->getChild());
    }


    /**
     * Set up the mesh. Horizontal division is provided while vertical one is created basing on the geometry bounding boxes.
     *
     * @param meshx Horizontal mesh
     **/
    void setMesh(const RectilinearMesh1d& meshx) {
        RectilinearMesh2d meshxy(geometry->getChild());
        mesh = RectilinearMesh2d(meshx, meshxy.c1);
    }


    /**
     * Set up the mesh. Both horizontal and vertical divisions are provided.
     *
     * @param meshxy The mesh
     **/
    void setMesh(const RectilinearMesh2d& meshxy) {
        mesh = meshxy;
    }


    virtual std::string getName() const { return "Optical: Effective Index Method 2D"; }

    virtual std::string getDescription() const {
        return "Calculate optical modes and optical field distribution using the effective index method "
               "in Cartesian two-dimentional space.";
    }

    /**
     * Find the mode around the specified propagation constant.
     *
     * This method remembers the determined mode, for rietrieval of the field profiles.
     *
     * @param beta initial propagation constant to search the mode around
     * @return determined propagation constant
     **/
    dcomplex computeMode(dcomplex beta);


    /**
     * Find the modes within the specified range
     *
     * This method \b does \b not remember the determined modes!
     *
     * @param beta1 one end of the range to browse
     * @param beta2 another end of the range to browse
     * @param steps number of steps for range browsing
     * @return vector of determined propagation constants
     **/
    std::vector<dcomplex> findModes(dcomplex beta1, dcomplex beta2, int steps=100);


    /**
     * Find approximate modes by scanning the desired range
     *
     * @param beta1 one end of the range to browse
     * @param beta2 another end of the range to browse
     * @param steps number of steps for range browsing
     * @return vector of determined potential propagation constants
     **/
    std::vector<dcomplex> findModesMap(dcomplex beta1, dcomplex beta2, int steps=100);


    /// Receiver of the wavelength
    ReceiverFor<Wavelength> inWavelength;

    /// Receiver for temperature
    ReceiverFor<Temperature, space::Cartesian2d> inTemperature;

    /// Provider for computed propagation constant
    ProviderFor<PropagationConstant>::WithValue outBeta;

    /// Provider of optical field
    ProviderFor<OpticalIntensity, space::Cartesian2d>::Delegate outIntensity;


  private:

    /// Method computing the distribution of light intensity
    shared_ptr<const std::vector<double>> getLightIntenisty(const Mesh<2>& dst_mesh, InterpolationMethod method=DEFAULT_INTERPOLATION);

};


}} // namespace plask::eim

#endif // PLASK__MODULE_OPTICAL_EIM_HPP
