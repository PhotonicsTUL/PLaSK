#ifndef PLASK__MODULE_OPTICAL_EIM_HPP
#define PLASK__MODULE_OPTICAL_EIM_HPP

#include <plask/plask.hpp>

namespace plask { namespace eim {

class RootDigger;

/**
 * Module performing calculations in 2D Cartesian space using effective index method
 */
class EffectiveIndex2dModule: public Module {

    friend class RootDigger;

    /// Geometry in which the calculations are performed
    shared_ptr<const CartesianExtend> geometry;

    /// The mesh used for cutting the structure into one-dimentional stripes
    shared_ptr<RectilinearMesh2d> mesh;

    Data2dLog<dcomplex,double> log_value;

  public:

    /**
     * Default constructor creates default mesh based on geometry
     *
     * \param geometry geometry in which the calculations are done
     */
    EffectiveIndex2dModule(shared_ptr<const CartesianExtend> geometry) :
        geometry(geometry), outBeta(NAN), outIntensity(this, &EffectiveIndex2dModule::getLightIntenisty),
        log_value(dataLog<dcomplex, double>("beta", "char_val")) {
        inTemperature = 300.;
        auto child = geometry->getChild();
        if (!child) throw NoChildException();
        mesh = make_shared<RectilinearMesh2d>(child);
    }

    /**
     * Constructor with 1D mesh provided
     *
     * \param geometry geometry in which the calculations are done
     * \param mesh horizontal mesh for dividing geometry
     */
    EffectiveIndex2dModule(shared_ptr<const CartesianExtend> geometry, const RectilinearMesh1d& meshx) :
        geometry(geometry), outBeta(NAN), outIntensity(this, &EffectiveIndex2dModule::getLightIntenisty),
        log_value(dataLog<dcomplex, double>("beta", "char_val")) {
        inTemperature = 300.;
        auto child = geometry->getChild();
        if (!child) throw NoChildException();
        RectilinearMesh2d meshxy(child);
        mesh = make_shared<RectilinearMesh2d>(meshx, meshxy.c1);
    }

    /**
     * Constructor with 2D mesh provided
     *
     * \param geometry geometry in which the calculations are done
     * \param mesh mesh for dividing geometry
     */
    EffectiveIndex2dModule(shared_ptr<const CartesianExtend> geometry, shared_ptr<RectilinearMesh2d> mesh) :
        geometry(geometry), mesh(mesh), outBeta(NAN), outIntensity(this, &EffectiveIndex2dModule::getLightIntenisty),
        log_value(dataLog<dcomplex, double>("beta", "char_val")) {
        inTemperature = 300.;
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

    /// Return function value for root digger
    dcomplex char_val(dcomplex x) { return 0.; /* TODO */ }

    /// Method computing the distribution of light intensity
    shared_ptr<const std::vector<double>> getLightIntenisty(const Mesh<2>& dst_mesh, InterpolationMethod method=DEFAULT_INTERPOLATION);

};


}} // namespace plask::eim

#endif // PLASK__MODULE_OPTICAL_EIM_HPP
