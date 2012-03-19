#ifndef PLASK__MODULE_OPTICAL_EIM_HPP
#define PLASK__MODULE_OPTICAL_EIM_HPP

#include <plask/plask.hpp>

namespace plask { namespace eim {


/**
 * Module performing calculations in 2D Cartesian space using effective index method
 */
struct EffectiveIndex2dModule: public Module {

    EffectiveIndex2dModule(const RectilinearMesh1d meshx);

    EffectiveIndex2dModule(const RectilinearMesh2d meshxy);

    /// Provider of optical field
    ProviderFor<OpticalIntensity, space::Cartesian2d>::Delegate outIntensity;

    /// Receiver for temperature
    plask::ReceiverFor<Temperature, space::Cartesian2d> inTemperature;

    /// Method computing the distribution of light intensity
    shared_ptr<const std::vector<double>> getLightIntenisty(const Mesh<space::Cartesian2d>& dst_mesh, InterpolationMethod method=DEFAULT_INTERPOLATION);

};


}} // namespace plask::eim

#endif // PLASK__MODULE_OPTICAL_EIM_HPP