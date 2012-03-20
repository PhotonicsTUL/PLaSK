#ifndef PLASK__TEMPERATURE_H
#define PLASK__TEMPERATURE_H

#include "provider.h"

namespace plask {

/**
 * Physical property tag class for temperature:
 *
 * - plask::ProviderFor<Temperature> is an abstract base class for temperature providers, which has following implementation:
 *      - plask::ProviderFor<Temperature>::Delegate is a class for temperature provider which delegate calls to functor given in its constructor;
 *      - plask::ProviderFor<Temperature>::WithValue is a template (parametrized by mesh type) of class for temperature provider which holds vector of value (value member) and mesh (mesh member).
 * - plask::ReceiverFor<Temperature> is a class for temperature receivers.
 *
 * Example:
 * @code
 * // module calculating temperature in Cartesian 2d space:
 * struct PModule1 {
 *      plask::ProviderFor<Temperature, plask::Cartesian2d>::WithValue<SomeMeshType> outTemperature;
 *      // outTemperature.value stores temperature values in points pointed by outTemperature.mesh
 *      // outTemperature.value has type plask::shared_ptr< std::vector<double> >
 *      // outTemperature.mesh has type SomeMeshType (which should inharit from plask::Mest<plask::Cartesian2d>)
 *      // ...
 * };
 *
 * // another module which calculates temperature in Cartesian 2d space:
 * struct PModule2 {
 *      plask::ProviderFor<Temperature, plask::Cartesian2d>::Delegate outTemperature;
 *
 *      plask::shared_ptr< const std::vector<double> > getTemperature(const plask::Mesh<plask::Cartesian2d>& dst_mesh, plask::InterpolationMethod method) {
 *              // calculate and return temperature code
 *      }
 *      // ...
 *      PModule1(): outTemperature(this, &PModule2::getTemperature) {}
 * };
 *
 * // needs temperature in Cartesian 2d space:
 * struct RModule {
 *      plask::ReceiverFor<Temperature, plask::Cartesian2d> inTemperature;
 *      // ...
 * };
 *
 * //... in program:
 * PModule1 m1;
 * PModule2 m2;
 * RModule r;
 * r.inTemperature <<= m1.outTemperature;   //connect
 * r.inTemperature <<= m2.outTemperature;   //change data source of r from m1 to m2
 * @endcode
 *
 * @see @ref modules_write; @ref providers; plask::ProviderFor
 */
struct Temperature: public ScalarFieldProperty {};

//TODO in gcc 4.7 can be done by new typedefs:

/*
 * Provides temperature fields (temperature in points describe by given mesh).
 */
//typedef ProviderFor<Temperature> TemperatureProvider;

/*
 * Receive temperature fields (temperature in points describe by given mesh).
 */
//typedef ReceiverFor<Temperature> TemperatureReceiver;

} // namespace plask

#endif // PLASK__TEMPERATURE_H
