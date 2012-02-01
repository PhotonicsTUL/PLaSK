#ifndef PLASK__TEMPERATURE_H
#define PLASK__TEMPERATURE_H

#include "provider.h"

namespace plask {

/**
 * Physical property tag class for temperature:
 * - plask::ProviderFor<Temperature> is abstract, base class for temperature providers, which have following implementation:
 *      - plask::ProviderFor<Temperature>::Delegate is class for temperature provider which delegate calls to functor given in its constructor;
 *      - plask::ProviderFor<Temperature>::WithValue is a template (parametrized by mesh type) of class for temperature provider which hold vector of value (value member) and mesh (mesh member).
 * - plask::ReceiverFor<Temperature> is class for temperature receivers.
 *
 * Example:
 * @code
 * //calculate temperature in Cartesian 2d space:
 * struct Module1 {
 *      plask::ProviderFor<Temperature, plask::Cartesian2d>::Delegate outTemperature;
 *
 *      plask::shared_ptr< const std::vector<double> > getTemperature(const plask::Mesh<plask::Cartesian2d>& dst_mesh, plask::InterpolationMethod method) {
 *              //calculate and return temperature code
 *      }
 *      //...
 *      Module1(): outTemperature(&Module1::getTemperature) {}
 * };
 *
 * //needs temperature in Cartesian 2d space:
 * struct Module2 {
 *      plask::ReceiverFor<Temperature, plask::Cartesian2d> inTemperature;
 *      //...
 * };
 *
 * //... in program:
 * Module1 m1;
 * Module1 m2;
 * m1.inTemperature = m2.outTemperature;
 * @endcode
 *
 * @see @ref modules_write; @ref providers; plask::ProviderFor
 */
struct Temperature: ScalarFieldProperty {};

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
