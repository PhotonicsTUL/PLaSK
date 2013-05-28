#ifndef PLASK__TEMPERATURE_H
#define PLASK__TEMPERATURE_H

#include <plask/provider/providerfor.h>
#include <plask/provider/combined_provider.h>

namespace plask {

/**
 * Physical property tag class for temperature:
 *
 * - plask::ProviderFor<Temperature> is an abstract base class for temperature providers, which has following implementations:
 *      - plask::ProviderFor<Temperature>::Delegate is a class for temperature provider which delegate calls to functor given in its constructor;
 *      - plask::ProviderFor<Temperature>::WithValue is a template (parametrized by mesh type) of class for temperature provider which holds vector of value (value member) and mesh (mesh member).
 * - plask::ReceiverFor<Temperature> is a class for temperature receivers.
 *
 * Example:
 * @code
 * // solver calculating temperature in Cartesian 2d space:
 * struct PSolver1: plask::SolverOver<plask::Geometry2DCartesian> {
 *      plask::ProviderFor<Temperature, plask::Cartesian2D>::WithValue<SomeMeshType> outTemperature;
 *      // outTemperature.value stores temperature values in points pointed by outTemperature.mesh
 *      // outTemperature.value has type plask::shared_ptr< std::vector<double> >
 *      // outTemperature.mesh has type SomeMeshType (which should inherit from plask::MeshD<plask::Cartesian2D>)
 *      // ...
 * };
 *
 * // another solver which calculates temperature in Cartesian 2d space:
 * struct PSolver2: plask::SolverWithMesh<plask::Geometry2DCartesian, plask::RectilinearMesh2D> {
 *      plask::ProviderFor<Temperature, plask::Cartesian2D>::Delegate outTemperature;
 *
 *      plask::DataVector<double> my_temperature;
 *
 *      plask::DataVector<double> getTemperature(const plask::MeshD<plask::Cartesian2D>& dst_mesh, plask::InterpolationMethod method) {
 *          return interpolate(*mesh, my_temperature, dst_mesh, method);
 *      }
 *      // ...
 *      PSolver1(): outTemperature(this, &PSolver2::getTemperature) {}
 * };
 *
 * // needs temperature in Cartesian 2d space:
 * struct RSolver {
 *      plask::ReceiverFor<Temperature, plask::Cartesian2D> inTemperature;
 *      // ...
 * };
 *
 * //... in program:
 * PSolver1 m1;
 * PSolver2 m2;
 * RSolver r;
 * r.inTemperature << m1.outTemperature;   // connect
 * r.inTemperature << m2.outTemperature;   // change data source of r from m1 to m2
 * @endcode
 *
 * @see @ref solvers_writing; @ref providers; plask::ProviderFor
 */

/**
 * Temperature [K].
 */
struct Temperature: public ScalarFieldProperty {
    static constexpr const char* NAME = "temperature";
    static inline double getDefaultValue() { return 300.; }
};

/**
 * Heat flux in 2D or 3D space [W/m] or [W/m²].
 */
template <int dim>
struct HeatFlux: public VectorFieldProperty<dim> {
    static constexpr const char* NAME = "heat flux";
};

/**
 * Density of heat sources.
 */
struct HeatDensity: public ScalarFieldProperty {
    static constexpr const char* NAME = "heat sources density";
};

/**
 * Provider which sum heat densities from one or more source.
 */
template <typename SpaceT>
struct HeatDensitySumProvider: public FieldSumProvider<HeatDensity,SpaceT> {};

} // namespace plask

#endif // PLASK__TEMPERATURE_H
