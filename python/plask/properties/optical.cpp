#include "../python_globals.hpp"
#include "../python_property.hpp"

#include "plask/properties/optical.hpp"

namespace plask { namespace python {

/**
 * Register standard optical properties to Python.
 *
 * Add new optical properties here
 */
void register_standard_properties_optical()
{
    registerProperty<LightMagnitude>();
    registerProperty<LightE>();
    registerProperty<LightH>();

    registerProperty<ModeLightMagnitude>();
    //TODO RegisterCombinedProvider<LightMagnitudeSumProvider<Geometry2DCartesian>>("SumOfLightMagnitude");
    registerProperty<ModeLightE>();
    registerProperty<ModeLightH>();

    registerProperty<ModeWavelength>();
    registerProperty<ModeLoss>();
    registerProperty<ModePropagationConstant>();
    registerProperty<ModeEffectiveIndex>();
}

}} // namespace plask::python
