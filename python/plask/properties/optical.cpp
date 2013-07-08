#include "../python_globals.h"
#include "../python_property.h"

#include <plask/properties/optical.h>

namespace plask { namespace python {

/**
 * Register standard optical properties to Python.
 *
 * Add new optical properties here
 */
void register_standard_properties_optical()
{
    registerProperty<OpticalIntensity>();
    registerProperty<LightIntensity>();
    RegisterScaledProvider<ScaledFieldProvider<LightIntensity,OpticalIntensity,Geometry2DCartesian>>("LightIntensityAutoscaled");
    RegisterCombinedProvider<LightIntensitySumProvider<Geometry2DCartesian>>("SumOfLightIntensity");

    registerProperty<Wavelength>();
    registerProperty<ModalLoss>();
    registerProperty<PropagationConstant>();
    registerProperty<EffectiveIndex>();
}

}} // namespace plask>();