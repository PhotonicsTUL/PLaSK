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
    registerProperty<LightIntensity>();
    RegisterScaledProvider<ScaledFieldProvider<LightIntensity,LightIntensity,Geometry2DCartesian>>("ScaledLightIntensity");
    //TODO RegisterCombinedProvider<LightIntensitySumProvider<Geometry2DCartesian>>("SumOfLightIntensity");

    registerProperty<OpticalElectricField>();
    registerProperty<OpticalMagneticField>();

    registerProperty<Wavelength>();
    registerProperty<ModalLoss>();
    registerProperty<PropagationConstant>();
    registerProperty<EffectiveIndex>();
}

}} // namespace plask>();
