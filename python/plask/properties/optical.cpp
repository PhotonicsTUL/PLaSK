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
    registerProperty<LightMagnitude>();
    RegisterScaledProvider<ScaledFieldProvider<LightMagnitude,LightMagnitude,Geometry2DCartesian>>("ScaledLightMagnitude");
    //TODO RegisterCombinedProvider<LightMagnitudeSumProvider<Geometry2DCartesian>>("SumOfLightMagnitude");

    registerProperty<OpticalElectricField>();
    registerProperty<OpticalMagneticField>();

    registerProperty<Wavelength>();
    registerProperty<ModalLoss>();
    registerProperty<PropagationConstant>();
    registerProperty<EffectiveIndex>();
}

}} // namespace plask::python
