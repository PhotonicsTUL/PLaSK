#include "python_globals.h"
#include "python_property.h"

#include <plask/properties/thermal.h>
#include <plask/properties/electrical.h>
#include <plask/properties/gain.h>
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
    registerProperty<Wavelength>();
    registerProperty<ModalLoss>();
    registerProperty<PropagationConstant>();
    registerProperty<EffectiveIndex>();
}

}} // namespace plask>();