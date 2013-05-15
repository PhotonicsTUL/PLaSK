#include "python_globals.h"
#include "python_property.h"

#include <plask/properties/thermal.h>
#include <plask/properties/electrical.h>
#include <plask/properties/gain.h>
#include <plask/properties/optical.h>

namespace plask { namespace python {

/**
 * Register standard thermal properties to Python.
 *
 * Add new thermal properties here
 */
void register_standard_properties_thermal()
{
    registerProperty<Temperature>();
    registerProperty<HeatFlux2D>();
    registerProperty<HeatFlux3D>();
    registerProperty<HeatDensity>();
}

}} // namespace plask>();