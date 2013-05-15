#include "python_globals.h"
#include "python_property.h"

#include <plask/properties/thermal.h>
#include <plask/properties/electrical.h>
#include <plask/properties/gain.h>
#include <plask/properties/optical.h>

namespace plask { namespace python {

/**
 * Register standard electrical properties to Python.
 *
 * Add new electrical properties here
 */
void register_standard_properties_electrical()
{
    registerProperty<Potential>();
    registerProperty<CurrentDensity2D>();
    registerProperty<CurrentDensity3D>();
    registerProperty<CarriersConcentration>();
    registerProperty<ElectronsConcentration>();
    registerProperty<HolesConcentration>();
}

}} // namespace plask>();