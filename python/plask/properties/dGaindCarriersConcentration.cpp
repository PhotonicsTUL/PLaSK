#include "../python_globals.h"
#include "../python_property.h"

#include <plask/properties/gain.h>

namespace plask { namespace python {

/**
 * Register standard gain properties to Python.
 *
 * Add new gain properties here
 */
void register_standard_properties_GainOverCarriersConcentration()
{
    registerProperty<GainOverCarriersConcentration>();
}

}} // namespace plask>();
