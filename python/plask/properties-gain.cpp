#include "python_globals.h"
#include "python_property.h"

#include <plask/properties/thermal.h>
#include <plask/properties/electrical.h>
#include <plask/properties/gain.h>
#include <plask/properties/optical.h>

namespace plask { namespace python {

/**
 * Register standard gain properties to Python.
 *
 * Add new gain properties here
 */
void register_standard_properties_gain()
{
    registerProperty<Gain>();
}

}} // namespace plask>();