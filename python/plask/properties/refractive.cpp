#include "../python_globals.h"
#include "../python_property.h"

#include <plask/properties/optical.h>

namespace plask { namespace python {

/**
 * Register standard optical properties to Python.
 *
 * Add new optical properties here
 */
void register_standard_properties_refractive()
{
    registerProperty<RefractiveIndex>();
}

}} // namespace plask::python
