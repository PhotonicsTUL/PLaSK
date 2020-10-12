#include "../python_globals.hpp"
#include "../python_property.hpp"

#include "plask/properties/optical.hpp"

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
