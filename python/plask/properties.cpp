#include "python_globals.h"
#include "python_property.h"

#include <plask/properties/thermal.h>
#include <plask/properties/electrical.h>
#include <plask/properties/gain.h>
#include <plask/properties/optical.h>

namespace plask { namespace python {

void register_standard_properties_thermal();
void register_standard_properties_electrical();
void register_standard_properties_gain();
void register_standard_properties_optical();

/**
 * Register standard properties to Python.
 */
void register_standard_properties()
{
    register_standard_properties_thermal();
    register_standard_properties_electrical();
    register_standard_properties_gain();
    register_standard_properties_optical();
}

}} // namespace plask>();