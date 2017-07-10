#include "../python_globals.h"
#include "../python_property.h"

#include <plask/properties/gain.h>

namespace plask { namespace python {

/**
 * Register standard gain properties to Python.
 *
 * Add new gain properties here
 */
void register_standard_properties_gain()
{
    registerProperty<Gain>();
    py_enum<Gain::EnumType>()
        .value("", Gain::GAIN)
        .value("CONC", Gain::DGDN)
    ;
}

}} // namespace plask::python
