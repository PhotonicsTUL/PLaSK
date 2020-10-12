#include "../python_globals.hpp"
#include "../python_property.hpp"

#include "plask/properties/gain.hpp"

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
