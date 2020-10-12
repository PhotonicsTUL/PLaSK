#include "../python_globals.hpp"
#include "../python_property.hpp"

#include "plask/properties/thermal.hpp"

namespace plask { namespace python {

void register_standard_properties_temperature()
{
    registerProperty<Temperature>();
}

}} // namespace plask::python
