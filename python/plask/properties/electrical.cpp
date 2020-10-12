#include "../python_globals.hpp"
#include "../python_property.hpp"

#include "plask/properties/electrical.hpp"

namespace plask { namespace python {

void register_standard_properties_electrical()
{
    registerProperty<Conductivity>();
}

}} // namespace plask::python
