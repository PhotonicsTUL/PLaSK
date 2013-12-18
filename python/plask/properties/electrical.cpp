#include "../python_globals.h"
#include "../python_property.h"

#include <plask/properties/electrical.h>

namespace plask { namespace python {

void register_standard_properties_electrical()
{
    registerProperty<ElectricalConductivity>();
}

}} // namespace plask::python
