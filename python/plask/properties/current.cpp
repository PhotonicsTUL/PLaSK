#include "../python_globals.h"
#include "../python_property.h"

#include <plask/properties/electrical.h>

namespace plask { namespace python {

void register_standard_properties_current()
{
    registerProperty<Potential>();
    registerProperty<CurrentDensity2D>();
    registerProperty<CurrentDensity3D>();
}

}} // namespace plask>();