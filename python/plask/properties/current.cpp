#include "../python_globals.h"
#include "../python_property.h"

#include <plask/properties/electrical.h>

namespace plask { namespace python {

void register_standard_properties_current()
{
    //TODO registerProperty<CurrentDensity<2>>();
    //TODO registerProperty<CurrentDensity<3>>();
    //TODO
    registerProvider<ProviderFor<CurrentDensity, Geometry2DCartesian>>();
    registerProvider<ProviderFor<CurrentDensity, Geometry2DCylindrical>>();
    registerProvider<ProviderFor<CurrentDensity, Geometry3D>>();
}

}} // namespace plask>();
