#include "../python_globals.h"
#include "../python_property.h"

#include <plask/properties/electrical.h>

namespace plask { namespace python {

void register_standard_properties_current()
{
    //TODO registerProperty<CurrentDensity<2>>();
    //TODO registerProperty<CurrentDensity<3>>();
    registerProvider<ProviderFor<CurrentDensity<2>,Geometry2DCartesian>>();
    registerProvider<ProviderFor<CurrentDensity<2>,Geometry2DCylindrical>>();
    registerProvider<ProviderFor<CurrentDensity<3>,Geometry3D>>();
    registerReceiver<ReceiverFor<CurrentDensity<2>,Geometry2DCartesian>>();
    registerReceiver<ReceiverFor<CurrentDensity<2>,Geometry2DCylindrical>>();
    registerReceiver<ReceiverFor<CurrentDensity<3>,Geometry3D>>();
}

}} // namespace plask>();
