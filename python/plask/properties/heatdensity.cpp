#include "../python_globals.h"
#include "../python_property.h"

#include <plask/properties/thermal.h>

namespace plask { namespace python {

void register_standard_properties_heatdensity()
{
    registerProperty<HeatDensity>();
    RegisterCombinedProvider<HeatDensitySumProvider<Geometry2DCartesian>>("SumOfHeatDensity");
}

}} // namespace plask>();
