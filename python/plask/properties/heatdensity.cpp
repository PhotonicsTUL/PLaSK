#include "../python_globals.h"
#include "../python_property.h"

#include <plask/properties/thermal.h>

namespace plask { namespace python {

void register_standard_properties_heatdensity()
{
    registerProperty<Heat>();
    RegisterCombinedProvider<HeatSumProvider<Geometry2DCartesian>>("SumOfHeat");
}

}} // namespace plask>();
