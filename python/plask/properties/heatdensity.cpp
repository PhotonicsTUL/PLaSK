#include "../python_globals.hpp"
#include "../python_property.hpp"

#include "plask/properties/thermal.hpp"

namespace plask { namespace python {

void register_standard_properties_heatdensity()
{
    registerProperty<Heat>();
    RegisterCombinedProvider<HeatSumProvider<Geometry2DCartesian>>("HeatSumProvider2D");
    RegisterCombinedProvider<HeatSumProvider<Geometry2DCylindrical>>("HeatSumProviderCyl");
}

}} // namespace plask>();
