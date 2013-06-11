#include "../python_globals.h"
#include "../python_property.h"

#include <plask/properties/thermal.h>

namespace plask { namespace python {

void register_standard_properties_heatflux()
{
    //TODO registerProperty<HeatFlux<2>>();
    //TODO registerProperty<HeatFlux<3>>();
    registerProvider<ProviderFor<HeatFlux, Geometry2DCartesian>>();
    registerProvider<ProviderFor<HeatFlux, Geometry2DCylindrical>>();
    registerProvider<ProviderFor<HeatFlux, Geometry3D>>();

}

}} // namespace plask>();
