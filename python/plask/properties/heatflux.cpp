#include "../python_globals.h"
#include "../python_property.h"

#include <plask/properties/thermal.h>

namespace plask { namespace python {

void register_standard_properties_heatflux()
{
    //TODO registerProperty<HeatFlux<2>>();
    //TODO registerProperty<HeatFlux<3>>();
    //TODO registerProvider<ProviderFor<HeatFlux<2>,Geometry2DCartesian>>();
    registerProvider<ProviderFor<HeatFlux<2>,Geometry2DCylindrical>>();
    registerProvider<ProviderFor<HeatFlux<3>,Geometry3D>>();

}

}} // namespace plask>();