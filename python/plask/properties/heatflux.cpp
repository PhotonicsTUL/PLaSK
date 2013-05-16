#include "../python_globals.h"
#include "../python_property.h"

#include <plask/properties/thermal.h>

namespace plask { namespace python {

void register_standard_properties_heatflux()
{
    registerProperty<HeatFlux2D>();
    registerProperty<HeatFlux3D>();
}

}} // namespace plask>();