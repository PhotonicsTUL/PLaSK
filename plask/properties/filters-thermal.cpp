#include "thermal.h"
#include "electrical.h"
#include "gain.h"
#include "optical.h"

#include <plask/filters/factory.h>

namespace plask {

FiltersFactory::RegisterStandard<Temperature> registerTemperatureFilters;
//TODO FiltersFactory::RegisterStandard<HeatFlux<2>> registerHeatFlux<2>Filters;
//TODO FiltersFactory::RegisterStandard<HeatFlux<3>> registerHeatFlux<3>Filters;
FiltersFactory::RegisterStandard<Heat> registerHeatFilters;
FiltersFactory::RegisterStandard<ThermalConductivity> registerThermalConductivityFilters;

}   // namespace plask
