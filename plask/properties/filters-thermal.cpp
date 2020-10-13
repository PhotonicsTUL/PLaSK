#include "thermal.hpp"

#include "plask/filters/factory.hpp"

namespace plask {

FiltersFactory::RegisterStandard<Temperature> registerTemperatureFilters;
//TODO FiltersFactory::RegisterStandard<HeatFlux<2>> registerHeatFlux<2>Filters;
//TODO FiltersFactory::RegisterStandard<HeatFlux<3>> registerHeatFlux<3>Filters;
FiltersFactory::RegisterStandard<Heat> registerHeatFilters;
FiltersFactory::RegisterStandard<ThermalConductivity> registerThermalConductivityFilters;

}   // namespace plask
