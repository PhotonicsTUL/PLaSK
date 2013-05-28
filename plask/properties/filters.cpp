#include "thermal.h"
#include "electrical.h"
#include "gain.h"
#include "optical.h"

#include <plask/filters/factory.h>

namespace plask {

// Thermal
FiltersFactory::RegisterStandard<Temperature> registerTemperatureFilters;
//TODO FiltersFactory::RegisterStandard<HeatFlux<2>> registerHeatFlux<2>Filters;
//TODO FiltersFactory::RegisterStandard<HeatFlux<3>> registerHeatFlux<3>Filters;
FiltersFactory::RegisterStandard<HeatDensity> registerHeatDensityFilters;

// Electrical
FiltersFactory::RegisterStandard<Potential> registerPotentialFilters;
//TODO FiltersFactory::RegisterStandard<CurrentDensity<2>> registerCurrentDensity<2>Filters;
//TODO FiltersFactory::RegisterStandard<CurrentDensity<3>> registerCurrentDensity<3>Filters;
FiltersFactory::RegisterStandard<CarriersConcentration> registerCarriersConcentrationFilters;
FiltersFactory::RegisterStandard<ElectronsConcentration> registerElectronsConcentrationFilters;
FiltersFactory::RegisterStandard<HolesConcentration> registerHolesConcentrationFilters;

// Gain
FiltersFactory::RegisterStandard<Gain> registerGainFilters;

// Optical
FiltersFactory::RegisterStandard<OpticalIntensity> registerOpticalIntensityFilters;


}   // namespace plask
