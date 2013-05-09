#include "thermal.h"
#include "electrical.h"
#include "gain.h"
#include "optical.h"

#include <plask/filters/factory.h>

namespace plask {

// Thermal
FiltersFactory::RegisterStandard<Temperature> registerTemperatureFilters;
FiltersFactory::RegisterStandard<HeatFlux2D> registerHeatFlux2DFilters;
FiltersFactory::RegisterStandard<HeatFlux3D> registerHeatFlux3DFilters;
FiltersFactory::RegisterStandard<HeatDensity> registerHeatDensityFilters;

// Electrical
FiltersFactory::RegisterStandard<Potential> registerPotentialFilters;
FiltersFactory::RegisterStandard<CurrentDensity2D> registerCurrentDensity2DFilters;
FiltersFactory::RegisterStandard<CurrentDensity3D> registerCurrentDensity3DFilters;
FiltersFactory::RegisterStandard<CarriersConcentration> registerCarriersConcentrationFilters;
FiltersFactory::RegisterStandard<ElectronsConcentration> registerElectronsConcentrationFilters;
FiltersFactory::RegisterStandard<HolesConcentration> registerHolesConcentrationFilters;

// Gain
FiltersFactory::RegisterStandard<Gain> registerGainFilters;

// Optical
FiltersFactory::RegisterStandard<OpticalIntensity> registerOpticalIntensityFilters;


}   // namespace plask
