#include "thermal.h"
#include "electrical.h"
#include "gain.h"
#include "optical.h"

#include <plask/filters/factory.h>

namespace plask {

FiltersFactory::RegisterStandard<Potential> registerPotentialFilters;
FiltersFactory::RegisterStandard<CurrentDensity> registerCurrentDensityFilters;
FiltersFactory::RegisterStandard<CarriersConcentration> registerCarriersConcentrationFilters;
FiltersFactory::RegisterStandard<ElectronsConcentration> registerElectronsConcentrationFilters;
FiltersFactory::RegisterStandard<HolesConcentration> registerHolesConcentrationFilters;
FiltersFactory::RegisterStandard<ElectricalConductivity> registerElectricalConductivityFilters;

}   // namespace plask
