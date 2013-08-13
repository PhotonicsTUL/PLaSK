#include "thermal.h"
#include "electrical.h"
#include "gain.h"
#include "optical.h"

#include <plask/filters/factory.h>

namespace plask {

FiltersFactory::RegisterStandard<Potential> registerPotentialFilters;
//TODO FiltersFactory::RegisterStandard<CurrentDensity<2>> registerCurrentDensity<2>Filters;
//TODO FiltersFactory::RegisterStandard<CurrentDensity<3>> registerCurrentDensity<3>Filters;
FiltersFactory::RegisterStandard<CarriersConcentration> registerCarriersConcentrationFilters;
FiltersFactory::RegisterStandard<ElectronsConcentration> registerElectronsConcentrationFilters;
FiltersFactory::RegisterStandard<HolesConcentration> registerHolesConcentrationFilters;

}   // namespace plask
