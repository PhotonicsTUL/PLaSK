#include "electrical.h"

#include <plask/filters/factory.h>

namespace plask {

FiltersFactory::RegisterStandard<HolesConcentration> registerHolesConcentrationFilters;
FiltersFactory::RegisterStandard<ElectricalConductivity> registerElectricalConductivityFilters;

}   // namespace plask
