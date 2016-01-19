#include "electrical.h"

#include <plask/filters/factory.h>

namespace plask {

FiltersFactory::RegisterStandard<HoleConcentration> registerHoleConcentrationFilters;
FiltersFactory::RegisterStandard<Conductivity> registerConductivityFilters;

}   // namespace plask
