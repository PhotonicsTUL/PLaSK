#include "electrical.h"

#include <plask/filters/factory.h>

namespace plask {

FiltersFactory::RegisterStandard<BuiltinPotential> registerEnergyFilters;

}   // namespace plask
