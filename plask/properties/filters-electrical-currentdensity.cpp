#include "electrical.h"

#include <plask/filters/factory.h>

namespace plask {

FiltersFactory::RegisterStandard<CurrentDensity> registerCurrentDensityFilters;

}   // namespace plask
