#include "thermal.h"

#include <plask/filters/factory.h>

namespace plask {

FiltersFactory::RegisterStandard<Temperature> registerTemperatureFilters;

}   // namespace plask
