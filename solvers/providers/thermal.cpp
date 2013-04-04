#include "thermal.h"

#include <plask/filters/factory.h>

namespace plask {

FiltersFactory::RegisterStadnard<Temperature> registerTemperatureFilters;

}   // namespace plask
