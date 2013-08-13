#include "thermal.h"
#include "electrical.h"
#include "gain.h"
#include "optical.h"

#include <plask/filters/factory.h>

namespace plask {

FiltersFactory::RegisterStandard<Gain> registerGainFilters;

}   // namespace plask
