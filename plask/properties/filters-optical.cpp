#include "optical.h"

#include <plask/filters/factory.h>

namespace plask {

// Optical
//TODO FiltersFactory::RegisterStandard<LightMagnitude> registerLightMagnitudeFilters;
FiltersFactory::RegisterStandard<RefractiveIndex> registerRefractiveIndexFilters;

}   // namespace plask
