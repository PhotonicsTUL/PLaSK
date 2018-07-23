#include "optical.h"

#include <plask/filters/factory.h>

namespace plask {

// Optical
//TODO FiltersFactory::RegisterStandard<LightMagnitude> registerLightMagnitudeFilters;
//TODO FiltersFactory::RegisterStandard<LightMagnitude> registerModeLightMagnitudeFilters;
FiltersFactory::RegisterStandard<RefractiveIndex> registerRefractiveIndexFilters;

}   // namespace plask
