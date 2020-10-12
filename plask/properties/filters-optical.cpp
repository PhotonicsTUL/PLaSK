#include "optical.hpp"

#include "plask/filters/factory.hpp"

namespace plask {

// Optical
//TODO FiltersFactory::RegisterStandard<LightMagnitude> registerLightMagnitudeFilters;
//TODO FiltersFactory::RegisterStandard<LightMagnitude> registerModeLightMagnitudeFilters;
FiltersFactory::RegisterStandard<RefractiveIndex> registerRefractiveIndexFilters;

}   // namespace plask
