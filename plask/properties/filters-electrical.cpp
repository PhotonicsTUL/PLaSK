#include "electrical.hpp"

#include "plask/filters/factory.hpp"

namespace plask {

//FiltersFactory::RegisterStandard<HoleConcentration> registerHoleConcentrationFilters;
FiltersFactory::RegisterStandard<Conductivity> registerConductivityFilters;

}   // namespace plask
