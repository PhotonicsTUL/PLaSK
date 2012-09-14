#ifndef PLASK__HEATS_H
#define PLASK__HEATS_H

#include "providerfor.h"
#include "combined_provider.h"

namespace plask {

/**
 * Density of heat sources.
 */
struct HeatDensity: public ScalarFieldProperty {
    static constexpr const char* NAME = "heat sources density"; // mind lower case here
};

/**
 * Provider which sum heat densities fro one or more source.
 */
template <typename SpaceT>
struct HeatDensitySumProvider: public SumOnMeshProviderWithInterpolation<ProviderFor<HeatDensity, SpaceT>, double, SpaceT> {
};

}   // namespace plask

#endif // PLASK__HEATS_H
