#ifndef PLASK__ENERGYLEVELS_H
#define PLASK__ENERGYLEVELS_H

#include <vector>

#include <plask/provider/providerfor.h>

namespace plask {

/**
 * Energy levels for electrons and holes [eV]
 */
struct PLASK_API EnergyLevels: public MultiFieldProperty<std::vector<double>> {
    enum EnumType {
        ELECTRONS,
        HEAVY_HOLES,
        LIGHT_HOLES
    };
    static constexpr size_t NUM_VALS = 3;
    static constexpr const char* NAME = "energy levels for electrons and holes";
    static constexpr const char* UNIT = "eV";
};

} // namespace plask

#endif // PLASK__ENERGYLEVELS_H
