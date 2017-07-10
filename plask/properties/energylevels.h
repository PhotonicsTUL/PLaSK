#ifndef PLASK__ENERGYLEVELS_H
#define PLASK__ENERGYLEVELS_H

#include <vector>

#include <plask/provider/providerfor.h>

namespace plask {

struct EnergyLevels;

/**
 * Energy levels for electrons and holes [eV]
 */
struct PLASK_API EnergyLevels: public MultiValueProperty<EnergyLevels> {
    static constexpr const char* NAME = "energy levels for electrons and holes";
    static constexpr const char* UNIT = "eV";

    std::vector<double> electrons;
    std::vector<double> heavy_holes;
    std::vector<double> light_holes;
};

} // namespace plask

#endif // PLASK__ENERGYLEVELS_H
