#ifndef PLASK__ELECTRICAL_H
#define PLASK__ELECTRICAL_H

#include <plask/provider/providerfor.h>

namespace plask {

/**
 * Electric potential/voltage [V]
 */
struct Potential: public ScalarFieldProperty {
    static constexpr const char* NAME = "potential";
};

/**
 * Electric current density [kA/cm^2]
 * This is 2D vector for two-dimensional sovers
 */
struct CurrentDensity: public VectorFieldProperty<> {
    static constexpr const char* NAME = "current density";
};

/**
 * Majority carriers concentration [1/cm^3]
 * It is not specified whether the carriers are electrons or holes
 */
struct CarriersConcentration: public ScalarFieldProperty {
    static constexpr const char* NAME = "carriers concentration";
};

/**
 * Electrons concentration [1/cm^3]
 */
struct ElectronsConcentration: public ScalarFieldProperty {
    static constexpr const char* NAME = "electrons concentration";
};

/**
 * Holes concentration [1/cm^3]
 */
struct HolesConcentration: public ScalarFieldProperty {
    static constexpr const char* NAME = "holes concentration";
};

} // namespace plask

#endif // PLASK__ELECTRICAL_H
