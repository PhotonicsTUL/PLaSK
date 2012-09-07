#ifndef PLASK__HEATS_H
#define PLASK__HEATS_H

#include "providerfor.h"

namespace plask {

/**
 * Density of heat sources.
 */
struct Heats: public ScalarFieldProperty {
    static constexpr const char* NAME = "heat sources density"; // mind lower case here
};

}   // namespace plask

#endif // PLASK__HEATS_H
