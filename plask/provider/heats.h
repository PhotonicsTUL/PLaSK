#ifndef PLASK__HEATS_H
#define PLASK__HEATS_H

#include "providerfor.h"

namespace plask {

/**
 * Heats
 */
struct Heats: public ScalarFieldProperty {
    static constexpr const char* NAME = "heats"; // mind lower case here
};

}   // namespace plask

#endif // PLASK__HEATS_H
