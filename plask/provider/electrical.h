#ifndef PLASK__ELECTRICAL_H
#define PLASK__ELECTRICAL_H

#include "providerfor.h"

namespace plask {

struct Potential: public ScalarFieldProperty {
    static constexpr const char* NAME = "potential"; // mind lower case here
};

struct CurrentDensity2D: public VectorFieldProperty<2> {
    static constexpr const char* NAME = "current density 2D"; // mind lower case here
};

struct CurrentDensity3D: public VectorFieldProperty<3> {
    static constexpr const char* NAME = "current density 3D"; // mind lower case here
};

struct CarriersDensity: public ScalarFieldProperty {
    static constexpr const char* NAME = "carrier pairs density"; // mind lower case here
};

} // namespace plask

#endif // PLASK__ELECTRICAL_H
