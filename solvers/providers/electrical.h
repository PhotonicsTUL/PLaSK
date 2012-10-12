#ifndef PLASK__ELECTRICAL_H
#define PLASK__ELECTRICAL_H

#include <plask/provider/providerfor.h>

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

struct CarrierConcentration: public ScalarFieldProperty {
    static constexpr const char* NAME = "carrier pairs concentration"; // mind lower case here
};

struct ElectronConcentration: public ScalarFieldProperty {
    static constexpr const char* NAME = "electron concentration"; // mind lower case here
};

struct HoleConcentration: public ScalarFieldProperty {
    static constexpr const char* NAME = "hole concentration"; // mind lower case here
};

} // namespace plask

#endif // PLASK__ELECTRICAL_H
