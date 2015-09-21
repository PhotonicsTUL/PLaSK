#ifndef PLASK__ELECTRICAL_H
#define PLASK__ELECTRICAL_H

#include <plask/provider/providerfor.h>

namespace plask {

/**
 * Electric potential/voltage [V]
 */
struct PLASK_API Potential: public ScalarFieldProperty {
    static constexpr const char* NAME = "potential";
    static constexpr const char* UNIT = "V";
};

/**
 * Electric current density [kA/cm²]
 * This is 2D vector for two-dimensional sovers
 */
struct PLASK_API CurrentDensity: public VectorFieldProperty<> {
    static constexpr const char* NAME = "current density";
    static constexpr const char* UNIT = "kA/cm²";
};

/**
 * Majority carriers concentration [1/cm^3]
 * It is not specified whether the carriers are electrons or holes
 */
struct PLASK_API CarriersConcentration: public ScalarFieldProperty {
    static constexpr const char* NAME = "carriers concentration";
    static constexpr const char* UNIT = "1/cm³";
};

/**
 * Electrons concentration [1/cm^3]
 */
struct PLASK_API ElectronsConcentration: public ScalarFieldProperty {
    static constexpr const char* NAME = "electrons concentration";
    static constexpr const char* UNIT = "1/cm³";
};

/**
 * Holes concentration [1/cm^3]
 */
struct PLASK_API HolesConcentration: public ScalarFieldProperty {
    static constexpr const char* NAME = "holes concentration";
    static constexpr const char* UNIT = "1/cm³";
};

/**
 * Electrical conductivity [S/m]
 */
struct PLASK_API Conductivity: FieldProperty<Tensor2<double>> {
    static constexpr const char* NAME = "electrical conductivity";
    static constexpr const char* UNIT = "S/m";
};

/**
 * Built-in potential [V]
 */
struct PLASK_API BuiltinPotential: public ScalarFieldProperty {
    static constexpr const char* NAME = "built-in potential";
    static constexpr const char* UNIT = "V";
};

/**
 * Quasi-Fermi energy level for electrons [eV]
 */
struct PLASK_API QuasiFermiElectronLevel: public ScalarFieldProperty {
    static constexpr const char* NAME = "quasi-Fermi energy level for electrons";
    static constexpr const char* UNIT = "eV";
};

/**
 * Quasi-Fermi energy level for holes [eV]
 */
struct PLASK_API QuasiFermiHoleLevel: public ScalarFieldProperty {
    static constexpr const char* NAME = "quasi-Fermi energy level for holes";
    static constexpr const char* UNIT = "eV";
};

} // namespace plask

#endif // PLASK__ELECTRICAL_H
