#ifndef PLASK__ELECTRICAL_H
#define PLASK__ELECTRICAL_H

#include <plask/provider/providerfor.h>

namespace plask {

/**
 * Electric potential [V]
 */
struct PLASK_API Potential: public ScalarFieldProperty {
    static constexpr const char* NAME = "potential";
    static constexpr const char* UNIT = "V";
};

/**
 * Electric voltage [V]
 */
struct PLASK_API Voltage: public ScalarFieldProperty {
    static constexpr const char* NAME = "voltage";
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
struct PLASK_API ElectronConcentration: public ScalarFieldProperty {
    static constexpr const char* NAME = "electron concentration";
    static constexpr const char* UNIT = "1/cm³";
};

/**
 * Holes concentration [1/cm^3]
 */
struct PLASK_API HoleConcentration: public ScalarFieldProperty {
    static constexpr const char* NAME = "hole concentration";
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
 * Quasi-Fermi energy level for electrons [eV]
 */
struct PLASK_API QuasiFermiEnergyLevelForElectrons: public ScalarFieldProperty {
    static constexpr const char* NAME = "quasi-Fermi energy level for electrons";
    static constexpr const char* UNIT = "eV";
};

/**
 * Quasi-Fermi energy level for holes [eV]
 */
struct PLASK_API QuasiFermiEnergyLevelForHoles: public ScalarFieldProperty {
    static constexpr const char* NAME = "quasi-Fermi energy level for holes";
    static constexpr const char* UNIT = "eV";
};

/**
 * Conduction band edge [eV]
 */
struct PLASK_API ConductionBandEdge: public ScalarFieldProperty {
    static constexpr const char* NAME = "conduction band edge";
    static constexpr const char* UNIT = "eV";
};

/**
 * Valence band edge [eV]
 */
struct PLASK_API ValenceBandEdge: public ScalarFieldProperty {
    static constexpr const char* NAME = "valence band edge";
    static constexpr const char* UNIT = "eV";
};

} // namespace plask

#endif // PLASK__ELECTRICAL_H
