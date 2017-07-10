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
 * Carriers concentration [1/cm^3]
 * For majority carriers it is not specified whether the carriers are electrons or holes
 */
struct PLASK_API CarriersConcentration: public MultiFieldProperty<double> {
    enum EnumType {
        MAJORITY = 0,
        PAIRS,
        ELECTRONS,
        HOLES
    };
    static constexpr size_t NUM_VALS = 4;
    static constexpr const char* NAME = "carriers concentration";
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
 * Quasi-Fermi levels for electrons and holes [eV]
 */
struct PLASK_API QuasiFermiLevels: public MultiFieldProperty<double> {
    enum EnumType {
        ELECTRONS,
        HOLES
    };
    static constexpr size_t NUM_VALS = 2;
    static constexpr const char* NAME = "quasi-Fermi levels for electrons and holes";
    static constexpr const char* UNIT = "eV";
};

/**
 * Conduction and valence band edges [eV]
 */
struct PLASK_API BandEdges: public MultiFieldProperty<double> {
    enum EnumType {
        CONDUCTION,
        VALENCE_HEAVY,
        VALENCE_LIGHT,
        SPIN_OFF
    };
    static constexpr size_t NUM_VALS = 2;
    static constexpr const char* NAME = "conduction and valence band edges";
    static constexpr const char* UNIT = "eV";
};

} // namespace plask

#endif // PLASK__ELECTRICAL_H
