#include "../python_globals.hpp"
#include "../python_property.hpp"

#include "plask/properties/energylevels.hpp"

namespace plask { namespace python {

void register_standard_properties_energy_levels()
{
//     py::class_<EnergyLevels>("EnergyLevels")
//         .def_readonly("electrons", &EnergyLevels::electrons)
//         .def_readonly("heavy_holes", &EnergyLevels::heavy_holes)
//         .def_readonly("light_holes", &EnergyLevels::light_holes)
//     ;
    registerProperty<EnergyLevels,false>();
    py_enum<EnergyLevels::EnumType>()
        .value("ELECTRONS", EnergyLevels::ELECTRONS)
        .value("HEAVY_HOLES", EnergyLevels::HEAVY_HOLES)
        .value("LIGHT_HOLES", EnergyLevels::LIGHT_HOLES)
    ;
}

}} // namespace plask::python
