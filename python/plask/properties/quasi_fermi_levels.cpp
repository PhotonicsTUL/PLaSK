#include "../python_globals.hpp"
#include "../python_property.hpp"

#include "plask/properties/electrical.hpp"

namespace plask { namespace python {

void register_standard_properties_quasi_Fermi_levels()
{
    registerProperty<FermiLevels>();
    py_enum<FermiLevels::EnumType>()
        .value("ELECTRONS", FermiLevels::ELECTRONS)
        .value("HOLES", FermiLevels::HOLES)
    ;
}

}} // namespace plask::python
