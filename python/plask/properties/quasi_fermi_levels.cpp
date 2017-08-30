#include "../python_globals.h"
#include "../python_property.h"

#include <plask/properties/electrical.h>

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