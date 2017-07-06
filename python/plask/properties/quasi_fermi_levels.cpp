#include "../python_globals.h"
#include "../python_property.h"

#include <plask/properties/electrical.h>

namespace plask { namespace python {

void register_standard_properties_quasi_Fermi_levels()
{
    registerProperty<QuasiFermiLevels>();
    py_enum<QuasiFermiLevels::EnumType>()
        .value("ELECTRONS", QuasiFermiLevels::ELECTRONS)
        .value("HOLES", QuasiFermiLevels::HOLES)
    ;
}

}} // namespace plask::python
