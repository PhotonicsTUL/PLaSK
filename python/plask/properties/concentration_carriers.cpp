#include "../python_globals.h"
#include "../python_property.h"

#include <plask/properties/electrical.h>

namespace plask { namespace python {

void register_standard_properties_concentration_carriers()
{
    registerProperty<CarriersConcentration>();
    py_enum<CarriersConcentration::EnumType>()
        .value("MAJORITY", CarriersConcentration::MAJORITY)
        .value("PAIRS", CarriersConcentration::PAIRS)
        .value("ELECTRONS", CarriersConcentration::ELECTRONS)
        .value("HOLES", CarriersConcentration::HOLES)
    ;
}

}} // namespace plask::python
