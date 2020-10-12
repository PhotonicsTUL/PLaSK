#include "../python_globals.hpp"
#include "../python_property.hpp"

#include "plask/properties/electrical.hpp"

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
