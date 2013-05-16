#include "../python_globals.h"
#include "../python_property.h"

#include <plask/properties/electrical.h>

namespace plask { namespace python {

void register_standard_properties_concentration()
{
    registerProperty<CarriersConcentration>();
    registerProperty<ElectronsConcentration>();
    registerProperty<HolesConcentration>();
}

}} // namespace plask>();