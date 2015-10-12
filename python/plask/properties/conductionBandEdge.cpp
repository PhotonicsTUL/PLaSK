#include "../python_globals.h"
#include "../python_property.h"

#include <plask/properties/electrical.h>

namespace plask { namespace python {

void register_standard_properties_conduction_band_edge()
{
    registerProperty<ConductionBandEdge>();
}

}} // namespace plask::python
