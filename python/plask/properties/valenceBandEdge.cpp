#include "../python_globals.h"
#include "../python_property.h"

#include <plask/properties/electrical.h>

namespace plask { namespace python {

void register_standard_properties_valence_band_edge()
{
    registerProperty<ValenceBandEdge>();
}

}} // namespace plask::python
