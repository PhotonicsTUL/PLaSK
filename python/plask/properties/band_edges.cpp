#include "../python_globals.h"
#include "../python_property.h"

#include <plask/properties/electrical.h>

namespace plask { namespace python {

void register_standard_properties_band_edges()
{
    registerProperty<BandEdges>();
    py_enum<BandEdges::EnumType>()
        .value("CONDUCTION", BandEdges::CONDUCTION)
        .value("VALENCE", BandEdges::VALENCE)
    ;
}

}} // namespace plask::python
