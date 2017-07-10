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
        .value("SPINOFF", BandEdges::SPIN_OFF)
        .value("SPIN_OFF", BandEdges::SPIN_OFF)
    ;
}

}} // namespace plask::python
