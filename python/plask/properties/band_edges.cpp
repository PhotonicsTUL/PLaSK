#include "../python_globals.hpp"
#include "../python_property.hpp"

#include "plask/properties/electrical.hpp"

namespace plask { namespace python {

void register_standard_properties_band_edges()
{
    registerProperty<BandEdges>();
    py_enum<BandEdges::EnumType>()
        .value("CONDUCTION", BandEdges::CONDUCTION)
        .value("VALENCE_HEAVY", BandEdges::VALENCE_HEAVY)
        .value("VALENCE_LIGHT", BandEdges::VALENCE_LIGHT)
        .value("SPINOFF", BandEdges::SPIN_OFF)
        .value("SPIN_OFF", BandEdges::SPIN_OFF)
    ;
}

}} // namespace plask::python
