#include "../python_globals.h"
#include "../python_property.h"

#include <plask/properties/electrical.h>

namespace plask { namespace python {

void register_standard_properties_quasi_Fermi_hole_level()
{
    registerProperty<QuasiFermiEnergyLevelForHoles>();
}

}} // namespace plask::python
