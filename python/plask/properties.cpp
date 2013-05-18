#include "python_globals.h"
#include "python_property.h"

namespace plask { namespace python {

// void register_standard_properties_thermal();
// void register_standard_properties_electrical();
void register_standard_properties_gain();
void register_standard_properties_optical();


void register_standard_properties_temperature();
void register_standard_properties_heatflux();

void register_standard_properties_voltage();
void register_standard_properties_current();

void register_standard_properties_concentration_carriers();
void register_standard_properties_concentration_electrons();
void register_standard_properties_concentration_holes();


/**
 * Register standard properties to Python.
 */
void register_standard_properties()
{
    // register_standard_properties_thermal();
    register_standard_properties_temperature();
    register_standard_properties_heatflux();

    // register_standard_properties_electrical();
    register_standard_properties_voltage();
    register_standard_properties_current();
    register_standard_properties_concentration_carriers();
    register_standard_properties_concentration_electrons();
    register_standard_properties_concentration_holes();

    register_standard_properties_gain();

    register_standard_properties_optical();
}

}} // namespace plask>();
