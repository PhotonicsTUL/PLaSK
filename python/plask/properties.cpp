#include "python_globals.h"
#include "python_property.h"

namespace plask { namespace python {

// void register_standard_properties_thermal();
// void register_standard_properties_electrical();
void register_standard_properties_gain();
void register_standard_properties_GainOverCarriersConcentration();
void register_standard_properties_optical();
void register_standard_properties_refractive();


void register_standard_properties_thermal();
void register_standard_properties_temperature();
void register_standard_properties_heatdensity();
void register_standard_properties_heatflux();

void register_standard_properties_electrical();
void register_standard_properties_voltage();
void register_standard_properties_current();

void register_standard_properties_concentration_carriers();
void register_standard_properties_concentration_electrons();
void register_standard_properties_concentration_holes();

py::object filter_module;

/**
 * Register standard properties to Python.
 */
void register_standard_properties()
{
    filter_module = py::object(py::handle<>(py::borrowed(PyImport_AddModule("plask.filter"))));
    py::scope().attr("filter") = filter_module;
    filter_module.attr("__doc__") =
        "Data filters for standard properties.\n\n"

        "This module contains data filters for standard properties. Filters are\n"
        "solver-like classes that translate the fields computed in one geometry to\n"
        "another one. This geometry can have either the same or different dimension.\n\n"

        "All filter classes are used the same way. They are constructed with a single\n"
        "argument, which is a target geometry. The type of this geometry must match\n"
        "the suffix of the filter (``2D`` for two-dimensional Cartesian geometry, ``Cyl``\n"
        "for axi-symmetric cylindrical geometry, and ``3D`` for three-dimensional one.\n"
        "An example temperature filter for target 2D geometry can be constructed as\n"
        "follows:\n\n"

        ">>> temp_filter = filter.Temperature2D(mygeometry2d)\n\n"

        "Having an existing filter, you may attach a source provider to it, using bracket\n"
        "indexing. The `index` is a geometry object either existing in the target geometry\n"
        "or containing it (e.g. a :class:`geometry.Extrusion` object that is the root of\n"
        "the ``my_geometry_2d`` geometry). The `indexed` element is a proper data receiver\n"
        "that can be used for connecting the source data.\n\n"

        ">>> temp_filter[some_object_in_mygeometry2d]\n"
        "<_plask.ReceiverForTemperature2D at 0x43a5210>\n"
        ">> temp_filter[mygeometry2d.extrusion]\n"
        "<_plask.ReceiverForTemperature3D at 0x44751a0>\n\n"
        ">>> temp_filter[mygeometry2d.extrusion] = thermal_solver_3d.outTemperature\n\n"

        "After connecting the source, the tranlated data can be obtained using the filter\n"
        "member ``out``, which is a provider that can be connected to other solvers.\n\n"

        ">>> temp_filter.out\n"
        "<_plask.ProviderForTemperature2D at 0x43a5fa0>\n"
        ">>> other_solver_in_2d.inTemperature = temp_filter.out\n\n"

        "After the connection the filter does its job automatically.\n\n"

        "See also:\n"
        "    :ref:`sec-solvers-filters`.\n\n"

        "    Definition of filters in the XPL file: :xml:tag:`filter` tag.\n\n"

        "    Example using filters: :ref:`sec-tutorial-threshold-of-array`.\n"
    ;

    register_standard_properties_thermal();
    register_standard_properties_temperature();
    register_standard_properties_heatdensity();
    register_standard_properties_heatflux();

    register_standard_properties_electrical();
    register_standard_properties_voltage();
    register_standard_properties_current();
    register_standard_properties_concentration_carriers();
    register_standard_properties_concentration_electrons();
    register_standard_properties_concentration_holes();

    register_standard_properties_gain();
    register_standard_properties_GainOverCarriersConcentration();

    register_standard_properties_optical();
    register_standard_properties_refractive();
}

}} // namespace plask>();
