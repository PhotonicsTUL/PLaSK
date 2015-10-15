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
void register_standard_properties_conduction_band_edge();
void register_standard_properties_valence_band_edge();

void register_standard_properties_builtin_potential();
void register_standard_properties_quasi_Fermi_electron_level();
void register_standard_properties_quasi_Fermi_hole_level();

const char* docstring_receiver =
    "%1%Receiver%2%()\n\n"

    "Receiver of the %3%%4% [%5%].\n\n"

    "You may connect a provider to this receiver usign either the `connect` method\n"
    "or an assignement operator. Then, you can read the provided value by calling\n"
    "this receiver with arguments identical as the ones of the corresponding\n"
    "provider :class:`~plask.flow.%1%Provider%2%`.\n\n"

    "Example:\n"
    "   Connect the reveiver to a provider from some other solver:\n\n"

    "   >>> solver.in%1% = other_solver.out%1%\n\n"

    "See also:\n"
    "   Provider of %3%: :class:`plask.flow.%1%Provider%2%`\n\n"
    "   Data filter for %3%: :class:`plask.flow.%1%Filter%2%`";

const char* docstring_receiver_connect =
    "Connect some provider to the receiver.\n\n"

    "Example:\n"
    "   >>> solver.in%1%.connect(other_solver.out%1%)\n\n"

    "Note:\n"
    "   You may achieve the same effect by using the asignmnent operator\n"
    "   if you put an exisiting provider at the right side of this operator:\n\n"

    "   >>> solver.in%1% = other_solver.out%1%\n";

const char* docstring_receiver_assign =
    "Assign constant value to the receiver.\n\n"

    "The receiver will always serve this value to the solver regardless of the\n"
    "spatial coordinates. Use for manually setting uniform fields (e.g. constant\n"
    "temperature.\n\n"

    "Example:\n"
    "   >>> solver.in%1%.assign(300.)\n"
    "   >>> solver.in%1%(any_mesh)[0]\n"
    "   300.\n"
    "   >>> solver.in%1%(any_mesh)[-1]\n"
    "   300.\n\n"

    "Note:\n"
    "   You may achieve the same effect by using the asignmnent operator\n"
    "   if you put a value at the right side of this operator:\n\n"

    "   >>> solver.in%1% = 300.";

PLASK_PYTHON_API template <PropertyType propertyType> const char* docstring_provider();

PLASK_PYTHON_API template <> const char* docstring_provider<SINGLE_VALUE_PROPERTY>() { return
    "%1%Provider%2%(data)\n\n"

    "Provider of the %3%%4% [%7%].\n\n"

    "This class is used for %3% provider in binary solvers. You can also create\n"
    "a custom provider for your Python solver.\n\n"

    "Args:\n"
    "   data: provided value or callable returning it on request.\n"
    "       The callable must accept the same arguments as the provider\n"
    "       ``__call__`` method (see below).\n\n"

    "To obtain the value from the provider simply call it. The call signature\n"
    "is as follows:\n\n"

    ".. method:: solver.out%1%(%5%)\n\n"

    "   %6%\n"

    "   :return: Value of the %3% **[%7%]**.\n\n"

    "Example:\n"
    "   Connect the provider to a receiver in some other solver:\n\n"

    "   >>> other_solver.in%1% = solver.out%1%\n\n"

    "   Obtain the provided value:\n\n"

    "   >>> solver.out%1%(%5%)\n"
    "   1000\n\n"

    "See also:\n"
    "   Receiver of %3%: :class:`plask.flow.%1%Receiver%2%`\n";
}

PLASK_PYTHON_API template <> const char* docstring_provider<MULTI_VALUE_PROPERTY>() { return
    "%1%Provider%2%(data)\n\n"

    "Provider of the %3%%4% [%7%].\n\n"

    "This class is used for %3% provider in binary solvers. You can also create\n"
    "a custom provider for your Python solver.\n\n"

    "Args:\n"
    "   data: provided value or callable returning it on request.\n"
    "       The callable must accept the same arguments as the provider\n"
    "       ``__call__`` method (see below). It must also be able to give its\n"
    "       length (i.e. have the ``__len__`` method defined) that gives the\n"
    "       number of different provided values.\n\n"

    "To obtain the value from the provider simply call it. The call signature\n"
    "is as follows:\n\n"

    ".. method:: solver.out%1%(n=0%5%)\n\n"

    "   :param int n: Value number.\n"
    "   %6%\n"

    "   :return: Value of the %3% **[%7%]**.\n\n"

    "You may obtain the number of different values this provider can return by\n"
    "testing its length.\n\n"

    "Example:\n"
    "   Connect the provider to a receiver in some other solver:\n\n"

    "   >>> other_solver.in%1% = solver.out%1%\n\n"

    "   Obtain the provided value:\n\n"

    "   >>> solver.out%1%(0%5%)\n"
    "   1000\n\n"

    "   Test the number of provided values:\n\n"

    "   >>> len(solver.out%1%)\n"
    "   3\n\n"

    "See also:\n"
    "   Receiver of %3%: :class:`plask.flow.%1%Receiver%2%`\n";
}

PLASK_PYTHON_API template <> const char* docstring_provider<FIELD_PROPERTY>() { return
    "%1%Provider%2%(data)\n\n"

    "Provider of the %3%%4% [%7%].\n\n"

    "This class is used for %3% provider in binary solvers. You can also create\n"
    "a custom provider for your Python solver.\n\n"

    "Args:\n"
    "   data: ``Data`` object to interpolate or callable returning it for given mesh.\n"
    "       The callable must accept the same arguments as the provider\n"
    "       ``__call__`` method (see below).\n\n"

    "To obtain the value from the provider simply call it. The call signature\n"
    "is as follows:\n\n"

    ".. method:: solver.out%1%(mesh%5%, interpolation='default')\n\n"

    "   :param mesh mesh: Target mesh to get the field at.\n"
    "   :param str interpolation: Requested interpolation method.\n"
    "   %6%\n"

    "   :return: Data with the %3% on the specified mesh **[%7%]**.\n\n"

    "Example:\n"
    "   Connect the provider to a receiver in some other solver:\n\n"

    "   >>> other_solver.in%1% = solver.out%1%\n\n"

    "   Obtain the provided field:\n\n"

    "   >>> solver.out%1%(mesh%5%)\n"
    "   <plask.Data at 0x1234567>\n\n"

    "See also:\n"
    "   Receiver of %3%: :class:`plask.flow.%1%Receiver%2%`\n"
    "   Data filter for %3%: :class:`plask.flow.%1%Filter%2%`";
}

PLASK_PYTHON_API template <> const char* docstring_provider<MULTI_FIELD_PROPERTY>() { return
    "%1%Provider%2%(data)\n\n"

    "Provider of the %3%%4% [%7%].\n\n"

    "This class is used for %3% provider in binary solvers. You can also create\n"
    "a custom provider for your Python solver.\n\n"

    "Args:\n"
    "   data: ``Data`` object to interpolate or callable returning it for given mesh.\n"
    "       The callable must accept the same arguments as the provider\n"
    "       ``__call__`` method (see below). It must also be able to give its\n"
    "       length (i.e. have the ``__len__`` method defined) that gives the\n"
    "       number of different provided values.\n\n"

    "To obtain the value from the provider simply call it. The call signature\n"
    "is as follows:\n\n"

    ".. method:: solver.out%1%(n=0, mesh%5%, interpolation='default')\n\n"

    "   :param int n: Value number.\n"
    "   :param mesh mesh: Target mesh to get the field at.\n"
    "   :param str interpolation: Requested interpolation method.\n"
    "   %6%\n"

    "   :return: Data with the %3% on the specified mesh **[%7%]**.\n\n"

    "You may obtain the number of different values this provider can return by\n"
    "testing its length.\n\n"

    "Example:\n"
    "   Connect the provider to a receiver in some other solver:\n\n"

    "   >>> other_solver.in%1% = solver.out%1%\n\n"

    "   Obtain the provided field:\n\n"

    "   >>> solver.out%1%(0, mesh%5%)\n"
    "   <plask.Data at 0x1234567>\n\n"

    "   Test the number of provided values:\n\n"

    "   >>> len(solver.out%1%)\n"
    "   3\n\n"

    "See also:\n"
    "   Receiver of %3%: :class:`plask.flow.%1%Receiver%2%`\n"
    "   Data filter for %3%: :class:`plask.flow.%1%Filter%2%`";
}

PLASK_PYTHON_API template const char* docstring_provider<SINGLE_VALUE_PROPERTY>();
PLASK_PYTHON_API template const char* docstring_provider<MULTI_VALUE_PROPERTY>();
PLASK_PYTHON_API template const char* docstring_provider<FIELD_PROPERTY>();
PLASK_PYTHON_API template const char* docstring_provider<MULTI_FIELD_PROPERTY>();


py::object flow_module;

/**
 * Register standard properties to Python.
 */
void register_standard_properties()
{
    flow_module = py::object(py::handle<>(py::borrowed(PyImport_AddModule("plask.flow"))));
    py::scope().attr("flow") = flow_module;
    flow_module.attr("__doc__") =
        "Data flow classes for standard properties.\n\n"

        "This module contains providers, receivers, and filters for standard properties.\n"
        "These classes are present in binary solvers, but you may also use them in your\n"
        "custom Python solvers.\n\n"

        ".. rubric:: Providers and Receivers\n\n"

        "Existing providers can be connected to receivers by using a simple assignment\n"
        "operator:\n\n"

        ">>> first_solver.inTemperature = second_solver.outTemperature\n\n"

        "You can manually retrieve data from any provider or a connected receiver by\n"
        "calling it like a function:\n\n"

        ">>> second_solver.outTemperature(mymesh)\n"
        "<plask.Data at 0x584c140>\n"
        ">>> first_solver.inTemperature(mymesh, 'spline')\n"
        "<plask.Data at 0x584c140>\n\n"

        "Providers and receivers of most quantities give spatial distributions of the\n"
        "corresponding fields and, thus, require the target mesh as its argument. In\n"
        "addition you may specify the interpolation method as in the example above.\n"
        "If the interpolation method is omitted, its default value, depending is assumed\n"
        "by the solver automatically.\n\n"

        "Some properties (e.g. the light intensity) require the result number given as\n"
        "the first argument (this is e.g. the consecutive mode number). Others take some\n"
        "optional arguments that are specified at the end (e.g. the gain requires to be\n"
        "given the wavelength at which the gain is computed.\n\n"

        "In PLaSK you can create your custom Python solvers. They may contain the default\n"
        "providers and receivers defined here. Receivers are simple objects that can be\n"
        "attached to providers later and read as shown above. On the contrary, providers\n"
        "require you to create a callable that returns the data to be provided when\n"
        "requested.\n\n"

        "Example:\n"
        "   To create the solver that gets temperature from another source and\n"
        "   increases it by 60 K, use the following class:\n\n"

        "   >>> class Hotter(object):\n"
        "   ...     def __init__(self):\n"
        "   ...         self.inTemperature = flow.TemperatureReceiver2D()\n"
        "   ...         self.outTemperature = flow.TemperatureProvider2D(\n"
        "   ...             lambda mesh, meth: self.get_data(mesh, meth))\n"
        "   ...     def get_data(self, mesh, method):\n"
        "   ...         temp = self.inTemperature(mesh, method)\n"
        "   ...         return temp.array + 60.\n\n"

        ".. rubric:: Filters\n\n"

        "Filters are solver-like classes that translate the fields computed in one\n"
        "geometry to another one. This geometry can have either the same or different\n"
        "dimension.\n\n"

        "All filter classes are used the same way. They are constructed with a single\n"
        "argument, which is a target geometry. The type of this geometry must match\n"
        "the suffix of the filter (``2D`` for two-dimensional Cartesian geometry, ``Cyl``\n"
        "for axi-symmetric cylindrical geometry, and ``3D`` for three-dimensional one.\n"
        "An example temperature filter for target 2D geometry can be constructed as\n"
        "follows:\n\n"

        ">>> temp_filter = flow.TemperatureFilter2D(mygeometry2d)\n\n"

        "Having an existing filter, you may attach a source provider to it, using bracket\n"
        "indexing. The `index` is a geometry object either existing in the target geometry\n"
        "or containing it (e.g. a :class:`geometry.Extrusion` object that is the root of\n"
        "the ``my_geometry_2d`` geometry). The `indexed` element is a proper data receiver\n"
        "that can be used for connecting the source data.\n\n"

        ">>> temp_filter[some_object_in_mygeometry2d]\n"
        "<plask.ReceiverForTemperature2D at 0x43a5210>\n"
        ">>> temp_filter[mygeometry2d.extrusion]\n"
        "<plask.ReceiverForTemperature3D at 0x44751a0>\n\n"
        ">>> temp_filter[mygeometry2d.extrusion] = thermal_solver_3d.outTemperature\n\n"

        "After connecting the source, the tranlated data can be obtained using the filter\n"
        "member ``out``, which is a provider that can be connected to other solvers.\n\n"

        ">>> temp_filter.out\n"
        "<plask.ProviderForTemperature2D at 0x43a5fa0>\n"
        ">>> other_solver_in_2d.inTemperature = temp_filter.out\n\n"

        "After the connection the filter does its job automatically.\n\n"

        "See also:\n"
        "   :ref:`sec-solvers-filters`.\n\n"

        "   Definition of filters in the XPL file: :xml:tag:`filter` tag.\n\n"

        "   Example using filters: :ref:`sec-tutorial-threshold-of-array`.\n"
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

    register_standard_properties_builtin_potential();
    register_standard_properties_quasi_Fermi_electron_level();
    register_standard_properties_quasi_Fermi_hole_level();
    register_standard_properties_conduction_band_edge();
    register_standard_properties_valence_band_edge();
    
    register_standard_properties_gain();
    register_standard_properties_GainOverCarriersConcentration();

    register_standard_properties_optical();
    register_standard_properties_refractive();
}

}} // namespace plask>();
