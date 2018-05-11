#include "python_globals.h"
#include "python_property.h"

namespace plask { namespace python {

// void register_standard_properties_thermal();
// void register_standard_properties_electrical();
void register_standard_properties_gain();
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
void register_standard_properties_band_edges();

void register_standard_properties_builtin_potential();
void register_standard_properties_quasi_Fermi_levels();
void register_standard_properties_energy_levels();

PLASK_PYTHON_API const char* docstring_receiver =
    u8"{0}Receiver{1}()\n\n"

    u8"Receiver of the {2}{3} [{4}].\n\n"

    u8"You may connect a provider to this receiver usign either the `connect` method\n"
    u8"or an assignement operator. Then, you can read the provided value by calling\n"
    u8"this receiver with arguments identical as the ones of the corresponding\n"
    u8"provider :class:`~plask.flow.{0}Provider{1}`.\n\n"

    u8"Example:\n"
    u8"   Connect the reveiver to a provider from some other solver:\n\n"

    u8"   >>> solver.in{0} = other_solver.out{0}\n\n"

    u8"See also:\n"
    u8"   Provider of {2}: :class:`plask.flow.{0}Provider{1}`\n\n"
    u8"   Data filter for {2}: :class:`plask.flow.{0}Filter{1}`";

PLASK_PYTHON_API const char* docstring_receiver_attach =
    u8"Attach some provider or constant value to the receiver.\n\n"

    u8"Args:\n"
    u8"    source: source provider or value.\n\n"

    u8"Example:\n"
    u8"   >>> solver.in{0}.attach(300.)\n"
    u8"   >>> solver.in{0}(any_mesh)[0]\n"
    u8"   300.\n"
    u8"   >>> solver.in{0}(any_mesh)[-1]\n"
    u8"   300.\n\n"
    u8"   >>> solver.in{0}.attach(other_solver.out{0})\n\n"

    u8"Note:\n"
    u8"   You may achieve the same effect by using the asignmnent operator\n"
    u8"   if you put an exisiting provider at the right side of this operator:\n\n"

    u8"   >>> solver.in{0} = other_solver.out{0}\n";

template <PropertyType propertyType> PLASK_PYTHON_API const char* docstring_provider_impl();

template <> PLASK_PYTHON_API const char* docstring_provider_impl<SINGLE_VALUE_PROPERTY>() { return
    u8"{0}Provider{1}(data)\n\n"

    u8"Provider of the {2}{3} [{6}].\n\n"

    u8"This class is used for {2} provider in binary solvers.\n"
    u8"You can also create a custom provider for your Python solver.\n\n"

    u8"Args:\n"
    u8"   data: provided value or callable returning it on request.\n"
    u8"         The callable must accept the same arguments as the provider\n"
    u8"         ``__call__`` method (see below).\n\n"

    u8"To obtain the value from the provider simply call it. The call signature\n"
    u8"is as follows:\n\n"

    u8".. method:: solver.out{0}({4})\n\n"

    u8"   {5}\n"

    u8"   :return: Value of the {2} **[{6}]**.\n\n"

    u8"Example:\n"
    u8"   Connect the provider to a receiver in some other solver:\n\n"

    u8"   >>> other_solver.in{0} = solver.out{0}\n\n"

    u8"   Obtain the provided value:\n\n"

    u8"   >>> solver.out{0}({4})\n"
    u8"   1000\n\n"

    u8"See also:\n"
    u8"   Receiver of {2}: :class:`plask.flow.{0}Receiver{1}`\n";
}

template <> PLASK_PYTHON_API const char* docstring_provider_impl<MULTI_VALUE_PROPERTY>() { return
    u8"{0}Provider{1}(data)\n\n"

    u8"Provider of the {2}{3} [{6}].\n\n"

    u8"This class is used for {2} provider in binary solvers.\n"
    u8"You can also create a custom provider for your Python solver.\n\n"

    u8"Args:\n"
    u8"   data: provided value or callable returning it on request.\n"
    u8"       The callable must accept the same arguments as the provider\n"
    u8"       ``__call__`` method (see below). It must also be able to give its\n"
    u8"       length (i.e. have the ``__len__`` method defined) that gives the\n"
    u8"       number of different provided values.\n\n"

    u8"To obtain the value from the provider simply call it. The call signature\n"
    u8"is as follows:\n\n"

    u8".. method:: solver.out{0}({7}{4})\n\n"

    u8"   {8}\n"
    u8"   {5}\n"

    u8"   :return: Value of the {2} **[{6}]**.\n\n"

    u8"You may obtain the number of different values this provider can return by\n"
    u8"testing its length.\n\n"

    u8"Example:\n"
    u8"   Connect the provider to a receiver in some other solver:\n\n"

    u8"   >>> other_solver.in{0} = solver.out{0}\n\n"

    u8"   Obtain the provided value:\n\n"

    u8"   >>> solver.out{0}(0{4})\n"
    u8"   1000\n\n"

    u8"   Test the number of provided values:\n\n"

    u8"   >>> len(solver.out{0})\n"
    u8"   3\n\n"

    u8"See also:\n"
    u8"   Receiver of {2}: :class:`plask.flow.{0}Receiver{1}`\n";
}

template <> PLASK_PYTHON_API const char* docstring_provider_impl<FIELD_PROPERTY>() { return
    u8"{0}Provider{1}(data)\n\n"

    u8"Provider of the {2}{3} [{6}].\n\n"

    u8"This class is used for {2} provider in binary solvers.\n"
    u8"You can also create a custom provider for your Python solver.\n\n"

    u8"Args:\n"
    u8"   data: ``Data`` object to interpolate or callable returning it for given mesh.\n"
    u8"       The callable must accept the same arguments as the provider\n"
    u8"       ``__call__`` method (see below).\n\n"

    u8"To obtain the value from the provider simply call it. The call signature\n"
    u8"is as follows:\n\n"

    u8".. method:: solver.out{0}(mesh{4}, interpolation='default')\n\n"

    u8"   :param mesh mesh: Target mesh to get the field at.\n"
    u8"   :param str interpolation: Requested interpolation method.\n"
    u8"   {5}\n"

    u8"   :return: Data with the {2} on the specified mesh **[{6}]**.\n\n"

    u8"Example:\n"
    u8"   Connect the provider to a receiver in some other solver:\n\n"

    u8"   >>> other_solver.in{0} = solver.out{0}\n\n"

    u8"   Obtain the provided field:\n\n"

    u8"   >>> solver.out{0}(mesh{4})\n"
    u8"   <plask.Data at 0x1234567>\n\n"

    u8"See also:\n"
    u8"   Receiver of {2}: :class:`plask.flow.{0}Receiver{1}`\n"
    u8"   Data filter for {2}: :class:`plask.flow.{0}Filter{1}`";
}

template <> PLASK_PYTHON_API const char* docstring_provider_impl<MULTI_FIELD_PROPERTY>() { return
    u8"{0}Provider{1}(data)\n\n"

    u8"Provider of the {2}{3} [{6}].\n\n"

    u8"This class is used for {2} provider in binary solvers.\n"
    u8"You can also create a custom provider for your Python solver.\n\n"

    u8"Args:\n"
    u8"   data: ``Data`` object to interpolate or callable returning it for given mesh.\n"
    u8"       The callable must accept the same arguments as the provider\n"
    u8"       ``__call__`` method (see below). It must also be able to give its\n"
    u8"       length (i.e. have the ``__len__`` method defined) that gives the\n"
    u8"       number of different provided values.\n\n"

    u8"To obtain the value from the provider simply call it. The call signature\n"
    u8"is as follows:\n\n"

    u8".. method:: solver.out{0}({7}, mesh{4}, interpolation='default')\n\n"

    u8"   {8}\n"
    u8"   :param mesh mesh: Target mesh to get the field at.\n"
    u8"   :param str interpolation: Requested interpolation method.\n"
    u8"   {5}\n"

    u8"   :return: Data with the {2} on the specified mesh **[{6}]**.\n\n"

    u8"You may obtain the number of different values this provider can return by\n"
    u8"testing its length.\n\n"

    u8"Example:\n"
    u8"   Connect the provider to a receiver in some other solver:\n\n"

    u8"   >>> other_solver.in{0} = solver.out{0}\n\n"

    u8"   Obtain the provided field:\n\n"

    u8"   >>> solver.out{0}(0, mesh{4})\n"
    u8"   <plask.Data at 0x1234567>\n\n"

    u8"   Test the number of provided values:\n\n"

    u8"   >>> len(solver.out{0})\n"
    u8"   3\n\n"

    u8"See also:\n"
    u8"   Receiver of {2}: :class:`plask.flow.{0}Receiver{1}`\n"
    u8"   Data filter for {2}: :class:`plask.flow.{0}Filter{1}`";
}

// template PLASK_PYTHON_API const char* docstring_provider_impl<SINGLE_VALUE_PROPERTY>();
// template PLASK_PYTHON_API const char* docstring_provider_impl<MULTI_VALUE_PROPERTY>();
// template PLASK_PYTHON_API const char* docstring_provider_impl<FIELD_PROPERTY>();
// template PLASK_PYTHON_API const char* docstring_provider_impl<MULTI_FIELD_PROPERTY>();


PLASK_PYTHON_API py::object flow_module;

/**
 * Register standard properties to Python.
 */
void register_standard_properties()
{
    flow_module = py::object(py::handle<>(py::borrowed(PyImport_AddModule("plask.flow"))));
    py::scope().attr("flow") = flow_module;
    flow_module.attr("__doc__") =
        u8"Data flow classes for standard properties.\n\n"

        u8"This module contains providers, receivers, and filters for standard properties.\n"
        u8"These classes are present in binary solvers, but you may also use them in your\n"
        u8"custom Python solvers.\n\n"

        u8".. rubric:: Providers and Receivers\n\n"

        u8"Existing providers can be connected to receivers by using a simple assignment\n"
        u8"operator:\n\n"

        u8">>> first_solver.inTemperature = second_solver.outTemperature\n\n"

        u8"You can manually retrieve data from any provider or a connected receiver by\n"
        u8"calling it like a function:\n\n"

        u8">>> second_solver.outTemperature(mymesh)\n"
        u8"<plask.Data at 0x584c140>\n"
        u8">>> first_solver.inTemperature(mymesh, 'spline')\n"
        u8"<plask.Data at 0x584c140>\n\n"

        u8"Providers and receivers of most quantities give spatial distributions of the\n"
        u8"corresponding fields and, thus, require the target mesh as its argument. In\n"
        u8"addition you may specify the interpolation method as in the example above.\n"
        u8"If the interpolation method is omitted, its default value, depending is assumed\n"
        u8"by the solver automatically.\n\n"

        u8"Some properties (e.g. the light intensity) require the result number given as\n"
        u8"the first argument (this is e.g. the consecutive mode number). Others take some\n"
        u8"optional arguments that are specified at the end (e.g. the gain requires to be\n"
        u8"given the wavelength at which the gain is computed.\n\n"

        u8"In PLaSK you can create your custom Python solvers. They may contain the default\n"
        u8"providers and receivers defined here. Receivers are simple objects that can be\n"
        u8"attached to providers later and read as shown above. On the contrary, providers\n"
        u8"require you to create a callable that returns the data to be provided when\n"
        u8"requested.\n\n"

        u8"Example:\n"
        u8"   To create the solver that gets temperature from another source and\n"
        u8"   increases it by 60 K, use the following class:\n\n"

        u8"   >>> class Hotter(object):\n"
        u8"   ...     def __init__(self):\n"
        u8"   ...         self.inTemperature = flow.TemperatureReceiver2D()\n"
        u8"   ...         self.outTemperature = flow.TemperatureProvider2D(\n"
        u8"   ...             lambda mesh, meth: self.get_data(mesh, meth))\n"
        u8"   ...     def get_data(self, mesh, method):\n"
        u8"   ...         temp = self.inTemperature(mesh, method)\n"
        u8"   ...         return temp.array + 60.\n\n"

        u8".. rubric:: Filters\n\n"

        u8"Filters are solver-like classes that translate the fields computed in one\n"
        u8"geometry to another one. This geometry can have either the same or different\n"
        u8"dimension.\n\n"

        u8"All filter classes are used the same way. They are constructed with a single\n"
        u8"argument, which is a target geometry. The type of this geometry must match\n"
        u8"the suffix of the filter (``2D`` for two-dimensional Cartesian geometry, ``Cyl``\n"
        u8"for axi-symmetric cylindrical geometry, and ``3D`` for three-dimensional one.\n"
        u8"An example temperature filter for target 2D geometry can be constructed as\n"
        u8"follows:\n\n"

        u8">>> temp_filter = flow.TemperatureFilter2D(mygeometry2d)\n\n"

        u8"Having an existing filter, you may attach a source provider to it, using bracket\n"
        u8"indexing. The `index` is a geometry object either existing in the target geometry\n"
        u8"or containing it (e.g. a :class:`geometry.Extrusion` object that is the root of\n"
        u8"the ``my_geometry_2d`` geometry). The `indexed` element is a proper data receiver\n"
        u8"that can be used for connecting the source data.\n\n"

        u8">>> temp_filter[some_object_in_mygeometry2d]\n"
        u8"<plask.ReceiverForTemperature2D at 0x43a5210>\n"
        u8">>> temp_filter[mygeometry2d.extrusion]\n"
        u8"<plask.ReceiverForTemperature3D at 0x44751a0>\n\n"
        u8">>> temp_filter[mygeometry2d.extrusion] = thermal_solver_3d.outTemperature\n\n"

        u8"After connecting the source, the tranlated data can be obtained using the filter\n"
        u8"member ``out``, which is a provider that can be connected to other solvers.\n\n"

        u8">>> temp_filter.out\n"
        u8"<plask.ProviderForTemperature2D at 0x43a5fa0>\n"
        u8">>> other_solver_in_2d.inTemperature = temp_filter.out\n\n"

        u8"After the connection the filter does its job automatically.\n\n"

        u8"See also:\n"
        u8"   :ref:`sec-solvers-filters`.\n\n"

        u8"   Definition of filters in the XPL file: :xml:tag:`filter` tag.\n\n"

        u8"   Example using filters: :ref:`sec-tutorial-threshold-of-array`.\n"
    ;

    register_standard_properties_thermal();
    register_standard_properties_temperature();
    register_standard_properties_heatdensity();
    register_standard_properties_heatflux();

    register_standard_properties_electrical();
    register_standard_properties_voltage();
    register_standard_properties_current();
    register_standard_properties_concentration_carriers();

    register_standard_properties_builtin_potential();
    register_standard_properties_quasi_Fermi_levels();
    register_standard_properties_energy_levels();
    register_standard_properties_band_edges();

    register_standard_properties_gain();

    register_standard_properties_optical();
    register_standard_properties_refractive();
}

}} // namespace plask>();
