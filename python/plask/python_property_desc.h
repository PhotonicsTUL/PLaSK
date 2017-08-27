#ifndef PLASK__PYTHON_PROPERTY_DESC_H
#define PLASK__PYTHON_PROPERTY_DESC_H

#include <boost/python.hpp>

#include <plask/properties/gain.h>
#include <plask/properties/optical.h>

namespace plask { namespace python {

template <typename PropertyT> inline const char* docstrig_property_optional_args() { return ""; }
template <typename PropertyT> inline const char* docstrig_property_optional_args_desc() { return ""; }
template <typename PropertyT> static constexpr const char* docstring_provider();
template <typename PropertyT> static constexpr const char* docstring_attr_provider();


template <> inline const char* docstrig_property_optional_args<Gain>() { return ", wavelength"; }
template <> inline const char* docstrig_property_optional_args_desc<Gain>() { return
    ":param float wavelength: The wavelength at which the gain is computed [nm].\n";
}
template <> struct PropertyArgsField<Gain> {
    static py::detail::keywords<4> value() {
        return boost::python::arg("self"), boost::python::arg("mesh"), boost::python::arg("wavelength"), boost::python::arg("interpolation")=INTERPOLATION_DEFAULT;
    }
};
template <> struct PropertyArgsMultiField<Gain> {
    static py::detail::keywords<5> value() {
        return boost::python::arg("self"), boost::python::arg("deriv"), boost::python::arg("mesh"), boost::python::arg("wavelength"), boost::python::arg("interpolation")=INTERPOLATION_DEFAULT;
    }
};
template <> constexpr const char* docstring_provider<Gain>() { return
    "{0}Provider{1}(data)\n\n"

    "Provider of the {2}{3} [{6}].\n\n"

    "This class is used for {2} provider in binary solvers.\n"
    "You can also create a custom provider for your Python solver.\n\n"

    "Args:\n"
    "   data: ``Data`` object to interpolate or callable returning it for given mesh.\n"
    "       The callable must accept the same arguments as the provider\n"
    "       ``__call__`` method (see below). It must also be able to give its\n"
    "       length (i.e. have the ``__len__`` method defined) that gives the\n"
    "       number of different provided derivatives (including the gain itself).\n\n"

    "To obtain the value from the provider simply call it. The call signature\n"
    "is as follows:\n\n"

    ".. method:: solver.out{0}(deriv='', mesh{4}, interpolation='default')\n\n"

    "   :param str deriv: Gain derivative to return. can be '' (empty) or 'conc'.\n"
    "                     In the latter case, the gain derivative over carriers\n"
    "                     concentration is returned.\n"
    "   :param mesh mesh: Target mesh to get the field at.\n"
    "   :param str interpolation: Requested interpolation method.\n"
    "   {5}\n"

    "   :return: Data with the {2} on the specified mesh **[{6}]**.\n\n"

    "You may obtain the number of different derivatives this provider can return\n"
    "by testing its length.\n\n"

    "Example:\n"
    "   Connect the provider to a receiver in some other solver:\n\n"

    "   >>> other_solver.in{0} = solver.out{0}\n\n"

    "   Obtain the provided field:\n\n"

    "   >>> solver.out{0}(0, mesh{4})\n"
    "   <plask.Data at 0x1234567>\n\n"

    "   Test the number of provided values:\n\n"

    "   >>> len(solver.out{0})\n"
    "   3\n\n"

    "See also:\n"
    "   Receiver of {2}: :class:`plask.flow.{0}Receiver{1}`\n"
    "   Data filter for {2}: :class:`plask.flow.{0}Filter{1}`";
}
template <> constexpr const char* docstring_attr_provider<Gain>() { return
    "Provider of the computed {2} [{3}].\n"
    "{4}\n\n"

    "{7}(deriv='', mesh{5}, interpolation='default')\n\n"

    ":param str deriv: Gain derivative to return. can be '' (empty) or 'conc'.\n"
    "                  In the latter case, the gain derivative over carriers\n"
    "                  concentration is returned.\n"
    ":param mesh mesh: Target mesh to get the field at.\n"
    ":param str interpolation: Requested interpolation method.\n"
    "{6}\n"

    ":return: Data with the {2} on the specified mesh **[{3}]**.\n\n"

    "You may obtain the number of different values this provider can return by\n"
    "testing its length.\n\n"

    "Example:\n"
    "   Connect the provider to a receiver in some other solver:\n\n"

    "   >>> other_solver.in{0} = solver.{7}\n\n"

    "   Obtain the provided field:\n\n"

    "   >>> solver.{7}(mesh{5})\n"
    "   <plask.Data at 0x1234567>\n\n"

    "   Test the number of provided values:\n\n"

    "   >>> len(solver.{7})\n"
    "   3\n\n"

    "See also:\n\n"
    "   Provider class: :class:`plask.flow.{0}Provider{1}`\n\n"
    "   Receiver class: :class:`plask.flow.{0}Receiver{1}`\n";
}
template <>
constexpr const char* docstring_provider_call_multi_param<Gain>() {
    return ":param str deriv: Gain derivative to return. can be '' (empty) or 'conc'.\n"
           "                  In the latter case, the gain derivative over carriers\n"
           "                  concentration is returned.\n";
}


}} // namespace plask

#endif // PLASK__PYTHON_PROPERTY_DESC_H
