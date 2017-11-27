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
    u8":param float wavelength: The wavelength at which the gain is computed [nm].\n";
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
    u8"{0}Provider{1}(data)\n\n"

    u8"Provider of the {2}{3} [{6}].\n\n"

    u8"This class is used for {2} provider in binary solvers.\n"
    u8"You can also create a custom provider for your Python solver.\n\n"

    u8"Args:\n"
    u8"   data: ``Data`` object to interpolate or callable returning it for given mesh.\n"
    u8"       The callable must accept the same arguments as the provider\n"
    u8"       ``__call__`` method (see below). It must also be able to give its\n"
    u8"       length (i.e. have the ``__len__`` method defined) that gives the\n"
    u8"       number of different provided derivatives (including the gain itself).\n\n"

    u8"To obtain the value from the provider simply call it. The call signature\n"
    u8"is as follows:\n\n"

    u8".. method:: solver.out{0}(deriv, mesh{4}, interpolation='default')\n\n"

    u8"   :param str deriv: Gain derivative to return. can be '' (empty) or 'conc'.\n"
    u8"                     In the latter case, the gain derivative over carriers\n"
    u8"                     concentration is returned.\n"
    u8"   :param mesh mesh: Target mesh to get the field at.\n"
    u8"   :param str interpolation: Requested interpolation method.\n"
    u8"   {5}\n"

    u8"   :return: Data with the {2} on the specified mesh **[{6}]**.\n\n"

    u8"You may obtain the number of different derivatives this provider can return\n"
    u8"by testing its length.\n\n"

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
template <> constexpr const char* docstring_attr_provider<Gain>() { return
    u8"Provider of the computed {2} [{3}].\n"
    u8"{4}\n\n"

    u8"{7}(deriv='', mesh{5}, interpolation='default')\n\n"

    u8":param str deriv: Gain derivative to return. can be '' (empty) or 'conc'.\n"
    u8"                  In the latter case, the gain derivative over carriers\n"
    u8"                  concentration is returned.\n"
    u8":param mesh mesh: Target mesh to get the field at.\n"
    u8":param str interpolation: Requested interpolation method.\n"
    u8"{6}\n"

    u8":return: Data with the {2} on the specified mesh **[{3}]**.\n\n"

    u8"You may obtain the number of different values this provider can return by\n"
    u8"testing its length.\n\n"

    u8"Example:\n"
    u8"   Connect the provider to a receiver in some other solver:\n\n"

    u8"   >>> other_solver.in{0} = solver.{7}\n\n"

    u8"   Obtain the provided field:\n\n"

    u8"   >>> solver.{7}(mesh{5})\n"
    u8"   <plask.Data at 0x1234567>\n\n"

    u8"   Test the number of provided values:\n\n"

    u8"   >>> len(solver.{7})\n"
    u8"   3\n\n"

    u8"See also:\n\n"
    u8"   Provider class: :class:`plask.flow.{0}Provider{1}`\n\n"
    u8"   Receiver class: :class:`plask.flow.{0}Receiver{1}`\n";
}
template <>
constexpr const char* docstring_provider_call_multi_param<Gain>() {
    return ":param str deriv: Gain derivative to return. can be '' (empty) or 'conc'.\n"
           "                  In the latter case, the gain derivative over carriers\n"
           "                  concentration is returned.\n";
}


}} // namespace plask

#endif // PLASK__PYTHON_PROPERTY_DESC_H
