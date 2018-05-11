#ifndef PLASK__PYTHON_PROPERTY_DESC_H
#define PLASK__PYTHON_PROPERTY_DESC_H

#include <boost/python.hpp>

#include <plask/properties/gain.h>
#include <plask/properties/electrical.h>
#include <plask/properties/optical.h>

namespace plask { namespace python {

template <typename PropertyT> inline const char* docstrig_property_optional_args() { return ""; }
template <typename PropertyT> inline const char* docstrig_property_optional_args_desc() { return ""; }
template <typename PropertyT> inline const char* docstring_provider_multi_param() { return "n=0"; }
template <typename PropertyT> inline const char* docstring_provider_multi_param_desc() { return u8":param int n: Value number.\n"; }


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
template <> inline const char* docstrig_property_optional_args<Gain>() { return ", wavelength"; }
template <> inline const char* docstrig_property_optional_args_desc<Gain>() { return
    u8":param float wavelength: The wavelength at which the gain is computed [nm].\n";
}
template <> inline const char* docstring_provider_multi_param<Gain>() { return "deriv=''"; }
template <> inline const char* docstring_provider_multi_param_desc<Gain>() {
    return ":param str deriv: Gain derivative to return. can be '' (empty) or 'conc'.\n"
           "                  In the latter case, the gain derivative over carriers\n"
           "                  concentration is returned.\n";
}


template <> struct PropertyArgsMultiField<CarriersConcentration> {
    static py::detail::keywords<4> value() {
        return boost::python::arg("self"), boost::python::arg("type"), boost::python::arg("mesh"), boost::python::arg("interpolation")=INTERPOLATION_DEFAULT;
    }
};
template <> inline const char* docstring_provider_multi_param<CarriersConcentration>() { return "type=''"; }
template <> inline const char* docstring_provider_multi_param_desc<CarriersConcentration>() {
    return u8":param str type: Detailed information which carriers are returned. It can be\n"
           u8"                 'majority' to return majority carriers in given material,\n"
           u8"                 'pairs' for the concentration of electron-hole pairs,\n"
           u8"                 'electrons', or 'holes' for particular carriers type.\n";
}


}} // namespace plask

#endif // PLASK__PYTHON_PROPERTY_DESC_H
