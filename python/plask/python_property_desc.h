#ifndef PLASK__PYTHON_PROPERTY_DESC_H
#define PLASK__PYTHON_PROPERTY_DESC_H

#include <boost/python.hpp>

#include <plask/properties/gain.h>
#include <plask/properties/optical.h>

namespace plask { namespace python {

template <typename PropertyT> inline const char* docstrig_property_optional_args() { return ""; }
template <typename PropertyT> inline const char* docstrig_property_optional_args_desc() { return ""; }


template <> inline const char* docstrig_property_optional_args<Gain>() { return ", wavelength"; }
template <> inline const char* docstrig_property_optional_args<GainOverCarriersConcentration>() { return ", wavelength"; }
template <> inline const char* docstrig_property_optional_args_desc<Gain>() { return
    ":param float wavelength: The wavelength at which the gain is computed [nm].\n";
}
template <> inline const char* docstrig_property_optional_args_desc<GainOverCarriersConcentration>() { return
    ":param float wavelength: The wavelength at which the gain is computed [nm].\n";
}
template <> struct PropertyArgsField<Gain> {
    static py::detail::keywords<4> value() {
        return boost::python::arg("self"), boost::python::arg("mesh"), boost::python::arg("wavelength"), boost::python::arg("interpolation")=INTERPOLATION_DEFAULT;
    }
};
template <> struct PropertyArgsField<GainOverCarriersConcentration> {
    static py::detail::keywords<4> value() {
        return boost::python::arg("self"), boost::python::arg("mesh"), boost::python::arg("wavelength"), boost::python::arg("interpolation")=INTERPOLATION_DEFAULT;
    }
};



}} // namespace plask

#endif // PLASK__PYTHON_PROPERTY_DESC_H
