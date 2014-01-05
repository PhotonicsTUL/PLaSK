#ifndef PLASK__PYTHON_PROPERTY_DESC_H
#define PLASK__PYTHON_PROPERTY_DESC_H

#include <plask/properties/gain.h>

namespace plask { namespace python {

template <typename PropertyT> inline const char* docstrig_property_optional_args() { return ""; }

template <> inline const char* docstrig_property_optional_args<Gain>() { return ", wavelength"; }

template <> inline const char* docstrig_property_optional_args<GainOverCarriersConcentration>() { return ", wavelength"; }


template <typename PropertyT> inline const char* docstrig_property_optional_args_desc() { return ""; }

template <> inline const char* docstrig_property_optional_args_desc<Gain>() { return
    "``wavelength`` is the wavelength at which the gain is computed.\n";
}

template <> inline const char* docstrig_property_optional_args_desc<GainOverCarriersConcentration>() { return
    "``wavelength`` is the wavelength at which the gain is computed.\n";
}



}} // namespace plask

#endif // PLASK__PYTHON_PROPERTY_DESC_H
