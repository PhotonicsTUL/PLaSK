#ifndef PLASK__GAIN_H
#define PLASK__GAIN_H

#include "../math.h"
#include "providerfor.h"

namespace plask {

/**
 * Material gain [1/cm].
 *
 * This is the gain property. It should have the same unit as absorption.
 * Providers must set it to NaN everywhere outside of the active region.
 * Optical solvers should thread NaNs as zeros.
 * It can also be set negative in case there is some absorption which is not
 * covered by the material database.
 *
 * It can also be a gain profile. Some optical solvers can determine
 * the threshold gain as a constant, which should be added to it in order to
 * obtain the zero modal gain (threshold). The regions where it is NaN should
 * not be affected.
 *
 * Providers of material gain should accept additional parameter,
 * which is the wavelength for which the gain should be computed.
 */
struct MaterialGain : public FieldProperty<double, double> {
    static constexpr const char* NAME = "material gain";
};


} // namespace plask

#endif // PLASK__GAIN_H
