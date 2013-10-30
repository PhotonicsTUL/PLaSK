#ifndef PLASK__GAIN_H
#define PLASK__GAIN_H

#include <plask/math.h>
#include <plask/provider/providerfor.h>

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
struct Gain : public FieldProperty<double, double> {
    static constexpr const char* NAME = "material gain";
    static inline double getDefaultValue() { return NAN; }
};

/**
 * Derivative of material gain over carriers concentration [cm^2].
 *
 * Providers of material gain derivative should accept additional parameter,
 * which is the wavelength for which the derivative should be computed.
 */
struct GainOverCarriersConcentration : public FieldProperty<double, double> {
    static constexpr const char* NAME = "material gain over carriers concentration derivative";
    static inline double getDefaultValue() { return 0.; }

};


} // namespace plask

#endif // PLASK__GAIN_H
