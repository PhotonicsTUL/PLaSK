#ifndef PLASK__OPTICAL_H
#define PLASK__OPTICAL_H

#include "../math.h"
#include "providerfor.h"

namespace plask {

/**
 * Intensity of optical E field. It is calculated as abs(E), providing that E is electric field vector.
 *
 * This property should be provided by every optical solver as it has a nice advantage that does not depend on
 * the internal representation of the field (whether it is scalar or vectorial one).
 */
struct OpticalIntensity : public ScalarFieldProperty {
    static constexpr const char* NAME = "light intensity";
};

/**
 * Wavelength. It can be either computed by some optical solvers or set by the user.
 *
 * It is a complex number, so it can contain information about both the wavelength and losses.
 * Its imaginary part is defined as \f$ \Im(\lambda)=-\frac{\Re(\lambda)^2}{2\pi c}\Im(\omega) \f$.
 */
struct Wavelength : public SingleValueProperty<dcomplex> {
    static constexpr const char* NAME = "wavelength";
};

/**
 * Propagation constant. It can be either computed by some optical solvers or set by the user.
 *
 * It is a complex number, so it can contain information about both the propagation and losses.
 */
struct PropagationConstant : public SingleValueProperty<dcomplex> {
    static constexpr const char* NAME = "propagation constant";
};

/**
 * Effective index. It can be either computed by some optical solvers or set by the user.
 *
 * It is a complex number, so it can contain information about both the propagation and losses.
 */
struct EffectiveIndex : public SingleValueProperty<dcomplex> {
    static constexpr const char* NAME = "effective index";
};

} // namespace plask

#endif // PLASK__OPTICAL_H
