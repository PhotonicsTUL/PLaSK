#ifndef PLASK__OPTICAL_H
#define PLASK__OPTICAL_H

#include <plask/math.h>
#include <plask/provider/providerfor.h>
#include <plask/provider/scaled_provider.h>
#include <plask/provider/combined_provider.h>

namespace plask {

/**
 * Profile of the optical E field per unit power [kΩ/m²].
 *
 * This quantity that may be multiplied by the actual emitted power [mW] allows to determine
 * the actual electric field [V/m] inside the laser.
 *
 * This property should be provided by every optical solver as it has a nice advantage that does not depend on
 * the internal representation of the field (whether it is scalar or vectorial one).
 */
struct OpticalIntensity: public ScalarFieldProperty {
    static constexpr const char* NAME = "intensity profile";
};

/**
 * Profile of the optical E field [V²/m²].
 *
 * This property may be obtained by scaling the OpticalIntensity by the emitted power.
 */
struct LightIntensity: public ScalarFieldProperty {
    static constexpr const char* NAME = "light intensity";
};

/**
 * Provider which scales intensity profile to get light intensity
 */
template <typename SpaceT>
struct LightIntensityProvider: public ScaledFieldProvider<LightIntensity, OpticalIntensity, SpaceT> {};

/**
 * Provider which sums light intensities from one or more sources.
 */
template <typename SpaceT>
struct LightIntensitySumProvider: public FieldSumProvider<LightIntensity, SpaceT> {};


/**
 * Wavelength [nm]. It can be either computed by some optical solvers or set by the user.
 *
 * It is a complex number, so it can contain information about both the wavelength and losses.
 * Its imaginary part is defined as \f$ \Im(\lambda)=-\frac{\Re(\lambda)^2}{2\pi c}\Im(\omega) \f$.
 */
struct Wavelength: public SingleValueProperty<double> {
    static constexpr const char* NAME = "wavelength";
};

/**
 * Modal loss [1/cm].
 */
struct ModalLoss: public SingleValueProperty<double> {
    static constexpr const char* NAME = "modal extinction";
};

/**
 * Propagation constant [1/µm]. It can be either computed by some optical solvers or set by the user.
 *
 * It is a complex number, so it can contain information about both the propagation and losses.
 */
struct PropagationConstant: public SingleValueProperty<dcomplex> {
    static constexpr const char* NAME = "propagation constant";
};

/**
 * Effective index. It can be either computed by some optical solvers or set by the user.
 *
 * It is a complex number, so it can contain information about both the propagation and losses.
 */
struct EffectiveIndex: public SingleValueProperty<dcomplex> {
    static constexpr const char* NAME = "effective index";
};

} // namespace plask

#endif // PLASK__OPTICAL_H
