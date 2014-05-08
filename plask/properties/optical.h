#ifndef PLASK__OPTICAL_H
#define PLASK__OPTICAL_H

#include <plask/math.h>
#include <plask/provider/providerfor.h>
#include <plask/provider/scaled_provider.h>
#include <plask/provider/combined_provider.h>

namespace plask {

/**
 * Refractive index tensor
 */
struct RefractiveIndex: FieldProperty<Tensor3<dcomplex>,double> {
    static constexpr const char* NAME = "refractive index";
    static constexpr const char* UNIT = "-";
    static inline Tensor3<dcomplex> getDefaultValue() { return Tensor3<dcomplex>(1.); }
};

/**
 * Profile of the optical field 1/2 E × conj(E) / Z0 [W/m²].
 *
 * Additional integer parameter is mode number.
 */
struct LightMagnitude: public MultiFieldProperty<double> {
    static constexpr const char* NAME = "optical field magnitude";
    static constexpr const char* UNIT = "W/m²";
};

/**
 * Provider which scales intensity profile to get light intensity
 */
template <typename SpaceT>
struct LightMagnitudeProvider: public ScaledFieldProvider<LightMagnitude, LightMagnitude, SpaceT> {};

/**
 * Provider which sums light intensities from one or more sources.
 */
template <typename SpaceT>
struct LightMagnitudeSumProvider: public FieldSumProvider<LightMagnitude, SpaceT> {};


/**
 * Profile of the optical electric field [V/m].
 *
 * Additional integer parameter is mode number.
 */
struct OpticalElectricField: public MultiFieldProperty<Vec<3,dcomplex>> {
    static constexpr const char* NAME = "electric field";
    static constexpr const char* UNIT = "V/m";
};


/**
 * Profile of the optical magnetic field [A/m].
 *
 * Additional integer parameter is mode number.
 */
struct OpticalMagneticField: public MultiFieldProperty<Vec<3,dcomplex>> {
    static constexpr const char* NAME = "magnetic field";
    static constexpr const char* UNIT = "A/m";
};


/**
 * Wavelength [nm]. It can be either computed by some optical solvers or set by the user.
 *
 * It is a complex number, so it can contain information about both the wavelength and losses.
 * Its imaginary part is defined as \f$ \Im(\lambda)=-\frac{\Re(\lambda)^2}{2\pi c}\Im(\omega) \f$.
 *
 * Additional integer parameter is mode number.
 */
struct Wavelength: public MultiValueProperty<double> {
    static constexpr const char* NAME = "wavelength";
    static constexpr const char* UNIT = "nm";
};

/**
 * Modal loss [1/cm].
 *
 * Additional integer parameter is mode number.
 */
struct ModalLoss: public MultiValueProperty<double> {
    static constexpr const char* NAME = "modal extinction";
    static constexpr const char* UNIT = "1/cm";
};

/**
 * Propagation constant [1/µm]. It can be either computed by some optical solvers or set by the user.
 *
 * It is a complex number, so it can contain information about both the propagation and losses.
 *
 * Additional integer parameter is mode number.
 */
struct PropagationConstant: public MultiValueProperty<dcomplex> {
    static constexpr const char* NAME = "propagation constant";
    static constexpr const char* UNIT = "1/µm";
};

/**
 * Effective index. It can be either computed by some optical solvers or set by the user.
 *
 * It is a complex number, so it can contain information about both the propagation and losses.
 *
 * Additional integer parameter is mode number.
 */
struct EffectiveIndex: public MultiValueProperty<dcomplex> {
    static constexpr const char* NAME = "effective index";
    static constexpr const char* UNIT = "-";
};

} // namespace plask

#endif // PLASK__OPTICAL_H
