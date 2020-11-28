#ifndef PLASK__OPTICAL_H
#define PLASK__OPTICAL_H

#include "plask/math.hpp"
#include "plask/provider/providerfor.hpp"
#include "plask/provider/scaled_provider.hpp"
#include "plask/provider/combined_provider.hpp"

namespace plask {

/**
 * Refractive index tensor
 */
struct PLASK_API RefractiveIndex: FieldProperty<Tensor3<dcomplex>> {
    static constexpr const char* NAME = "refractive index";
    static constexpr const char* UNIT = "-";
    static inline Tensor3<dcomplex> getDefaultValue() { return Tensor3<dcomplex>(1.); }
};

/**
 * Profile of the optical field 1/2 E × conj(E) / Z0 [W/m²].
 */
struct PLASK_API LightMagnitude: public FieldProperty<double> {
    static constexpr const char* NAME = "optical field magnitude";
    static constexpr const char* UNIT = "W/m²";
};


/**
 * Profile of the optical electric field [V/m].
 */
struct PLASK_API LightE: public FieldProperty<Vec<3,dcomplex>> {
    static constexpr const char* NAME = "electric field";
    static constexpr const char* UNIT = "V/m";
};


/**
 * Profile of the optical magnetic field [A/m].
 */
struct PLASK_API LightH: public FieldProperty<Vec<3,dcomplex>> {
    static constexpr const char* NAME = "magnetic field";
    static constexpr const char* UNIT = "A/m";
};


/**
 * Profile of the optical field 1/2 E × conj(E) / Z0 [W/m²].
 *
 * Multimode version. Additional integer parameter is the mode number.
 */
struct PLASK_API ModeLightMagnitude: public MultiFieldProperty<double> {
    static constexpr const char* NAME = "optical field magnitude";
    static constexpr const char* UNIT = "W/m²";
};

/**
 * Provider which sums light intensities from one or more sources.
 */
template <typename SpaceT>
struct LightMagnitudeSumProvider: public FieldSumProvider<ModeLightMagnitude, SpaceT> {};


/**
 * Profile of the optical electric field [V/m].
 *
 * Multimode version. Additional integer parameter is the mode number.
 */
struct PLASK_API ModeLightE: public MultiFieldProperty<Vec<3,dcomplex>> {
    static constexpr const char* NAME = "electric field";
    static constexpr const char* UNIT = "V/m";
};


/**
 * Profile of the optical magnetic field [A/m].
 *
 * Multimode version. Additional integer parameter is the mode number.
 */
struct PLASK_API ModeLightH: public MultiFieldProperty<Vec<3,dcomplex>> {
    static constexpr const char* NAME = "magnetic field";
    static constexpr const char* UNIT = "A/m";
};


/**
 * Wavelength [nm]. It can be either computed by some optical solvers or set by the user.
 *
 * It is a complex number, so it can contain information about both the wavelength and losses.
 * Its imaginary part is defined as \f$ \Im(\lambda)=-\frac{\Re(\lambda)^2}{2\pi c}\Im(\omega) \f$.
 *
 * Additional integer parameter is the mode number.
 */
struct PLASK_API ModeWavelength: public MultiValueProperty<double> {
    static constexpr const char* NAME = "wavelength";
    static constexpr const char* UNIT = "nm";
};

/**
 * Modal loss [1/cm].
 *
 * Additional integer parameter is the mode number.
 */
struct PLASK_API ModeLoss: public MultiValueProperty<double> {
    static constexpr const char* NAME = "modal extinction";
    static constexpr const char* UNIT = "1/cm";
};

/**
 * Propagation constant [1/µm]. It can be either computed by some optical solvers or set by the user.
 *
 * It is a complex number, so it can contain information about both the propagation and losses.
 *
 * Additional integer parameter is the mode number.
 */
struct PLASK_API ModePropagationConstant: public MultiValueProperty<dcomplex> {
    static constexpr const char* NAME = "propagation constant";
    static constexpr const char* UNIT = "1/µm";
};

/**
 * Effective index. It can be either computed by some optical solvers or set by the user.
 *
 * It is a complex number, so it can contain information about both the propagation and losses.
 *
 * Additional integer parameter is the mode number.
 */
struct PLASK_API ModeEffectiveIndex: public MultiValueProperty<dcomplex> {
    static constexpr const char* NAME = "effective index";
    static constexpr const char* UNIT = "-";
};

} // namespace plask

#endif // PLASK__OPTICAL_H