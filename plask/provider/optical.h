#ifndef PLASK__OPTICAL_H
#define PLASK__OPTICAL_H

#include "../math.h"

namespace plask {

/**
 * \brief Intensity of optical E field.
 *
 * Intensity of optical electric (E) field. It is calculated as abs(E), providing that E is electric field vector.
 *
 * This property should be provided by every optical module as it has a nice advantage that does not depend on
 * the internal representation of field (whether it is scalar or vectorial one).
 */
struct OpticalIntensity : public ScalarFieldProperty {};

} // namespace plask

#endif // PLASK__OPTICAL_H