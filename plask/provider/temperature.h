
#include "provider.h"

namespace plask {

/**
 * Physical property tag class for temperature.
 */
struct Temperature: ScalarFieldProperty {};

/**
 * Provides temperature fields (temperature in points describe by given mesh).
 */
typedef ProviderFor<Temperature> TemperatureProvider;

/**
 * Receive temperature fields (temperature in points describe by given mesh).
 */
typedef ReceiverFor<Temperature> TemperatureReceiver;

} // namespace plask
