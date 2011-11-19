
#include "provider.h"

namespace plask {

struct Temperature: ScalarFieldProperty {};

/**
 * Provides temperature fields (temperature in points describe by given mesh).
 */
typedef ProviderFor<Temperature> TemperatureProvider;

/**
 * Recive temperature fields (temperature in points describe by given mesh).
 */
typedef ReceiverFor<Temperature> TemperatureReceiver;

} // namespace plask
