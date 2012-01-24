#ifndef PLASK__TEMPERATURE_H
#define PLASK__TEMPERATURE_H

#include "provider.h"

namespace plask {

/**
 * Physical property tag class for temperature.
 */
struct Temperature: ScalarFieldProperty {};

//TODO in gcc 4.7 can be done by new typedefs:

/*
 * Provides temperature fields (temperature in points describe by given mesh).
 */
//typedef ProviderFor<Temperature> TemperatureProvider;

/*
 * Receive temperature fields (temperature in points describe by given mesh).
 */
//typedef ReceiverFor<Temperature> TemperatureReceiver;

} // namespace plask

#endif // PLASK__TEMPERATURE_H
