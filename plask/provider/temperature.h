
#include "provider.h"

namespace plask {

/**
TODO maybe:
struct Temperature {
	typedef double type;
	static const bool isNumeric = true;
}

Provider<Distribution<Temperature>> temp;	//on grid temp. provider or:
DistProvider<Temperature> temp;

Provider<Temperature> temp;	//for one value provider
*/

/**
Provides temperatures in all space.
*/
template <ModuleType>
class TemperatureProvider: public OnMeshInterpolatedProvider<ModuleType, double> {

	TemperatureProvider(
	  ModuleType* module,
	  typename OnMeshInterpolatedProvider<ModuleType, double>::MethodPtr module_value_get_method
	)
	: OnMeshInterpolatedProvider<ModuleType, double>(module, module_value_get_method) {}

};

/**
Recive (and typically use in calculation) temperature.
*/
struct TemperatureReciver: OnMeshInterpolatedReciver<TemperatureProvider> {};

} // namespace plask
