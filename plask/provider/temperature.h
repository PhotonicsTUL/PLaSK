
#include "provider.h"

namespace plask {

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
