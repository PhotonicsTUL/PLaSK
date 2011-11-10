
#include "provider.h"

namespace plask {

/**
Provides temperatures in all space.
*/
template <ModuleType>
class TemperatureProvider: public OnGridInterpolatedProvider<ModuleType, double> {

	TemperatureProvider(
	  ModuleType* module,
	  typename OnGridInterpolatedProvider<ModuleType, double>::Method_Ptr module_value_get_method
	)
	: OnGridInterpolatedProvider<ModuleType, double>(module, module_value_get_method) {}

};

/**
Recive (and typically use in calculation) temperature.
*/
struct TemperatureReciver: OnGridInterpolatedReciver<TemperatureProvider> {};

} // namespace plask
