
#include "provider.h"

namespace plask {

/**
TODO maybe:
struct Temperature {
	typedef double type;
	static const bool isNumeric = true;
        static const bool isField = true;
}

struct Temperature : ScalarDistribution {};
struct Temperature : Distribution<double> {};


typedef Provider<Temperature> TemperatureProvider;	//on grid temp. provider or:


Provider<Temperature> temp;	//for one value provider

//Or:
struct Tempereture: ScalarField {}
Provider<Temperature> temp;
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
struct TemperatureReceiver: OnMeshInterpolatedReceiver<TemperatureProvider> {};

} // namespace plask
