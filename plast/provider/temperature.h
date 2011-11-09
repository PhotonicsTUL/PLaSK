#include <memory>
#include <vector>

namespace plast {

/**
Provides temperatures in all space.
*/
class TemperatureProvider {

	std::vector<TemperatureReciver*> recivers;

	public:

	std::shared_ptr< std::vector<double> > temperatures;

	TemperatureProvider(): temperatures(new std::vector<double>) {}

	std::vector<double>& getTemperatures() { return *temperatures; }

	virtual std::shared_ptr< const std::vector<double> > getTemperatures(Grid& grid, InterpolationMethod method);

	/**
	Call onTemperatureChanged for all recivers.
	Should be call by subclass after recalculation of temperatures.
	*/
	void fireTemperatureChanged();

};

/**
Recive (and typically use in calculation) temperature.
*/
struct TemperatureReciver {

	TemperatureProvider* provider;

	/**
	This methos is called on temperatures provided by provider changed. By default do nothing (subclass can use overwrite this).
	*/
	virtual void onTemperatureChanged(TemperatureProvider& changed_provider) {}

	/**
	Get termperatures from provider.
	@param grid set of points for which we need temperatures
	@param mehod interpolation method
	*/	
	virtual std::shared_ptr< const std::vector<double> > getTemperatures(Grid& grid, InterpolationMethod method) throw (NoProvider) {
	    if (!provider) throw NoProvider("temperature");
	    return provider->getTemperatures(grid, method);
	}
	
	void setProvider(TemperatureProvider* provider) {
	}

};

}	//namespace plast
