#include <memory>
#include <vector>

namespace plast {

class TemperatureProvider {

	std::vector<TemperatureReciver*> recivers;

	public:

	std::shared_ptr< std::vector<double> > temperatures;

	TemperatureProvider(): temperatures(new std::vector<double>) {}

	std::vector<double>& getTemperatures() { return *temperatures; }

	virtual std::shared_ptr< const std::vector<double> > getTemperatures(Grid& grid, InterpolationMethod method);

	void fireTemperatureChanged();

};

struct TemperatureReciver {

	virtual void onTemperatureChanged(TemperatureProvider& changed_provider) {}

};

}	//namespace plast
