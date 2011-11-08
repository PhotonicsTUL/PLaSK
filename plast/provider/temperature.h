#include <memory>
#include <vector>


class TemperatureProvider {

	std::vector<TemperatureReciver*> recivers;

	std::shared_ptr< std::vector<double> > temperatures;

	public:

	TemperatureProvider(): temperatures(new std::vector<double>) {}

	std::vector<double> getTemperatures() { return *temperatures; }

	virtual void getTemperature(Grid& grid, Dest<double>& dest, InterpolationMethod method) = 0;

	virtual bool isTemperatureUseGrid(Grid& grid);

	std::shared_ptr< std::vector<double> > getTemperature(Grid& grid);

	void fireTemperatureChanged();

};

struct TemperatureReciver {

	virtual void onTemperatureChanged(TemperatureProvider& changed_provider) {}

};


