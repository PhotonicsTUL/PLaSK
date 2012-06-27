#include "your_module.h"

namespace plask { namespace modules { namespace your_module {

void YourModule::compute(double parameter) {
    // The code of this method probably should be in cpp file...
    // But below we show some key elements
    initCalculation(); // This must be called before any calculation!
    writelog(LOG_INFO, "Begining calculation of something");
    auto temperature = inTemperature(*mesh); // Obtain temperature from some other module
    // [...] Do your computations here
    outSingleValue = new_computed_value;
    writelog(LOG_RESULT, "Found new value of something = $1$", new_computed_value);
    outSingleValue.fireChanged(); // Inform other modules that you have computed a new value
    outSomeField.fireChanged();
}

}}} // namespace
