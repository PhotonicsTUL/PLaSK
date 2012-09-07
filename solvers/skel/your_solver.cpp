#include "your_solver.h"

namespace plask { namespace solvers { namespace your_solver {

void YourSolver::loadParam(const std::string& param, XMLReader& reader, Manager&) {
    // Load a configuration parameter from XML.
    // Below you have an example
    if (param == "newton") {
        newton.tolx = reader.getAttribute<double>("tolx", newton.tolx);
        newton.tolf = reader.getAttribute<double>("tolf", newton.tolf);
        newton.maxstep = reader.getAttribute<double>("maxstep", newton.maxstep);
    } else if (param == "wavelength") {
        std::string = reader.requireText();
        inWavelength.setValue(boost::lexical_cast<double>(wavelength));
    } else
        throw XMLUnexpectedElementException(reader, "<geometry>, <mesh>, <newton>, or <wavelength>", param);
    reader.requireTagEnd();
}


void YourSolver::compute(double parameter) {
    // Below we show some key elements of the computational methods
    initCalculation(); // This must be called before any calculation!
    writelog(LOG_INFO, "Begining calculation of something");
    auto temperature = inTemperature(*mesh); // Obtain temperature from some other solver
    // [...] Do your computations here
    outSingleValue = new_computed_value;
    writelog(LOG_RESULT, "Found new value of something = $1$", new_computed_value);
    outSingleValue.fireChanged(); // Inform other solvers that you have computed a new value
    outSomeField.fireChanged();
}

}}} // namespace
