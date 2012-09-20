#include "simple_diffusion_cyl.h"

namespace plask { namespace solvers { namespace diffusion {

SimpleDiffusionSolverCyl::SimpleDiffusionSolverCyl(const std::string& name): SolverOver<Geometry2DCylindrical>(name),
    outCarrierDensity(this, &SimpleDiffusionSolverCyl::getCarrierDensity) // getCarrierDensity will be called whether provider value is requested
{
    inTemperature = 300.; // temperature receiver has some sensible value
}




void SimpleDiffusionSolverCyl::loadParam(const std::string& param, XMLReader& reader, Manager&)
{/*
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
*/}




std::vector<Box2D> SimpleDiffusionSolverCyl::detectQuantumWells() {
}





void SimpleDiffusionSolverCyl::compute(double parameter)
{/*
    // Below we show some key objects of the computational methods
    initCalculation(); // This must be called before any calculation!
    writelog(LOG_INFO, "Begining calculation of something");
    auto temperature = inTemperature(*mesh); // Obtain temperature from some other solver
    // [...] Do your computations here
    outSingleValue = new_computed_value;
    writelog(LOG_RESULT, "Found new value of something = $1$", new_computed_value);
    outSingleValue.fireChanged(); // Inform other solvers that you have computed a new value
    outSomeField.fireChanged();
*/}


void SimpleDiffusionSolverCyl::onInitialize() // In this function check if geometry and mesh are set
{
    if (!geometry) throw NoGeometryException(getId());
}


void SimpleDiffusionSolverCyl::onInvalidate() // This will be called when e.g. geometry or mesh changes and your results become outdated
{/*
    outSingleValue.invalidate(); // clear the value
    my_data.reset();
    // Make sure that no provider returns any value.
    // If this method has been called, before next computations, onInitialize will be called.
*/}




const DataVector<const double> SimpleDiffusionSolverCyl::getCarrierDensity(const MeshD<2>& dst_mesh, InterpolationMethod method)
{/*
    if (!outSingleValue.hasValue())  // this is one possible indication that the solver is invalidated
        throw NoValue(SomeSingleValueProperty::NAME);
    return interpolate(*mesh, my_data, dst_mesh, defInterpolation<INTERPOLATION_LINEAR>(method)); // interpolate your data to the requested mesh

*/}


}}} // namespace
