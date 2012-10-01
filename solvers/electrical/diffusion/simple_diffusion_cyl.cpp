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
        reader.requireTagEnd();
    } else if (param == "wavelength") {
        std::string = reader.requireTextUntilEnd();
        inWavelength.setValue(boost::lexical_cast<double>(wavelength));
    } else
        throw XMLUnexpectedElementException(reader, "<geometry>, <mesh>, <newton>, or <wavelength>", param);
*/}




std::deque<Box2D> SimpleDiffusionSolverCyl::detectQuantumWells()
{
    shared_ptr<RectilinearMesh2D> mesh = RectilinearMesh2DSimpleGenerator()(geometry->getChild());
    shared_ptr<RectilinearMesh2D> points = mesh->getMidpointsMesh();

    std::deque<Box2D> results;

    // Now compact each row (it can contain only one QW and each must start and end in the same point)
    double left, right;
    bool foundQW = false;
    for (int j = 0; j < points->axis1.size(); ++j) {
        bool inQW = false;
        for (int i = 0; i < points->axis0.size(); ++i) {
            auto point = points->at(i,j);
            auto tags = geometry->getRolesAt(point);
            bool QW = tags.find("QW") != tags.end() || tags.find("QD") != tags.end();
            if (QW && !inQW) { // QW start
                if (foundQW && left != mesh->axis0[i])
                    throw Exception("This solver can only handle quantum wells of identical size located exactly one above another");
                left = mesh->axis0[i];
                inQW = true;
            }
            if (!QW && inQW) { // QW end
                if (foundQW && right != mesh->axis0[i])
                    throw Exception("This solver can only handle quantum wells of identical size located exactly one above another");
                right = mesh->axis0[i];
                results.push_back(Box2D(left, mesh->axis1[j], right, mesh->axis1[j+1]));
                foundQW = true;
                inQW = false;
            }
        }
        if (inQW) { // handle situation when QW spans to the end of the structure
            if (foundQW && right != mesh->axis0[points->axis0.size()])
                throw Exception("This solver can only handle quantum wells of identical size located exactly one above another");
            right = mesh->axis0[points->axis0.size()];
            results.push_back(Box2D(left, mesh->axis1[j], right, mesh->axis1[j+1]));
            foundQW = true;
            inQW = false;
        }
    }

    // Compact results in vertical direction
    //TODO

    return results;
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
