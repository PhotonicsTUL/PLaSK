#include "reflection_solver.h"

namespace plask { namespace solvers { namespace vslab {

// ReflectionSolver::ReflectionSolver(const std::string& name=""): SolverWithMesh<ForExample_Geometry2DCartesian, ForExample_RectilinearMesh2D>(name),
//     outSomeField(this, &ReflectionSolver::getDelegated) // getDelegated will be called whether provider value is requested
// {
//     inTemperature = 300.; // temperature receiver has some sensible value
// }
//
//
// void ReflectionSolver::loadConfiguration(XMLReader& reader, Manager& manager) {
//     // Load a configuration parameter from XML.
//     // Below you have an example
//     while (reader.requireTagOrEnd()) {
//         std::string param = reader.getNodeName();
//         if (param == "newton") {
//             newton.tolx = reader.getAttribute<double>("tolx", newton.tolx);
//             newton.tolf = reader.getAttribute<double>("tolf", newton.tolf);
//             newton.maxstep = reader.getAttribute<double>("maxstep", newton.maxstep);
//             reader.requireTagEnd();
//         } else if (param == "wavelength") {
//             std::string = reader.requireTextUntilEnd();
//             inWavelength.setValue(boost::lexical_cast<double>(wavelength));
//         } else
//             parseStandardConfiguration(reader, manager, "<geometry>, <mesh>, <newton>, or <wavelength>");
//     }
// }
//
//
// void ReflectionSolver::compute(double parameter)
// {
//     // Below we show some key objects of the computational methods
//     initCalculation(); // This must be called before any calculation!
//     writelog(LOG_INFO, "Begining calculation of something");
//     auto temperature = inTemperature(*mesh); // Obtain temperature from some other solver
//     // [...] Do your computations here
//     outSingleValue = new_computed_value;
//     writelog(LOG_RESULT, "Found new value of something = $1$", new_computed_value);
//     outSingleValue.fireChanged(); // Inform other solvers that you have computed a new value
//     outSomeField.fireChanged();
// }
//
//
// void ReflectionSolver::onInitialize() // In this function check if geometry and mesh are set
// {
//     if (!geometry) throw NoGeometryException(getId());
//     if (!mesh) throw NoMeshException(getId());
//     my_data.reset(mesh->size()); // and e.g. allocate memory
// }
//
//
// void ReflectionSolver::onInvalidate() // This will be called when e.g. geometry or mesh changes and your results become outdated
// {
//     outSingleValue.invalidate(); // clear the value
//     my_data.reset();
//     // Make sure that no provider returns any value.
//     // If this method has been called, before next computations, onInitialize will be called.
// }
//
//
// const DataVector<const double> ReflectionSolver::getDelegated(const MeshD<2>& dst_mesh, InterpolationMethod method) {
//     if (!outSingleValue.hasValue())  // this is one possible indication that the solver is invalidated
//         throw NoValue(SomeSingleValueProperty::NAME);
//     return interpolate(*mesh, my_data, dst_mesh, defInterpolation<INTERPOLATION_LINEAR>(method)); // interpolate your data to the requested mesh
// }


}}} // namespace
