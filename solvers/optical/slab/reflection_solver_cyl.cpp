#include "reflection_solver_cyl.h"

namespace plask { namespace solvers { namespace slab {

FourierReflectionCyl::FourierReflectionCyl(const std::string& name): ModalSolver<Geometry2DCylindrical>(name),
    order(5),
    refine(8)
{
}


void FourierReflectionCyl::loadConfiguration(XMLReader& reader, Manager& manager)
{
    while (reader.requireTagOrEnd()) {
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
            parseStandardConfiguration(reader, manager, "<geometry>, <mesh>, <newton>, or <wavelength>");
    }
}



void FourierReflectionCyl::onInitialize()
{
    setupLayers();
}
//
//
// void FourierReflectionCyl::onInvalidate() // This will be called when e.g. geometry or mesh changes and your results become outdated
// {
//     outSingleValue.invalidate(); // clear the value
//     my_data.reset();
//     // Make sure that no provider returns any value.
//     // If this method has been called, before next computations, onInitialize will be called.
// }


double FourierReflectionCyl::computeMode(dcomplex neff) {
    initCalculation();
    return NAN;
}


const DataVector<const double> FourierReflectionCyl::getIntensity(const MeshD<2>& dst_mesh, int num, InterpolationMethod method) {
//     if (!outSingleValue.hasValue())  // this is one possible indication that the solver is invalidated
//         throw NoValue(SomeSingleValueProperty::NAME);
//     return interpolate(*mesh, my_data, dst_mesh, defInterpolation<INTERPOLATION_LINEAR>(method)); // interpolate your data to the requested mesh
    return DataVector<const double>(dst_mesh.size(), 0.);
}


}}} // namespace
