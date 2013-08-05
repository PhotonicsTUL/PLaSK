#include "reflection_solver.h"

namespace plask { namespace solvers { namespace modal {

FourierReflection2D::FourierReflection2D(const std::string& name): SolverOver<Geometry2DCartesian>(name),
    order(5),
    refine(8),
    outIntensity(this, &FourierReflection2D::getIntensity)
{
    inTemperature = 300.; // temperature receiver has some sensible value
}


void FourierReflection2D::loadConfiguration(XMLReader& reader, Manager& manager)
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


void FourierReflection2D::setLayerMesh()
{
    auto mesh = RectilinearMesh2DSimpleGenerator()(geometry->getChild());
    layers = mesh->axis1;
    //TODO consider geometry objects non-uniform in vertical direction (step approximation)
}







// void FourierReflection2D::onInitialize() // In this function check if geometry and mesh are set
// {
//     if (!geometry) throw NoGeometryException(getId());
//     if (!mesh) throw NoMeshException(getId());
//     my_data.reset(mesh->size()); // and e.g. allocate memory
// }
//
//
// void FourierReflection2D::onInvalidate() // This will be called when e.g. geometry or mesh changes and your results become outdated
// {
//     outSingleValue.invalidate(); // clear the value
//     my_data.reset();
//     // Make sure that no provider returns any value.
//     // If this method has been called, before next computations, onInitialize will be called.
// }




const DataVector<const double> FourierReflection2D::getIntensity(const MeshD<2>& dst_mesh, InterpolationMethod method) {
//     if (!outSingleValue.hasValue())  // this is one possible indication that the solver is invalidated
//         throw NoValue(SomeSingleValueProperty::NAME);
//     return interpolate(*mesh, my_data, dst_mesh, defInterpolation<INTERPOLATION_LINEAR>(method)); // interpolate your data to the requested mesh
}


}}} // namespace
