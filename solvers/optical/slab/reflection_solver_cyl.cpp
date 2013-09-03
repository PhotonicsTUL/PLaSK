#include "reflection_solver_cyl.h"

namespace plask { namespace solvers { namespace slab {

FourierReflectionCyl::FourierReflectionCyl(const std::string& name): SlabSolver<Geometry2DCylindrical>(name),
    size(5),
    refine(8)
{
}


void FourierReflectionCyl::loadConfiguration(XMLReader& reader, Manager& manager)
{
    while (reader.requireTagOrEnd()) {
            parseStandardConfiguration(reader, manager, "<geometry>, <mesh>, <newton>, or <wavelength>");
    }
}



void FourierReflectionCyl::onInitialize()
{
    setupLayers();
}



size_t FourierReflectionCyl::findMode(dcomplex neff) {
    initCalculation();
    return 0;
}


const DataVector<const double> FourierReflectionCyl::getIntensity(size_t num, const MeshD<2>& dst_mesh, InterpolationMethod method) {
    return DataVector<const double>(dst_mesh.size(), 0.);
}


}}} // namespace
