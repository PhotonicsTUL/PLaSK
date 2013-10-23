#include "reflection_solver_2d.h"
#include "expansion_pw2d.h"

namespace plask { namespace solvers { namespace slab {

FourierReflection2D::FourierReflection2D(const std::string& name): ReflectionSolver<Geometry2DCartesian>(name),
    size(12),
    expansion(this),
    refine(8),
    outNeff(this, &FourierReflection2D::getEffectiveIndex, &FourierReflection2D::nummodes)
{
    detlog.global_prefix = this->getId();
    this->emitting = false;
}


void FourierReflection2D::loadConfiguration(XMLReader& reader, Manager& manager)
{
    while (reader.requireTagOrEnd()) {
        std::string param = reader.getNodeName();
        if (param == "expansion") {
            size = reader.getAttribute<double>("size", size);
            reader.requireTagEnd();
        } else
            parseStandardConfiguration(reader, manager, "TODO");
    }
}



void FourierReflection2D::onInitialize()
{
    setupLayers();
    expansion.init();
    diagonalizer.reset(new SimpleDiagonalizer(&expansion));    //TODO add other diagonalizer types
    init();
}


void FourierReflection2D::onInvalidate()
{
    modes.clear();
    expansion.free();
    diagonalizer.reset();
}


size_t FourierReflection2D::findMode(dcomplex neff)
{
    if (expansion.polarization != ExpansionPW2D::TEM)
        throw Exception("%1%: Cannot search for effective index with polarization separation", getId());
    klong = neff * k0;
    if (klong == 0.) klong = 1e-12;
    initCalculation();
    detlog.axis_arg_name = "neff";
    klong =  k0 *
        RootDigger(*this, [this](const dcomplex& x) { this->klong = x * this->k0; return this->determinant(); },
                   detlog, root)(neff);
    return insertMode();
}


DataVector<const Tensor3<dcomplex>> FourierReflection2D::getRefractiveIndexProfile(const RectilinearMesh2D& dst_mesh,
                                                                                   InterpolationMethod interp)
{
    initCalculation();
    std::map<size_t,DataVector<const Tensor3<dcomplex>>> cache;
    DataVector<Tensor3<dcomplex>> result(dst_mesh.size());
    for (size_t y = 0; y != dst_mesh.axis1.size(); ++y) {
        double h = dst_mesh.axis1[y];
        size_t n = getLayerFor(h);
        size_t l = stack[n];
        if (cache.find(l) == cache.end()) {
            cache[l] = expansion.getMaterialNR(l, dst_mesh.axis0, interp);
        }
        for (size_t x = 0; x != dst_mesh.axis0.size(); ++x) {
            result[dst_mesh.index(x,y)] = cache[l][x];
        }
    }
    return result;
}


std::vector<Vec<3,dcomplex>> FourierReflection2D::getReflectedAmplitudes(Polarization polarization)
{
}


double FourierReflection2D::getReflection(Polarization polarization)
{
}








const DataVector<const double> FourierReflection2D::getIntensity(size_t num, const MeshD<2>& dst_mesh, InterpolationMethod method)
{
//     if (!outSingleValue.hasValue())  // this is one possible indication that the solver is invalidated
//         throw NoValue(SomeSingleValueProperty::NAME);
//     return interpolate(*mesh, my_data, dst_mesh, defInterpolation<INTERPOLATION_LINEAR>(method)); // interpolate your data to the requested mesh
    return DataVector<const double>(dst_mesh.size(), 0.);
}


}}} // namespace
