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
    if (expansion.polarization != ExpansionPW2D::E_UNSPECIFIED)
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


cvector FourierReflection2D::getReflectedAmplitudes(ExpansionPW2D::Component polarization,
                                                    IncidentDirection direction, size_t* savidx)
{
    if (!expansion.initialized && klong == 0.)
        expansion.polarization = polarization;
    emitting = true;
    fields_determined = false;

    initCalculation();
    diagonalizer->initDiagonalization(k0, klong, ktran);

    size_t idx;
    if (polarization == ExpansionPW2D::E_UNSPECIFIED)
        throw BadInput(getId(), "Wrong incident polarization specified for reflectivity computation");
    if (expansion.symmetric) {
        if (expansion.symmetry == ExpansionPW2D::E_UNSPECIFIED)
            expansion.symmetry = polarization;
        else if (expansion.symmetry != polarization)
            throw BadInput(getId(), "Current symmetry is inconsistent with specified incident polarization");
    }
    if (expansion.separated) {
        expansion.polarization = polarization;
        idx = expansion.iE(0);
    } else {
        idx = (polarization == ExpansionPW2D::E_TRAN)? expansion.iEx(0) : expansion.iEz(0);
    }
    if (savidx) *savidx = idx;

    cvector incident(expansion.matrixSize(), 0.);
    incident[idx] = 1.;

    return getReflectionVector(incident, direction);
}


double FourierReflection2D::getReflection(ExpansionPW2D::Component polarization, IncidentDirection direction)
{
    // if (!expansion.periodic)
    //     throw NotImplemented(getId(), "Reflection coefficient can be computed only for periodic geometries");

    size_t idx;
    cvector reflected = getReflectedAmplitudes(polarization, direction, &idx).claim();

    size_t n = (direction==DIRECTION_BOTTOM)? 0 : stack.size()-1;
    size_t l = stack[n];
    // if (!expansion.diagonalQE(l))
    //     throw Exception("%1%: %2% layer must be uniform to compute reflection coefficient",
    //                     getId(), (direction==DIRECTION_BOTTOM)? "Bottom" : "Top");

    auto gamma = diagonalizer->Gamma(l);
    dcomplex gamma0 = gamma[idx];
    for (size_t i = 0; i != expansion.matrixSize(); ++i) {
        reflected[i] = reflected[i] * conj(reflected[i]) * gamma[i] / gamma0;
    }

    double result = 0.;
    int N = getSize();
    if (expansion.separated) {
        for (int i = -N; i <= N; ++i)
            result += real(reflected[expansion.iE(i)]);
    } else {
        for (int i = -N; i <= N; ++i) {
            result += real(reflected[expansion.iEx(i)]) + real(reflected[expansion.iEz(i)]);
        }
    }

    return result;
}


const DataVector<const double> FourierReflection2D::getIntensity(size_t num, const MeshD<2>& dst_mesh, InterpolationMethod method)
{
//     if (!outSingleValue.hasValue())  // this is one possible indication that the solver is invalidated
//         throw NoValue(SomeSingleValueProperty::NAME);
//     return interpolate(*mesh, my_data, dst_mesh, defInterpolation<INTERPOLATION_LINEAR>(method)); // interpolate your data to the requested mesh
    return DataVector<const double>(dst_mesh.size(), 0.);
}


}}} // namespace
