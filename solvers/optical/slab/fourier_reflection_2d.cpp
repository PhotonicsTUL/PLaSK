#include "fourier_reflection_2d.h"
#include "expansion_pw2d.h"

namespace plask { namespace solvers { namespace slab {

FourierReflection2D::FourierReflection2D(const std::string& name): ReflectionSolver<Geometry2DCartesian>(name),
    size(12),
    expansion(this),
    refine(8),
    outNeff(this, &FourierReflection2D::getEffectiveIndex, &FourierReflection2D::nummodes)
{
    detlog.global_prefix = this->getId();
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
    expansion.computeMaterialCoefficients();
}


void FourierReflection2D::onInvalidate()
{
    cleanup();
    modes.clear();
    expansion.free();
    diagonalizer.reset();
    fields.clear();
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
                                                    IncidentDirection incidence, size_t* savidx)
{
    if (!expansion.initialized && klong == 0.) expansion.polarization = polarization;
    emitting = true;
    fields_determined = DETERMINED_NOTHING;
    initCalculation();
    return getReflectionVector(incidentVector(polarization, savidx), incidence);
}


cvector FourierReflection2D::getTransmittedAmplitudes(ExpansionPW2D::Component polarization,
                                                      IncidentDirection incidence, size_t* savidx)
{
    if (!expansion.initialized && klong == 0.) expansion.polarization = polarization;
    emitting = true;
    fields_determined = DETERMINED_NOTHING;
    initCalculation();
    return getTransmissionVector(incidentVector(polarization, savidx), incidence);
}


double FourierReflection2D::getReflection(ExpansionPW2D::Component polarization, IncidentDirection incidence)
{
    size_t idx;
    cvector reflected = getReflectedAmplitudes(polarization, incidence, &idx).claim();

    if (!expansion.periodic)
        throw NotImplemented(getId(), "Reflection coefficient can be computed only for periodic geometries");

    size_t n = (incidence == INCIDENCE_BOTTOM)? 0 : stack.size()-1;
    size_t l = stack[n];
    if (!expansion.diagonalQE(l))
        writelog(LOG_WARNING, "%1% layer should be uniform to reliably compute reflection coefficient",
                              (incidence == INCIDENCE_BOTTOM)? "Bottom" : "Top");

    auto gamma = diagonalizer->Gamma(l);
    dcomplex gamma0 = gamma[idx];
    for (size_t i = 0; i != expansion.matrixSize(); ++i) {
        reflected[i] = reflected[i] * conj(reflected[i]) * gamma[i] / gamma0;
    }

    return sumAmplitutes(reflected);
}


double FourierReflection2D::getTransmission(ExpansionPW2D::Component polarization, IncidentDirection incidence)
{
    size_t idx;
    cvector transmitted = getTransmittedAmplitudes(polarization, incidence, &idx).claim();

    if (!expansion.periodic)
        throw NotImplemented(getId(), "Transmission coefficient can be computed only for periodic geometries");

    size_t ni = (incidence == INCIDENCE_TOP)? stack.size()-1 : 0;
    size_t nt = stack.size()-1-ni;
    size_t li = stack[ni], lt = stack[nt];
    if (!expansion.diagonalQE(lt))
        writelog(LOG_WARNING, "%1% layer should be uniform to reliably compute transmission coefficient",
                 (incidence == INCIDENCE_TOP)? "Bottom" : "Top");
    if (!expansion.diagonalQE(li))
        writelog(LOG_WARNING, "%1% layer should be uniform to reliably compute transmission coefficient",
                 (incidence == INCIDENCE_TOP)? "Top" : "Bottom");

    auto gamma = diagonalizer->Gamma(lt);
    dcomplex gamma0 = diagonalizer->Gamma(li)[idx];
    for (size_t i = 0; i != expansion.matrixSize(); ++i) {
        transmitted[i] = transmitted[i] * conj(transmitted[i]) * gamma[i] / gamma0;
    }

    return sumAmplitutes(transmitted);
}


const DataVector<const Vec<3,dcomplex>> FourierReflection2D::getE(size_t num, const MeshD<2>& dst_mesh, InterpolationMethod method)
{
    if (modes.size() <= num) throw NoValue(OpticalElectricField::NAME);
    if (modes[num].k0 != k0 || modes[num].beta != klong || modes[num].ktran != ktran) {
        k0 = modes[num].k0;
        klong = modes[num].beta;
        ktran = modes[num].ktran;
        fields_determined = DETERMINED_NOTHING;
    }
    return getFieldE(dst_mesh, method);
}


const DataVector<const Vec<3,dcomplex>> FourierReflection2D::getH(size_t num, const MeshD<2>& dst_mesh, InterpolationMethod method)
{
    if (modes.size() <= num) throw NoValue(OpticalMagneticField::NAME);
    if (modes[num].k0 != k0 || modes[num].beta != klong || modes[num].ktran != ktran) {
        k0 = modes[num].k0;
        klong = modes[num].beta;
        ktran = modes[num].ktran;
        fields_determined = DETERMINED_NOTHING;
    }
    return getFieldH(dst_mesh, method);
}


const DataVector<const double> FourierReflection2D::getIntensity(size_t num, const MeshD<2>& dst_mesh, InterpolationMethod method)
{
    if (modes.size() <= num) throw NoValue(LightIntensity::NAME);
    if (modes[num].k0 != k0 || modes[num].beta != klong || modes[num].ktran != ktran) {
        k0 = modes[num].k0;
        klong = modes[num].beta;
        ktran = modes[num].ktran;
        fields_determined = DETERMINED_NOTHING;
    }
    return getFieldIntensity(modes[num].power, dst_mesh, method);
}


}}} // namespace
