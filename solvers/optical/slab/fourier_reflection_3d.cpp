#include "fourier_reflection_3d.h"
#include "expansion_pw3d.h"

namespace plask { namespace solvers { namespace slab {

FourierReflection3D::FourierReflection3D(const std::string& name): ReflectionSolver<Geometry3D>(name),
    size_long(11), size_tran(11),
    expansion(this),
    refine_long(16), refine_tran(16)//,
//     outNeff(this, &FourierReflection3D::getEffectiveIndex, &FourierReflection3D::nummodes)
{
//     detlog.global_prefix = this->getId();
}
//
//
// void FourierReflection3D::loadConfiguration(XMLReader& reader, Manager& manager)
// {
//     while (reader.requireTagOrEnd()) {
//         std::string param = reader.getNodeName();
//         if (param == "expansion") {
//             size = reader.getAttribute<double>("size", size);
//             reader.requireTagEnd();
//         } else
//             parseStandardConfiguration(reader, manager, "TODO");
//     }
// }
//
//
//
void FourierReflection3D::onInitialize()
{
    setupLayers();
    expansion.init();
    diagonalizer.reset(new SimpleDiagonalizer(&expansion));    //TODO add other diagonalizer types
    init();
    expansion.computeMaterialCoefficients();
}


void FourierReflection3D::onInvalidate()
{
    cleanup();
    modes.clear();
    expansion.free();
    diagonalizer.reset();
    fields.clear();
}
//
//
// size_t FourierReflection3D::findMode(dcomplex neff)
// {
//     if (expansion.polarization != ExpansionPW3D::E_UNSPECIFIED)
//         throw Exception("%1%: Cannot search for effective index with polarization separation", getId());
//     klong = neff * k0;
//     if (klong == 0.) klong = 1e-12;
//     initCalculation();
//     detlog.axis_arg_name = "neff";
//     klong =  k0 *
//         RootDigger(*this, [this](const dcomplex& x) { this->klong = x * this->k0; return this->determinant(); },
//                    detlog, root)(neff);
//     return insertMode();
// }


DataVector<const Tensor3<dcomplex>> FourierReflection3D::getRefractiveIndexProfile(const RectilinearMesh3D& dst_mesh,
                                                                                   InterpolationMethod interp)
{
    initCalculation();
    std::map<size_t,DataVector<const Tensor3<dcomplex>>> cache;
    DataVector<Tensor3<dcomplex>> result(dst_mesh.size());
    for (size_t z = 0; z != dst_mesh.axis2.size(); ++z) {
        double h = dst_mesh.axis2[z];
        size_t n = getLayerFor(h);
        size_t l = stack[n];
        if (cache.find(l) == cache.end()) {
            cache[l] = expansion.getMaterialNR(l, dst_mesh.axis0, dst_mesh.axis1, interp);
        }
        for (size_t y = 0; y != dst_mesh.axis1.size(); ++y) {
            size_t offset = y * dst_mesh.axis0.size();
            for (size_t x = 0; x != dst_mesh.axis0.size(); ++x) {
                result[dst_mesh.index(x,y,z)] = cache[l][offset+x];
            }
        }
    }
    return result;
}


// cvector FourierReflection3D::getReflectedAmplitudes(ExpansionPW3D::Component polarization,
//                                                     IncidentDirection incidence, size_t* savidx)
// {
//     if (!expansion.initialized && klong == 0.) expansion.polarization = polarization;
//     emitting = true;
//     fields_determined = DETERMINED_NOTHING;
//     initCalculation();
//     return getReflectionVector(incidentVector(polarization, savidx), incidence);
// }
//
//
// cvector FourierReflection3D::getTransmittedAmplitudes(ExpansionPW3D::Component polarization,
//                                                       IncidentDirection incidence, size_t* savidx)
// {
//     if (!expansion.initialized && klong == 0.) expansion.polarization = polarization;
//     emitting = true;
//     fields_determined = DETERMINED_NOTHING;
//     initCalculation();
//     return getTransmissionVector(incidentVector(polarization, savidx), incidence);
// }
//
//
// double FourierReflection3D::getReflection(ExpansionPW3D::Component polarization, IncidentDirection incidence)
// {
//     size_t idx;
//     cvector reflected = getReflectedAmplitudes(polarization, incidence, &idx).claim();
//
//     if (!expansion.periodic)
//         throw NotImplemented(getId(), "Reflection coefficient can be computed only for periodic geometries");
//
//     size_t n = (incidence == INCIDENCE_BOTTOM)? 0 : stack.size()-1;
//     size_t l = stack[n];
//     if (!expansion.diagonalQE(l))
//         writelog(LOG_WARNING, "%1% layer should be uniform to reliably compute reflection coefficient",
//                               (incidence == INCIDENCE_BOTTOM)? "Bottom" : "Top");
//
//     auto gamma = diagonalizer->Gamma(l);
//     dcomplex gamma0 = gamma[idx];
//     dcomplex igamma0 = 1. / gamma0;
//     if (!expansion.separated) {
//         int N = getSize() + 1;
//         for (int n = expansion.symmetric? 0 : 1-N; n != N; ++n) {
//             size_t iz = expansion.iEz(n), ix = expansion.iEx(n);
//             reflected[ix] = reflected[ix] * conj(reflected[ix]) *  gamma0 / gamma[ix];
//             reflected[iz] = reflected[iz] * conj(reflected[iz]) * igamma0 * gamma[iz];
//         }
//     } else {
//         if (expansion.polarization == ExpansionPW3D::E_LONG) {
//             for (size_t i = 0; i != expansion.matrixSize(); ++i)
//                 reflected[i] = reflected[i] * conj(reflected[i]) * igamma0 * gamma[i];
//         } else {
//             for (size_t i = 0; i != expansion.matrixSize(); ++i)
//                 reflected[i] = reflected[i] * conj(reflected[i]) *  gamma0 / gamma[i];
//         }
//     }
//
//
//     return sumAmplitutes(reflected);
// }
//
//
// double FourierReflection3D::getTransmission(ExpansionPW3D::Component polarization, IncidentDirection incidence)
// {
//     size_t idx;
//     cvector transmitted = getTransmittedAmplitudes(polarization, incidence, &idx).claim();
//
//     if (!expansion.periodic)
//         throw NotImplemented(getId(), "Transmission coefficient can be computed only for periodic geometries");
//
//     size_t ni = (incidence == INCIDENCE_TOP)? stack.size()-1 : 0;
//     size_t nt = stack.size()-1-ni;
//     size_t li = stack[ni], lt = stack[nt];
//     if (!expansion.diagonalQE(lt))
//         writelog(LOG_WARNING, "%1% layer should be uniform to reliably compute transmission coefficient",
//                  (incidence == INCIDENCE_TOP)? "Bottom" : "Top");
//     if (!expansion.diagonalQE(li))
//         writelog(LOG_WARNING, "%1% layer should be uniform to reliably compute transmission coefficient",
//                  (incidence == INCIDENCE_TOP)? "Top" : "Bottom");
//
//     auto gamma = diagonalizer->Gamma(lt);
//     dcomplex igamma0 = 1. / diagonalizer->Gamma(li)[idx];
//     dcomplex gamma0 = gamma[idx] * gamma[idx] * igamma0;
//     if (!expansion.separated) {
//         int N = getSize() + 1;
//         for (int n = expansion.symmetric? 0 : 1-N; n != N; ++n) {
//             size_t iz = expansion.iEz(n), ix = expansion.iEx(n);
//             transmitted[ix] = transmitted[ix] * conj(transmitted[ix]) *  gamma0 / gamma[ix];
//             transmitted[iz] = transmitted[iz] * conj(transmitted[iz]) * igamma0 * gamma[iz];
//         }
//     } else {
//         if (expansion.polarization == ExpansionPW3D::E_LONG) {
//             for (size_t i = 0; i != expansion.matrixSize(); ++i)
//                 transmitted[i] = transmitted[i] * conj(transmitted[i]) * igamma0 * gamma[i];
//         } else {
//             for (size_t i = 0; i != expansion.matrixSize(); ++i)
//                 transmitted[i] = transmitted[i] * conj(transmitted[i]) *  gamma0 / gamma[i];
//         }
//     }
//
//     return sumAmplitutes(transmitted);
// }
//
//
const DataVector<const Vec<3,dcomplex>> FourierReflection3D::getE(size_t num, const MeshD<3>& dst_mesh, InterpolationMethod method)
{
//     if (modes.size() <= num) throw NoValue(OpticalElectricField::NAME);
//     if (modes[num].k0 != k0 || modes[num].klong != klong || modes[num].ktran != ktran) {
//         k0 = modes[num].k0;
//         klong = modes[num].klong;
//         ktran = modes[num].ktran;
//         fields_determined = DETERMINED_NOTHING;
//     }
//     return getFieldE(dst_mesh, method);
}


const DataVector<const Vec<3,dcomplex>> FourierReflection3D::getH(size_t num, const MeshD<3>& dst_mesh, InterpolationMethod method)
{
//     if (modes.size() <= num) throw NoValue(OpticalMagneticField::NAME);
//     if (modes[num].k0 != k0 || modes[num].klong != klong || modes[num].ktran != ktran) {
//         k0 = modes[num].k0;
//         klong = modes[num].klong;
//         ktran = modes[num].ktran;
//         fields_determined = DETERMINED_NOTHING;
//     }
//     return getFieldH(dst_mesh, method);
}


const DataVector<const double> FourierReflection3D::getIntensity(size_t num, const MeshD<3>& dst_mesh, InterpolationMethod method)
{
//     if (modes.size() <= num) throw NoValue(LightMagnitude::NAME);
//     if (modes[num].k0 != k0 || modes[num].klong != klong || modes[num].ktran != ktran) {
//         k0 = modes[num].k0;
//         klong = modes[num].klong;
//         ktran = modes[num].ktran;
//         fields_determined = DETERMINED_NOTHING;
//     }
//     return getFieldIntensity(modes[num].power, dst_mesh, method);
}
//
//
}}} // namespace
