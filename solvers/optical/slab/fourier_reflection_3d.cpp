#include "fourier_reflection_3d.h"
#include "expansion_pw3d.h"

namespace plask { namespace solvers { namespace slab {

FourierReflection3D::FourierReflection3D(const std::string& name): ReflectionSolver<Geometry3D>(name),
    size_long(12), size_tran(12),
    expansion(this),
    refine_long(16), refine_tran(16)//,
//     outNeff(this, &FourierReflection3D::getEffectiveIndex, &FourierReflection3D::nummodes)
{
    smooth = 0.05;
    // detlog.global_prefix = this->getId();
}

static inline PML readPML(XMLReader& reader) {
    PML pml;
    pml.factor = reader.getAttribute<dcomplex>("factor", pml.factor);
    pml.size = reader.getAttribute<double>("size", pml.size);
    pml.shift = reader.getAttribute<double>("shift", pml.shift);
    pml.order = reader.getAttribute<double>("order", pml.order);
    return pml;
}

template <typename T>
static inline void readComaAttr(XMLReader& reader, const std::string& attr, T& long_field, T& tran_field) {
    if (reader.hasAttribute(attr)) {
        std::string value = reader.requireAttribute<std::string>(attr);
        if (value.find(',') == std::string::npos)
            long_field = tran_field = boost::lexical_cast<T>(value);
        else {
            auto values = splitString2(value, ',');
            long_field = boost::lexical_cast<T>(values.first);
            tran_field = boost::lexical_cast<T>(values.first);
        }
        if (reader.hasAttribute(attr+"-long")) throw XMLConflictingAttributesException(reader, attr, attr+"-long");
        if (reader.hasAttribute(attr+"-tran")) throw XMLConflictingAttributesException(reader, attr, attr+"-tran");
    } else {
        long_field = reader.getAttribute<T>(attr+"-long", long_field);
        tran_field = reader.getAttribute<T>(attr+"-tran", tran_field);
    }
}


void FourierReflection3D::loadConfiguration(XMLReader& reader, Manager& manager)
{
    while (reader.requireTagOrEnd()) {
        std::string param = reader.getNodeName();
        if (param == "expansion") {
            readComaAttr(reader, "size", size_long, size_tran);
            readComaAttr(reader, "refine", refine_long, refine_tran);
            smooth = reader.getAttribute<double>("smooth", smooth);
            reader.requireTagEnd();
        } else if (param == "interface") {
            if (reader.hasAttribute("index")) {
                if (reader.hasAttribute("position")) throw XMLConflictingAttributesException(reader, "index", "position");
                if (reader.hasAttribute("object")) throw XMLConflictingAttributesException(reader, "index", "object");
                if (reader.hasAttribute("path")) throw XMLConflictingAttributesException(reader, "index", "path");
                setInterface(reader.requireAttribute<size_t>("interface"));
            } else if (reader.hasAttribute("position")) {
                if (reader.hasAttribute("object")) throw XMLConflictingAttributesException(reader, "index", "object");
                if (reader.hasAttribute("path")) throw XMLConflictingAttributesException(reader, "index", "path");
                setInterfaceAt(reader.requireAttribute<double>("interface"));
            } else if (reader.hasAttribute("object")) {
                auto object = manager.requireGeometryObject<GeometryObjectD<2>>(reader.requireAttribute("object"));
                PathHints path; if (auto pathattr = reader.getAttribute("path")) path = manager.requirePathHints(*pathattr);
                setInterfaceOn(object, path);
            } else if (reader.hasAttribute("path")) {
                throw XMLUnexpectedAttrException(reader, "path");
            }
            reader.requireTagEnd();
        } else if (param == "pmls") {
            pml_long = pml_tran = readPML(reader);
            while (reader.requireTagOrEnd()) {
                std::string node = reader.getNodeName();
                if (node == "long") {
                    pml_long = readPML(reader);
                } else if (node == "tran") {
                    pml_tran = readPML(reader);
                } else throw XMLUnexpectedElementException(reader, "<tran>, <long>, or </pmls>", node);
            }
        } else if (param == "mode") {
            k0 = 2e3*M_PI / reader.getAttribute<dcomplex>("wavelength", 2e3*M_PI / k0);
            ktran = reader.getAttribute<dcomplex>("k-tran", ktran);
            klong = reader.getAttribute<dcomplex>("k-long", klong);
//             if (reader.hasAttribute("symmetry")) {
//                 std::string repr = reader.requireAttribute("symmetry");
//                 ExpansionPW2D::Component val;
//                 AxisNames* axes = nullptr;
//                 if (geometry) axes = &geometry->axisNames;
//                 if (repr == "none" || repr == "NONE" || repr == "None")
//                     val = ExpansionPW3D::E_UNSPECIFIED;
//                 else if (repr == "Etran" || repr == "Et" || (axes && repr == "E"+axes->getNameForTran()) ||
//                          repr == "Hlong" || repr == "Hl" || (axes && repr == "H"+axes->getNameForLong()))
//                     val = ExpansionPW3D::E_TRAN;
//                 else if (repr == "Elong" || repr == "El" || (axes && repr == "E"+axes->getNameForLong()) ||
//                          repr == "Htran" || repr == "Ht" || (axes && repr == "H"+axes->getNameForTran()))
//                     val = ExpansionPW3D::E_LONG;
//                 else
//                     throw XMLBadAttrException(reader, "symmetry", repr, "symmetric field component name (maybe you need to specify the geometry first)");
//                 setSymmetry(val);
//             }
        } else if (param == "root") {
            readRootDiggerConfig(reader);
        } else if (param == "outer") {
            outdist = reader.requireAttribute<double>("dist");
            reader.requireTagEnd();
        } else
            parseStandardConfiguration(reader, manager);
    }
}



void FourierReflection3D::onInitialize()
{
    setupLayers();
    expansion.init();
    diagonalizer.reset(new SimpleDiagonalizer(&expansion));    //TODO add other diagonalizer types
    init();
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
const DataVector<const Vec<3,dcomplex>> FourierReflection3D::getE(size_t num, shared_ptr<const MeshD<3>> dst_mesh, InterpolationMethod method)
{
    throw NotImplemented("FourierReflection3D::getE");
//     if (modes.size() <= num) throw NoValue(LightE::NAME);
//     if (modes[num].k0 != k0 || modes[num].klong != klong || modes[num].ktran != ktran) {
//         k0 = modes[num].k0;
//         klong = modes[num].klong;
//         ktran = modes[num].ktran;
//         fields_determined = DETERMINED_NOTHING;
//     }
//     return getFieldE(dst_mesh, method);
}


const DataVector<const Vec<3,dcomplex>> FourierReflection3D::getH(size_t num, shared_ptr<const MeshD<3>> dst_mesh, InterpolationMethod method)
{
    throw NotImplemented("FourierReflection3D::getH");
//     if (modes.size() <= num) throw NoValue(LightH::NAME);
//     if (modes[num].k0 != k0 || modes[num].klong != klong || modes[num].ktran != ktran) {
//         k0 = modes[num].k0;
//         klong = modes[num].klong;
//         ktran = modes[num].ktran;
//         fields_determined = DETERMINED_NOTHING;
//     }
//     return getFieldH(dst_mesh, method);
}


const DataVector<const double> FourierReflection3D::getIntensity(size_t num, shared_ptr<const MeshD<3> > dst_mesh, InterpolationMethod method)
{
    throw NotImplemented("FourierReflection3D::getIntensity");
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
