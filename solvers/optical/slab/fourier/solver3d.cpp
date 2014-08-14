#include "solver3d.h"
#include "expansion3d.h"

namespace plask { namespace solvers { namespace slab {

FourierSolver3D::FourierSolver3D(const std::string& name): SlabSolver<Geometry3D>(name),
    size_long(12), size_tran(12),
    expansion(this),
    refine_long(16), refine_tran(16)//,
{
    detlog.global_prefix = this->getId();
    smooth = 0.005;
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
static inline void readComaAttr(XMLReader& reader, const std::string& attr, T& long_field, T& tran_field, bool nocommon=false) {
    if (reader.hasAttribute(attr)) {
        std::string value = reader.requireAttribute<std::string>(attr);
        if (value.find(',') == std::string::npos) {
            if (nocommon) throw XMLBadAttrException(reader, attr, value);
            long_field = tran_field = boost::lexical_cast<T>(value);
        } else {
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

static inline Expansion::Component readSymmetry(const FourierSolver3D* solver, const XMLReader& reader, const std::string& repr) {
    AxisNames* axes = nullptr;
    if (solver->getGeometry()) axes = &solver->getGeometry()->axisNames;
    if (repr == "none" || repr == "NONE" || repr == "None")
        return Expansion::E_UNSPECIFIED;
    else if (repr == "Etran" || repr == "Et" || (axes && repr == "E"+axes->getNameForTran()) ||
                repr == "Hlong" || repr == "Hl" || (axes && repr == "H"+axes->getNameForLong()))
        return Expansion::E_TRAN;
    else if (repr == "Elong" || repr == "El" || (axes && repr == "E"+axes->getNameForLong()) ||
                repr == "Htran" || repr == "Ht" || (axes && repr == "H"+axes->getNameForTran()))
        return Expansion::E_LONG;
    else
        throw XMLBadAttrException(reader, "symmetry", repr, "symmetric field component name (maybe you need to specify the geometry first)");
}

void FourierSolver3D::loadConfiguration(XMLReader& reader, Manager& manager)
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
            std::string sym_tran, sym_long;
            readComaAttr(reader, "symmetry", sym_long, sym_tran, true);
            if (sym_long != "") setSymmetryLong(readSymmetry(this, reader, sym_long));
            if (sym_tran != "") setSymmetryTran(readSymmetry(this, reader, sym_tran));
        } else if (param == "root") {
            readRootDiggerConfig(reader);
        } else if (param == "outer") {
            outdist = reader.requireAttribute<double>("dist");
            reader.requireTagEnd();
        } else
            parseStandardConfiguration(reader, manager);
    }
}



void FourierSolver3D::onInitialize()
{
    Solver::writelog(LOG_DETAIL, "Initializing Fourier3D solver (%1% layers in the stack, interface after %2% layer%3%)",
                               this->stack.size(), this->interface, (this->interface==1)? "" : "s");
    this->setupLayers();
    this->ensureInterface();
    expansion.init();
    initTransfer(expansion);
    this->recompute_coefficients = true;
}


void FourierSolver3D::onInvalidate()
{
    modes.clear();
    expansion.reset();
    transfer.reset();
}


size_t FourierSolver3D::findMode(FourierSolver3D::What what, dcomplex start)
{
    initCalculation();
    std::unique_ptr<RootDigger> root;
    switch (what) {
        case FourierSolver3D::WHAT_WAVELENGTH:
            detlog.axis_arg_name = "lam";
            root = getRootDigger([this](const dcomplex& x) { this->k0 = 2e3*M_PI / x; return transfer->determinant(); });
            break;
        case FourierSolver3D::WHAT_K0:
            detlog.axis_arg_name = "k0";
            root = getRootDigger([this](const dcomplex& x) { this->k0 = x; return transfer->determinant(); });
            break;
        case FourierSolver3D::WHAT_KLONG:
            detlog.axis_arg_name = "klong";
            root = getRootDigger([this](const dcomplex& x) { this->klong = x; return transfer->determinant(); });
            break;
        case FourierSolver3D::WHAT_KTRAN:
            detlog.axis_arg_name = "ktran";
            root = getRootDigger([this](const dcomplex& x) { this->klong = x; return transfer->determinant(); });
            break;
    }
    root->find(start);
    return insertMode();
}


// cvector FourierSolver3D::getReflectedAmplitudes(ExpansionPW3D::Component polarization,
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
// cvector FourierSolver3D::getTransmittedAmplitudes(ExpansionPW3D::Component polarization,
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
// double FourierSolver3D::getReflection(ExpansionPW3D::Component polarization, IncidentDirection incidence)
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
//     auto gamma = transfer->diagonalizer->Gamma(l);
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
// double FourierSolver3D::getTransmission(ExpansionPW3D::Component polarization, IncidentDirection incidence)
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
//     auto gamma = transfer->diagonalizer->Gamma(lt);
//     dcomplex igamma0 = 1. / transfer->diagonalizer->Gamma(li)[idx];
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
const DataVector<const Vec<3,dcomplex>> FourierSolver3D::getE(size_t num, shared_ptr<const MeshD<3>> dst_mesh, InterpolationMethod method)
{
    if (modes.size() <= num) throw NoValue(LightE::NAME);
    assert(transfer);
    if (modes[num].k0 != k0 || modes[num].klong != klong || modes[num].ktran != ktran) {
        k0 = modes[num].k0;
        klong = modes[num].klong;
        ktran = modes[num].ktran;
        transfer->fields_determined = Transfer::DETERMINED_NOTHING;
    }
    return transfer->getFieldE(dst_mesh, method);
}


const DataVector<const Vec<3,dcomplex>> FourierSolver3D::getH(size_t num, shared_ptr<const MeshD<3>> dst_mesh, InterpolationMethod method)
{
    if (modes.size() <= num) throw NoValue(LightH::NAME);
    assert(transfer);
    if (modes[num].k0 != k0 || modes[num].klong != klong || modes[num].ktran != ktran) {
        k0 = modes[num].k0;
        klong = modes[num].klong;
        ktran = modes[num].ktran;
        transfer->fields_determined = Transfer::DETERMINED_NOTHING;
    }
    return transfer->getFieldH(dst_mesh, method);
}


const DataVector<const double> FourierSolver3D::getIntensity(size_t num, shared_ptr<const MeshD<3> > dst_mesh, InterpolationMethod method)
{
    if (modes.size() <= num) throw NoValue(LightMagnitude::NAME);
    if (modes[num].k0 != k0 || modes[num].klong != klong || modes[num].ktran != ktran) {
        k0 = modes[num].k0;
        klong = modes[num].klong;
        ktran = modes[num].ktran;
        transfer->fields_determined = Transfer::DETERMINED_NOTHING;
    }
    return transfer->getFieldMagnitude(modes[num].power, dst_mesh, method);
}
//
//
}}} // namespace
