#include "fourier_reflection_2d.h"
#include "expansion_pw2d.h"

namespace plask { namespace solvers { namespace slab {

FourierReflection2D::FourierReflection2D(const std::string& name): ReflectionSolver<Geometry2DCartesian>(name),
    size(12),
    expansion(this),
    refine(32),
    outNeff(this, &FourierReflection2D::getEffectiveIndex, &FourierReflection2D::nummodes)
{
    detlog.global_prefix = this->getId();
}


void FourierReflection2D::loadConfiguration(XMLReader& reader, Manager& manager)
{
    while (reader.requireTagOrEnd()) {
        std::string param = reader.getNodeName();
        if (param == "expansion") {
            size = reader.getAttribute<size_t>("size", size);
            refine = reader.getAttribute<size_t>("refine", refine);
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
        } else if (param == "pml") {
            pml.factor = reader.getAttribute<dcomplex>("factor", pml.factor);
            pml.size = reader.getAttribute<double>("size", pml.size);
            pml.shift = reader.getAttribute<double>("shift", pml.shift);
            pml.order = reader.getAttribute<double>("order", pml.order);
            reader.requireTagEnd();
        } else if (param == "mode") {
            k0 = 2e3*M_PI / reader.getAttribute<dcomplex>("wavelength", 2e3*M_PI / k0);
            ktran = 2e3*M_PI / reader.getAttribute<dcomplex>("k-tran", ktran);
            klong = 2e3*M_PI / reader.getAttribute<dcomplex>("k-long", klong);
            if (reader.hasAttribute("symmetry")) {
                std::string repr = reader.requireAttribute("symmetry");
                ExpansionPW2D::Component val;
                AxisNames* axes = nullptr;
                if (geometry) axes = &geometry->axisNames;
                if (repr == "none" || repr == "NONE" || repr == "None")
                    val = ExpansionPW2D::E_UNSPECIFIED;
                else if (repr == "Etran" || repr == "Et" || (axes && repr == "E"+axes->getNameForTran()) ||
                         repr == "Hlong" || repr == "Hl" || (axes && repr == "H"+axes->getNameForLong()))
                    val = ExpansionPW2D::E_TRAN;
                else if (repr == "Elong" || repr == "El" || (axes && repr == "E"+axes->getNameForLong()) ||
                         repr == "Htran" || repr == "Ht" || (axes && repr == "H"+axes->getNameForTran()))
                    val = ExpansionPW2D::E_LONG;
                else
                    throw XMLBadAttrException(reader, "symmetry", repr, "symmetric field component name (maybe you need to specify the geometry first)");
                setSymmetry(val);
            }
            if (reader.hasAttribute("polarization")) {
                std::string repr = reader.requireAttribute("polarization");
                ExpansionPW2D::Component val;
                AxisNames* axes = nullptr;
                if (geometry) axes = &geometry->axisNames;
                if (repr == "none" || repr == "NONE" || repr == "None")
                    val = ExpansionPW2D::E_UNSPECIFIED;
                else if (repr == "Etran" || repr == "Et" || (axes && repr == "E"+axes->getNameForTran()) ||
                         repr == "Hlong" || repr == "Hl" || (axes && repr == "H"+axes->getNameForLong()))
                    val = ExpansionPW2D::E_TRAN;
                else if (repr == "Elong" || repr == "El" || (axes && repr == "E"+axes->getNameForLong()) ||
                         repr == "Htran" || repr == "Ht" || (axes && repr == "H"+axes->getNameForTran()))
                    val = ExpansionPW2D::E_LONG;
                else
                    throw XMLBadAttrException(reader, "polarization", repr, "existing field component name (maybe you need to specify the geometry first)");
                setPolarization(val);
            }
        } else if (param == "root") {
            root.tolx = reader.getAttribute<double>("tolx", root.tolx);
            root.tolf_min = reader.getAttribute<double>("tolf-min", root.tolf_min);
            root.tolf_max = reader.getAttribute<double>("tolf-max", root.tolf_max);
            root.maxstep = reader.getAttribute<double>("maxstep", root.maxstep);
            root.maxiter = reader.getAttribute<int>("maxiter", root.maxiter);
            reader.requireTagEnd();
        } else if (param == "mirrors") {
            double R1 = reader.requireAttribute<double>("R1");
            double R2 = reader.requireAttribute<double>("R2");
            mirrors.reset(std::make_pair(R1,R2));
            reader.requireTagEnd();
        } else if (param == "outer") {
            outdist = reader.requireAttribute<double>("dist");
            reader.requireTagEnd();
        } else
            parseStandardConfiguration(reader, manager);
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
        RootDigger(*this, [this](const dcomplex& x) { this->klong = dcomplex(real(x), imag(x)-getMirrorLosses(x)) * this->k0; return this->determinant(); },
                   detlog, root)(neff);
    return insertMode();
}


DataVector<const Tensor3<dcomplex>> FourierReflection2D::getRefractiveIndexProfile(const RectilinearMesh2D& dst_mesh,
                                                                                   InterpolationMethod interp)
{
    initCalculation();
    if (recompute_coefficients) {
        computeCoefficients();
        recompute_coefficients = false;
    }
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
    dcomplex igamma0 = 1. / gamma0;
    if (!expansion.separated) {
        int N = getSize() + 1;
        for (int n = expansion.symmetric? 0 : 1-N; n != N; ++n) {
            size_t iz = expansion.iEz(n), ix = expansion.iEx(n);
            reflected[ix] = reflected[ix] * conj(reflected[ix]) *  gamma0 / gamma[ix];
            reflected[iz] = reflected[iz] * conj(reflected[iz]) * igamma0 * gamma[iz];
        }
    } else {
        if (expansion.polarization == ExpansionPW2D::E_LONG) {
            for (size_t i = 0; i != expansion.matrixSize(); ++i)
                reflected[i] = reflected[i] * conj(reflected[i]) * igamma0 * gamma[i];
        } else {
            for (size_t i = 0; i != expansion.matrixSize(); ++i)
                reflected[i] = reflected[i] * conj(reflected[i]) *  gamma0 / gamma[i];
        }
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
    dcomplex igamma0 = 1. / diagonalizer->Gamma(li)[idx];
    dcomplex gamma0 = gamma[idx] * gamma[idx] * igamma0;
    if (!expansion.separated) {
        int N = getSize() + 1;
        for (int n = expansion.symmetric? 0 : 1-N; n != N; ++n) {
            size_t iz = expansion.iEz(n), ix = expansion.iEx(n);
            transmitted[ix] = transmitted[ix] * conj(transmitted[ix]) *  gamma0 / gamma[ix];
            transmitted[iz] = transmitted[iz] * conj(transmitted[iz]) * igamma0 * gamma[iz];
        }
    } else {
        if (expansion.polarization == ExpansionPW2D::E_LONG) {
            for (size_t i = 0; i != expansion.matrixSize(); ++i)
                transmitted[i] = transmitted[i] * conj(transmitted[i]) * igamma0 * gamma[i];
        } else {
            for (size_t i = 0; i != expansion.matrixSize(); ++i)
                transmitted[i] = transmitted[i] * conj(transmitted[i]) *  gamma0 / gamma[i];
        }
    }

    return sumAmplitutes(transmitted);
}


const DataVector<const Vec<3,dcomplex>> FourierReflection2D::getE(size_t num, const MeshD<2>& dst_mesh, InterpolationMethod method)
{
    if (modes.size() <= num) throw NoValue(LightE::NAME);
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
    if (modes.size() <= num) throw NoValue(LightH::NAME);
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
    if (modes.size() <= num) throw NoValue(LightMagnitude::NAME);
    if (modes[num].k0 != k0 || modes[num].beta != klong || modes[num].ktran != ktran) {
        k0 = modes[num].k0;
        klong = modes[num].beta;
        ktran = modes[num].ktran;
        fields_determined = DETERMINED_NOTHING;
    }
    return getFieldIntensity(modes[num].power, dst_mesh, method);
}


}}} // namespace
