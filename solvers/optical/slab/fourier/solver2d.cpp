#include "solver2d.h"
#include "expansion2d.h"

namespace plask { namespace solvers { namespace slab {

FourierSolver2D::FourierSolver2D(const std::string& name): SlabSolver<Geometry2DCartesian>(name),
    size(12),
    expansion(this),
    refine(32),
    outNeff(this, &FourierSolver2D::getEffectiveIndex, &FourierSolver2D::nummodes)
{
    detlog.global_prefix = this->getId();
    smooth = 0.00025;
}


void FourierSolver2D::loadConfiguration(XMLReader& reader, Manager& manager)
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
            ktran = reader.getAttribute<dcomplex>("k-tran", ktran);
            klong = reader.getAttribute<dcomplex>("k-long", klong);
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
            readRootDiggerConfig(reader);
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


void FourierSolver2D::onInitialize()
{
    this->setupLayers();
    this->ensureInterface();
    Solver::writelog(LOG_DETAIL, "Initializing Fourier2D solver (%1% layers in the stack, interface after %2% layer%3%)",
                               this->stack.size(), this->interface, (this->interface==1)? "" : "s");
    expansion.init();
    initTransfer(expansion);
    this->recompute_coefficients = true;
}


void FourierSolver2D::onInvalidate()
{
    modes.clear();
    expansion.reset();
    transfer.reset();
}


size_t FourierSolver2D::findMode(dcomplex neff)
{
    if (expansion.polarization != ExpansionPW2D::E_UNSPECIFIED)
        throw Exception("%1%: Cannot search for effective index with polarization separation", getId());
    klong = neff * k0;
    if (klong == 0.) klong = 1e-12;
    initCalculation();
    detlog.axis_arg_name = "neff";
    auto root = getRootDigger(
        [this](const dcomplex& x) {
            this->klong = dcomplex(real(x), imag(x)-getMirrorLosses(x)) * this->k0;
            return transfer->determinant();
        }
    );
    klong =  k0 * root->find(neff);
    return insertMode();
}


cvector FourierSolver2D::getReflectedAmplitudes(ExpansionPW2D::Component polarization,
                                                Transfer::IncidentDirection incidence, size_t* savidx)
{
    if (!expansion.initialized && klong == 0.) expansion.polarization = polarization;
    if (transfer) transfer->fields_determined = Transfer::DETERMINED_NOTHING;
    initCalculation();
    return transfer->getReflectionVector(incidentVector(polarization, savidx), incidence);
}


cvector FourierSolver2D::getTransmittedAmplitudes(ExpansionPW2D::Component polarization,
                                                  Transfer::IncidentDirection incidence, size_t* savidx)
{
    if (!expansion.initialized && klong == 0.) expansion.polarization = polarization;
    if (transfer) transfer->fields_determined = Transfer::DETERMINED_NOTHING;
    initCalculation();
    return transfer->getTransmissionVector(incidentVector(polarization, savidx), incidence);
}


double FourierSolver2D::getReflection(ExpansionPW2D::Component polarization, Transfer::IncidentDirection incidence)
{
    size_t idx;
    cvector reflected = getReflectedAmplitudes(polarization, incidence, &idx).claim();

    if (!expansion.periodic)
        throw NotImplemented(getId(), "Reflection coefficient can be computed only for periodic geometries");

    size_t n = (incidence == Transfer::INCIDENCE_BOTTOM)? 0 : stack.size()-1;
    size_t l = stack[n];
    if (!expansion.diagonalQE(l))
        Solver::writelog(LOG_WARNING, "%1% layer should be uniform to reliably compute reflection coefficient",
                                      (incidence == Transfer::INCIDENCE_BOTTOM)? "Bottom" : "Top");

    auto gamma = transfer->diagonalizer->Gamma(l);
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


double FourierSolver2D::getTransmission(ExpansionPW2D::Component polarization, Transfer::IncidentDirection incidence)
{
    size_t idx;
    cvector transmitted = getTransmittedAmplitudes(polarization, incidence, &idx).claim();

    if (!expansion.periodic)
        throw NotImplemented(getId(), "Transmission coefficient can be computed only for periodic geometries");

    size_t ni = (incidence == Transfer::INCIDENCE_TOP)? stack.size()-1 : 0;
    size_t nt = stack.size()-1-ni;
    size_t li = stack[ni], lt = stack[nt];
    if (!expansion.diagonalQE(lt))
        Solver::writelog(LOG_WARNING, "%1% layer should be uniform to reliably compute transmission coefficient",
                                      (incidence == Transfer::INCIDENCE_TOP)? "Bottom" : "Top");
    if (!expansion.diagonalQE(li))
        Solver::writelog(LOG_WARNING, "%1% layer should be uniform to reliably compute transmission coefficient",
                                     (incidence == Transfer::INCIDENCE_TOP)? "Top" : "Bottom");

    auto gamma = transfer->diagonalizer->Gamma(lt);
    dcomplex igamma0 = 1. / transfer->diagonalizer->Gamma(li)[idx];
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


const DataVector<const Vec<3,dcomplex>> FourierSolver2D::getE(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method)
{
    if (modes.size() <= num) throw NoValue(LightE::NAME);
    assert(transfer);
    if (modes[num].k0 != k0 || modes[num].beta != klong || modes[num].ktran != ktran) {
        k0 = modes[num].k0;
        klong = modes[num].beta;
        ktran = modes[num].ktran;
        transfer->fields_determined = Transfer::DETERMINED_NOTHING;
    }
    return transfer->getFieldE(dst_mesh, method);
}


const DataVector<const Vec<3,dcomplex>> FourierSolver2D::getH(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method)
{
    if (modes.size() <= num) throw NoValue(LightH::NAME);
    assert(transfer);
    if (modes[num].k0 != k0 || modes[num].beta != klong || modes[num].ktran != ktran) {
        k0 = modes[num].k0;
        klong = modes[num].beta;
        ktran = modes[num].ktran;
        transfer->fields_determined = Transfer::DETERMINED_NOTHING;
    }
    return transfer->getFieldH(dst_mesh, method);
}


const DataVector<const double> FourierSolver2D::getIntensity(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method)
{
    if (modes.size() <= num) throw NoValue(LightMagnitude::NAME);
    if (modes[num].k0 != k0 || modes[num].beta != klong || modes[num].ktran != ktran) {
        k0 = modes[num].k0;
        klong = modes[num].beta;
        ktran = modes[num].ktran;
        transfer->fields_determined = Transfer::DETERMINED_NOTHING;
    }
    return transfer->getFieldMagnitude(modes[num].power, dst_mesh, method);
}


}}} // namespace
