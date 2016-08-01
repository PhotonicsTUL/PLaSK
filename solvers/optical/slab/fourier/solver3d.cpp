#include "solver3d.h"
#include "expansion3d.h"

namespace plask { namespace solvers { namespace slab {

FourierSolver3D::FourierSolver3D(const std::string& name): SlabSolver<SolverOver<Geometry3D>>(name),
    size_long(12), size_tran(12),
    klong(0.), ktran(0.),
    symmetry_long(Expansion::E_UNSPECIFIED), symmetry_tran(Expansion::E_UNSPECIFIED),
    dct(2),
    expansion(this),
    refine_long(16), refine_tran(16),
    oversampling_long(1.), oversampling_tran(1.)
{
    pml_tran.factor = {1., -2.};
    pml_long.factor = {1., -2.};
    smooth = 0.00025;
}

static inline PML readPML(XMLReader& reader) {
    PML pml;
    pml.factor = reader.getAttribute<dcomplex>("factor", pml.factor);
    pml.size = reader.getAttribute<double>("size", pml.size);
    pml.dist = reader.getAttribute<double>("dist", pml.dist);
    if (reader.hasAttribute("order")) { //TODO Remove in the future
        writelog(LOG_WARNING, "XML line {:d} in <pml>: Attribute 'order' is obsolete, use 'shape' instead", reader.getLineNr());
        pml.order = reader.requireAttribute<double>("order");
    }
    pml.order = reader.getAttribute<double>("shape", pml.order);
    return pml;
}

static inline void updatePML(PML& pml, XMLReader& reader) {
    pml.factor = reader.getAttribute<dcomplex>("factor", pml.factor);
    pml.size = reader.getAttribute<double>("size", pml.size);
    pml.dist = reader.getAttribute<double>("dist", pml.dist);
    if (reader.hasAttribute("order")) { //TODO Remove in the future
        writelog(LOG_WARNING, "XML line {:d} in <pml>: Attribute 'order' is obsolete, use 'shape' instead", reader.getLineNr());
        pml.order = reader.requireAttribute<double>("order");
    }
    pml.order = reader.getAttribute<double>("shape", pml.order);
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
            tran_field = boost::lexical_cast<T>(values.second);
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
            readComaAttr(reader, "oversampling", oversampling_long, oversampling_tran);
            smooth = reader.getAttribute<double>("smooth", smooth);
            int dc = reader.getAttribute<int>("dct", dct);
            if (dc != 1 && dc != 2)
                throw XMLBadAttrException(reader, "dct", boost::lexical_cast<std::string>(dc), "\"1\" or \"2\"");
            dct = dc;
            group_layers = reader.getAttribute<bool>("group-layers", group_layers);
            lam0 = reader.getAttribute<double>("lam0", NAN);
            always_recompute_gain = reader.getAttribute<bool>("update-gain", always_recompute_gain);
            reader.requireTagEnd();
        } else if (param == "interface") {
            if (reader.hasAttribute("index")) {
                if (reader.hasAttribute("position")) throw XMLConflictingAttributesException(reader, "index", "position");
                if (reader.hasAttribute("object")) throw XMLConflictingAttributesException(reader, "index", "object");
                if (reader.hasAttribute("path")) throw XMLConflictingAttributesException(reader, "index", "path");
                setInterface(reader.requireAttribute<size_t>("index"));
            } else if (reader.hasAttribute("position")) {
                if (reader.hasAttribute("object")) throw XMLConflictingAttributesException(reader, "index", "object");
                if (reader.hasAttribute("path")) throw XMLConflictingAttributesException(reader, "index", "path");
                setInterfaceAt(reader.requireAttribute<double>("position"));
            } else if (reader.hasAttribute("object")) {
                auto object = manager.requireGeometryObject<GeometryObject>(reader.requireAttribute("object"));
                PathHints path; if (auto pathattr = reader.getAttribute("path")) path = manager.requirePathHints(*pathattr);
                setInterfaceOn(object, path);
            } else if (reader.hasAttribute("path")) {
                throw XMLUnexpectedAttrException(reader, "path");
            }
            reader.requireTagEnd();
        } else if (param == "vpml") {
            vpml.factor = reader.getAttribute<dcomplex>("factor", vpml.factor);
            vpml.size = reader.getAttribute<double>("size", vpml.size);
            vpml.dist = reader.getAttribute<double>("dist", vpml.dist);
            if (reader.hasAttribute("order")) { //TODO Remove in the future
                writelog(LOG_WARNING, "XML line {:d} in <vpml>: Attribute 'order' is obsolete, use 'shape' instead", reader.getLineNr());
                vpml.order = reader.requireAttribute<double>("order");
            }
            vpml.order = reader.getAttribute<double>("shape", vpml.order);
            reader.requireTagEnd();
        } else if (param == "transfer") {
            transfer_method = reader.enumAttribute<Transfer::Method>("method")
                .value("auto", Transfer::METHOD_AUTO)
                .value("reflection", Transfer::METHOD_REFLECTION)
                .value("admittance", Transfer::METHOD_ADMITTANCE)
                .get(transfer_method);
            reader.requireTagEnd();
        } else if (param == "pmls") {
            pml_long = pml_tran = readPML(reader);
            while (reader.requireTagOrEnd()) {
                std::string node = reader.getNodeName();
                if (node == "long") {
                    updatePML(pml_long, reader);
                    reader.requireTagEnd();
                } else if (node == "tran") {
                    updatePML(pml_tran, reader);
                    reader.requireTagEnd();
                } else throw XMLUnexpectedElementException(reader, "<tran>, <long>, or </pmls>", node);
            }
        } else if (param == "mode") {
            emission = reader.enumAttribute<Emission>("emission")
                                .value("undefined", EMISSION_UNSPECIFIED)
                                .value("top", EMISSION_TOP)
                                .value("bottom", EMISSION_BOTTOM)
                       .get(emission);
            k0 = 2e3*M_PI / reader.getAttribute<dcomplex>("wavelength", 2e3*M_PI / k0);
            ktran = reader.getAttribute<dcomplex>("k-tran", ktran);
            klong = reader.getAttribute<dcomplex>("k-long", klong);
            std::string sym_tran, sym_long;
            readComaAttr(reader, "symmetry", sym_long, sym_tran, true);
            if (sym_long != "") setSymmetryLong(readSymmetry(this, reader, sym_long));
            if (sym_tran != "") setSymmetryTran(readSymmetry(this, reader, sym_tran));
            reader.requireTagEnd();
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
    this->setupLayers();
    this->ensureInterface();
    Solver::writelog(LOG_DETAIL, "Initializing Fourier3D solver ({0} layers in the stack, interface after {1} layer{2})",
                               this->stack.size(), this->interface, (this->interface==1)? "" : "s");
    setExpansionDefaults();
    expansion.init();
    this->recompute_integrals = true;
}


void FourierSolver3D::onInvalidate()
{
    modes.clear();
    expansion.reset();
    transfer.reset();
}


size_t FourierSolver3D::findMode(FourierSolver3D::What what, dcomplex start)
{
    expansion.setSymmetryLong(symmetry_long);
    expansion.setSymmetryTran(symmetry_tran);
    expansion.setLam0(this->lam0);
    initCalculation();
    initTransfer(expansion, false);
    std::unique_ptr<RootDigger> root;
    switch (what) {
        case FourierSolver3D::WHAT_WAVELENGTH:
            expansion.setKlong(klong);
            expansion.setKtran(ktran);
            root = getRootDigger([this](const dcomplex& x) { expansion.setK0(2e3*M_PI/x); return transfer->determinant(); }, "lam");
            break;
        case FourierSolver3D::WHAT_K0:
            expansion.setKlong(klong);
            expansion.setKtran(ktran);
            root = getRootDigger([this](const dcomplex& x) { expansion.setK0(x); return transfer->determinant(); }, "k0");
            break;
        case FourierSolver3D::WHAT_KLONG:
            expansion.setK0(this->k0);
            expansion.setKtran(ktran);
            transfer->fields_determined = Transfer::DETERMINED_NOTHING;
            root = getRootDigger([this](const dcomplex& x) { expansion.klong = x; return transfer->determinant(); }, "klong");
            break;
        case FourierSolver3D::WHAT_KTRAN:
            expansion.setK0(this->k0);
            expansion.setKlong(klong);
            transfer->fields_determined = Transfer::DETERMINED_NOTHING;
            root = getRootDigger([this](const dcomplex& x) { expansion.klong = x; return transfer->determinant(); }, "ktran");
            break;
    }
    root->find(start);
    return insertMode();
}


cvector FourierSolver3D::getReflectedAmplitudes(Expansion::Component polarization,
                                                Transfer::IncidentDirection incidence,
                                                size_t* savidx)
{
    if (transfer) transfer->fields_determined = Transfer::DETERMINED_NOTHING;
    initCalculation();
    initTransfer(expansion, true);
    return transfer->getReflectionVector(incidentVector(polarization, savidx), incidence);
}


cvector FourierSolver3D::getTransmittedAmplitudes(Expansion::Component polarization,
                                                  Transfer::IncidentDirection incidence,
                                                  size_t* savidx)
{
    if (transfer) transfer->fields_determined = Transfer::DETERMINED_NOTHING;
    initCalculation();
    initTransfer(expansion, true);
    return transfer->getTransmissionVector(incidentVector(polarization, savidx), incidence);
}


double FourierSolver3D::getReflection(Expansion::Component polarization, Transfer::IncidentDirection incidence)
{
    double kt = real(ktran), kl = real(klong);

    size_t idx;
    cvector reflected = getReflectedAmplitudes(polarization, incidence, &idx).claim();

    if (!expansion.periodic_long || !expansion.periodic_tran)
        throw NotImplemented(getId(), "Reflection coefficient can be computed only for periodic geometries");

    size_t n = (incidence == Transfer::INCIDENCE_BOTTOM)? 0 : stack.size()-1;
    size_t l = stack[n];
    if (!expansion.diagonalQE(l))
        writelog(LOG_WARNING, "{0} layer should be uniform to reliably compute reflection coefficient",
                              (incidence == Transfer::INCIDENCE_BOTTOM)? "Bottom" : "Top");

    auto gamma = transfer->diagonalizer->Gamma(l);
    dcomplex igamma0 = 1. / gamma[idx];

    double incident = ((polarization==Expansion::E_LONG)? kl : kt);
    incident = 1. / (1. + incident*incident * real(igamma0*conj(igamma0)));

    double bl = 2*M_PI / (expansion.front-expansion.back) * (expansion.symmetric_long()? 0.5 : 1.0),
           bt = 2*M_PI / (expansion.right-expansion.left) * (expansion.symmetric_tran()? 0.5 : 1.0);

    double result = 0.;

    int ordl = getLongSize(), ordt = getTranSize();
    for (int t = -ordt; t <= ordt; ++t) {
        for (int l = -ordl; l <= ordl; ++l) {
            size_t ix = expansion.iEx(l,t), iy = expansion.iEy(l,t);
            //assert(abs(gamma[ix] - gamma[iy]) < 1e3*SMALL);
            double gx = l*bl-kl, gy = t*bt-kt;
            dcomplex Ex = reflected[ix], Ey = reflected[iy];
            dcomplex S = (gamma[ix]*gamma[ix]+gx*gx) * Ex*conj(Ex) + (gamma[iy]*gamma[iy]+gy*gy) * Ey*conj(Ey) +
                         gx * gy * (Ex*conj(Ey) + conj(Ex)*Ey);
            result += incident * real(igamma0 / (0.5*(gamma[ix]+gamma[iy])) * S);
        }
    }

    return result;
}


double FourierSolver3D::getTransmission(Expansion::Component polarization, Transfer::IncidentDirection incidence)
{
    double kt = real(ktran), kl = real(klong);

    size_t idx;
    cvector transmitted = getTransmittedAmplitudes(polarization, incidence, &idx).claim();

    if (!expansion.periodic_long || !expansion.periodic_tran)
        throw NotImplemented(getId(), "Reflection coefficient can be computed only for periodic geometries");

    size_t ni = (incidence == Transfer::INCIDENCE_TOP)? stack.size()-1 : 0;
    size_t nt = stack.size()-1-ni;
    size_t li = stack[ni], lt = stack[nt];
    if (!expansion.diagonalQE(lt))
        Solver::writelog(LOG_WARNING, "{0} layer should be uniform to reliably compute transmission coefficient",
                                      (incidence == Transfer::INCIDENCE_TOP)? "Bottom" : "Top");
    if (!expansion.diagonalQE(li))
        Solver::writelog(LOG_WARNING, "{0} layer should be uniform to reliably compute transmission coefficient",
                                     (incidence == Transfer::INCIDENCE_TOP)? "Top" : "Bottom");

    // we multiply all fields by gt / gi
    auto gamma = transfer->diagonalizer->Gamma(lt);
    dcomplex igamma0 = 1. / transfer->diagonalizer->Gamma(li)[idx];

    double incident = ((polarization==Expansion::E_LONG)? kl : kt);
    incident = 1. / (1. + incident*incident * real(igamma0*conj(igamma0)));

    double bl = 2*M_PI / (expansion.front-expansion.back) * (expansion.symmetric_long()? 0.5 : 1.0),
           bt = 2*M_PI / (expansion.right-expansion.left) * (expansion.symmetric_tran()? 0.5 : 1.0);

    double result = 0.;

    int ordl = getLongSize(), ordt = getTranSize();
    for (int t = -ordt; t <= ordt; ++t) {
        for (int l = -ordl; l <= ordl; ++l) {
            size_t ix = expansion.iEx(l,t), iy = expansion.iEy(l,t);
            //assert(abs(gamma[ix] - gamma[iy]) < 1e3*SMALL);
            double gx = l*bl-kl, gy = t*bt-kt;
            dcomplex Ex = transmitted[ix], Ey = transmitted[iy];
            dcomplex S = (gamma[ix]*gamma[ix]+gx*gx) * Ex*conj(Ex) + (gamma[iy]*gamma[iy]+gy*gy) * Ey*conj(Ey) +
                         gx * gy * (Ex*conj(Ey) + conj(Ex)*Ey);
            result += incident * real(igamma0 / (0.5*(gamma[ix]+gamma[iy])) * S);
        }
    }

    return result;
}


LazyData<Vec<3,dcomplex>> FourierSolver3D::getE(size_t num, shared_ptr<const MeshD<3>> dst_mesh, InterpolationMethod method)
{
    if (num >= modes.size()) throw BadInput(this->getId()+".outElectricField", "Mode {0} has not been computed", num);
    assert(transfer);
    applyMode(modes[num]);
    return transfer->getFieldE(modes[num].power, dst_mesh, method);
}


LazyData<Vec<3,dcomplex>> FourierSolver3D::getH(size_t num, shared_ptr<const MeshD<3>> dst_mesh, InterpolationMethod method)
{
    if (num >= modes.size()) throw BadInput(this->getId()+".outMagneticField", "Mode {0} has not been computed", num);
    assert(transfer);
    applyMode(modes[num]);
    return transfer->getFieldH(modes[num].power, dst_mesh, method);
}


LazyData<double> FourierSolver3D::getMagnitude(size_t num, shared_ptr<const MeshD<3> > dst_mesh, InterpolationMethod method)
{
    if (num >= modes.size()) throw BadInput(this->getId()+".outLightMagnitude", "Mode {0} has not been computed", num);
    assert(transfer);
    applyMode(modes[num]);
    return transfer->getFieldMagnitude(modes[num].power, dst_mesh, method);
}

}}} // namespace
