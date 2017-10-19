#include "solver2d.h"
#include "expansion2d.h"

namespace plask { namespace optical { namespace slab {

FourierSolver2D::FourierSolver2D(const std::string& name): SlabSolver<SolverOver<Geometry2DCartesian>>(name),
    beta(0.), ktran(0.),
    symmetry(Expansion::E_UNSPECIFIED),
    polarization(Expansion::E_UNSPECIFIED),
    size(12),
    dct(2),
    expansion(this),
    refine(32),
    oversampling(1.),
    outNeff(this, &FourierSolver2D::getEffectiveIndex, &FourierSolver2D::nummodes)
{
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
            oversampling = reader.getAttribute<double>("oversampling", oversampling);
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
        } else if (param == "pml") {
            pml.factor = reader.getAttribute<dcomplex>("factor", pml.factor);
            pml.size = reader.getAttribute<double>("size", pml.size);
            pml.dist = reader.getAttribute<double>("dist", pml.dist);
            if (reader.hasAttribute("order")) { //TODO Remove in the future
                writelog(LOG_WARNING, "XML line {:d} in <pml>: Attribute 'order' is obsolete, use 'shape' instead", reader.getLineNr());
                pml.order = reader.requireAttribute<double>("order");
            }
            pml.order = reader.getAttribute<double>("shape", pml.order);
            reader.requireTagEnd();
        } else if (param == "mode") {
            emission = reader.enumAttribute<Emission>("emission")
                                .value("undefined", EMISSION_UNSPECIFIED)
                                .value("top", EMISSION_TOP)
                                .value("bottom", EMISSION_BOTTOM)
                       .get(emission);
            k0 = 2e3*M_PI / reader.getAttribute<dcomplex>("wavelength", 2e3*M_PI / k0);
            ktran = reader.getAttribute<dcomplex>("k-tran", ktran);
            beta = reader.getAttribute<dcomplex>("k-long", beta);
            if (reader.hasAttribute("symmetry")) {
                std::string repr = reader.requireAttribute("symmetry");
                Expansion::Component val;
                AxisNames* axes = nullptr;
                if (geometry) axes = &geometry->axisNames;
                if (repr == "none" || repr == "NONE" || repr == "None")
                    val = Expansion::E_UNSPECIFIED;
                else if (repr == "Etran" || repr == "Et" || (axes && repr == "E"+axes->getNameForTran()) ||
                         repr == "Hlong" || repr == "Hl" || (axes && repr == "H"+axes->getNameForLong()))
                    val = Expansion::E_TRAN;
                else if (repr == "Elong" || repr == "El" || (axes && repr == "E"+axes->getNameForLong()) ||
                         repr == "Htran" || repr == "Ht" || (axes && repr == "H"+axes->getNameForTran()))
                    val = Expansion::E_LONG;
                else
                    throw XMLBadAttrException(reader, "symmetry", repr, "symmetric field component name (maybe you need to specify the geometry first)");
                setSymmetry(val);
            }
            if (reader.hasAttribute("polarization")) {
                std::string repr = reader.requireAttribute("polarization");
                Expansion::Component val;
                AxisNames* axes = nullptr;
                if (geometry) axes = &geometry->axisNames;
                if (repr == "none" || repr == "NONE" || repr == "None")
                    val = Expansion::E_UNSPECIFIED;
                else if (repr == "TE" || repr == "Etran" || repr == "Et" || (axes && repr == "E"+axes->getNameForTran()) ||
                         repr == "Hlong" || repr == "Hl" || (axes && repr == "H"+axes->getNameForLong()))
                    val = Expansion::E_TRAN;
                else if (repr == "TM" || repr == "Elong" || repr == "El" || (axes && repr == "E"+axes->getNameForLong()) ||
                         repr == "Htran" || repr == "Ht" || (axes && repr == "H"+axes->getNameForTran()))
                    val = Expansion::E_LONG;
                else
                    throw XMLBadAttrException(reader, "polarization", repr, "existing field component name (maybe you need to specify the geometry first)");
                setPolarization(val);
            }
            reader.requireTagEnd();
        } else if (param == "root") {
            readRootDiggerConfig(reader);
        } else if (param == "mirrors") {
            double R1 = reader.requireAttribute<double>("R1");
            double R2 = reader.requireAttribute<double>("R2");
            mirrors.reset(std::make_pair(R1,R2));
            reader.requireTagEnd();
        } else
            parseStandardConfiguration(reader, manager);
    }
}


void FourierSolver2D::onInitialize()
{
    this->setupLayers();
    if (this->interface == size_t(-1))
        Solver::writelog(LOG_DETAIL, "Initializing Fourier2D solver ({0} layers in the stack)",
                                     this->stack.size());
    else
        Solver::writelog(LOG_DETAIL, "Initializing Fourier2D solver ({0} layers in the stack, interface after {1} layer{2})",
                                     this->stack.size(), this->interface, (this->interface==1)? "" : "s");
    setExpansionDefaults();
    expansion.init();
    this->recompute_integrals = true;
}


void FourierSolver2D::onInvalidate()
{
    modes.clear();
    expansion.reset();
    transfer.reset();
}


size_t FourierSolver2D::findMode(FourierSolver2D::What what, dcomplex start)
{
    expansion.setSymmetry(symmetry);
    expansion.setPolarization(polarization);
    expansion.setLam0(this->lam0);
    initCalculation();
    ensureInterface();
    if (!transfer) initTransfer(expansion, false);
    std::unique_ptr<RootDigger> root;
    switch (what) {
        case FourierSolver2D::WHAT_WAVELENGTH:
            expansion.setBeta(beta);
            expansion.setKtran(ktran);
            root = getRootDigger([this](const dcomplex& x) { expansion.setK0(2e3*M_PI/x); return transfer->determinant(); }, "lam");
            break;
        case FourierSolver2D::WHAT_K0:
            expansion.setBeta(beta);
            expansion.setKtran(ktran);
            root = getRootDigger([this](const dcomplex& x) { expansion.setK0(x); return transfer->determinant(); }, "k0");
            break;
        case FourierSolver2D::WHAT_NEFF:
            if (expansion.separated())
                throw Exception("{0}: Cannot search for effective index with polarization separation", getId());
            expansion.setK0(k0);
            expansion.setKtran(ktran);
            clearFields();
            root = getRootDigger([this](const dcomplex& x) {
                    expansion.setBeta(x * expansion.k0);
                    return transfer->determinant();
                }, "neff");
            break;
        case FourierSolver2D::WHAT_KTRAN:
            if (expansion.symmetric())
                throw Exception("{0}: Cannot search for transverse wavevector with symmetry", getId());
            expansion.setK0(k0);
            expansion.setBeta(beta);
            root = getRootDigger([this](const dcomplex& x) { expansion.setKtran(x); return transfer->determinant(); }, "ktran");
            break;
    }
    root->find(start);
    return insertMode();
}


cvector FourierSolver2D::getReflectedAmplitudes(Expansion::Component polarization,
                                                Transfer::IncidentDirection incidence)
{
    size_t idx;

    double kt = real(expansion.ktran), kl = real(expansion.beta);

    if (!expansion.initialized && expansion.beta == 0.) expansion.polarization = polarization;
    initCalculation();
    if (!transfer) initTransfer(expansion, true);

    if (!expansion.periodic)
        throw NotImplemented(getId(), "Reflection coefficient can be computed only for periodic geometries");

    size_t n = (incidence == Transfer::INCIDENCE_BOTTOM)? 0 : stack.size()-1;
    size_t l = stack[n];

    cvector reflected = transfer->getReflectionVector(incidentVector(polarization, &idx), incidence).claim();

    if (!expansion.diagonalQE(l))
        Solver::writelog(LOG_WARNING, "{0} layer should be uniform to reliably compute reflection coefficient",
                                      (incidence == Transfer::INCIDENCE_BOTTOM)? "Bottom" : "Top");

    auto gamma = transfer->diagonalizer->Gamma(l);
    dcomplex gamma0 = gamma[idx];
    dcomplex igamma0 = 1. / gamma0;
    double incident = ((polarization==Expansion::E_LONG)? kl : kt);
    incident = 1. / (1. + incident*incident * real(igamma0*conj(igamma0)));

    if (!expansion.separated()) {
        double b = 2*M_PI / (expansion.right-expansion.left) * (expansion.symmetric()? 0.5 : 1.0);
        int N = getSize() + 1;
        for (int n = expansion.symmetric()? 0 : 1-N; n != N; ++n) {
            size_t iz = expansion.iEz(n), ix = expansion.iEx(n);
            //assert(abs(gamma[ix] - gamma[iz]) < 1e3*SMALL);
            double g = n*b-kt;
            dcomplex Ez = reflected[iz], Ex = reflected[ix];
            dcomplex S = (gamma[iz]*gamma[iz]+kl*kl) * Ez*conj(Ez) + (gamma[ix]*gamma[ix]+g*g) * Ex*conj(Ex) -
                         kl * g * (Ez*conj(Ex) + conj(Ez)*Ex);
            reflected[ix] = incident * real(igamma0 / (0.5*(gamma[ix]+gamma[iz])) * S);
            reflected[iz] = 0.;
        }
    } else {
        if (expansion.polarization == Expansion::E_LONG) {
            for (size_t i = 0; i != expansion.matrixSize(); ++i)
                reflected[i] = reflected[i]*conj(reflected[i]) * igamma0 * gamma[i];
        } else {
            for (size_t i = 0; i != expansion.matrixSize(); ++i)
                reflected[i] = incident * reflected[i]*conj(reflected[i]) * (gamma0*gamma0+kt*kt) / gamma[i] * igamma0;
        }
    }

    return reflected;
}


cvector FourierSolver2D::getTransmittedAmplitudes(Expansion::Component polarization,
                                                  Transfer::IncidentDirection incidence)
{
    size_t idx;

    double kt = real(ktran), kl = real(beta);

    if (!expansion.initialized && beta == 0.) expansion.polarization = polarization;
    initCalculation();
    if (!transfer) initTransfer(expansion, true);

    if (!expansion.periodic)
        throw NotImplemented(getId(), "Transmission coefficient can be computed only for periodic geometries");

    size_t ni = (incidence == Transfer::INCIDENCE_TOP)? stack.size()-1 : 0;
    size_t nt = stack.size()-1-ni;
    size_t li = stack[ni], lt = stack[nt];

    cvector transmitted = transfer->getTransmissionVector(incidentVector(polarization, &idx), incidence).claim();

    if (!expansion.diagonalQE(lt))
        Solver::writelog(LOG_WARNING, "{0} layer should be uniform to reliably compute transmission coefficient",
                                      (incidence == Transfer::INCIDENCE_TOP)? "Bottom" : "Top");
    if (!expansion.diagonalQE(li))
        Solver::writelog(LOG_WARNING, "{0} layer should be uniform to reliably compute transmission coefficient",
                                     (incidence == Transfer::INCIDENCE_TOP)? "Top" : "Bottom");

    auto gamma = transfer->diagonalizer->Gamma(lt);
    dcomplex gamma0 = gamma[idx];
    dcomplex igamma0 = 1. / transfer->diagonalizer->Gamma(li)[idx];
    double incident = ((polarization==Expansion::E_LONG)? kl : kt);
    incident = 1. / (1. + incident*incident * real(igamma0*conj(igamma0)));
    if (!expansion.separated()) {
        double b = 2*M_PI / (expansion.right-expansion.left) * (expansion.symmetric()? 0.5 : 1.0);
        int N = getSize() + 1;
        for (int n = expansion.symmetric()? 0 : 1-N; n != N; ++n) {
            size_t iz = expansion.iEz(n), ix = expansion.iEx(n);
            //assert(abs(gamma[ix] - gamma[iz]) < 1e3*SMALL);
            double g = n*b-kt;
            dcomplex Ez = transmitted[iz], Ex = transmitted[ix];
            dcomplex S = (gamma[iz]*gamma[iz]+kl*kl) * Ez*conj(Ez) + (gamma[ix]*gamma[ix]+g*g) * Ex*conj(Ex) -
                         kl * g * (Ez*conj(Ex) + conj(Ez)*Ex);
            transmitted[ix] = incident * real(igamma0 / (0.5*(gamma[ix]+gamma[iz])) * S);
            transmitted[iz] = 0.;
        }
    } else {
        if (expansion.polarization == Expansion::E_LONG) {
            for (size_t i = 0; i != expansion.matrixSize(); ++i)
                transmitted[i] = transmitted[i] * conj(transmitted[i]) * igamma0 * gamma[i];
        } else {
            for (size_t i = 0; i != expansion.matrixSize(); ++i)
                transmitted[i] = incident * transmitted[i] * conj(transmitted[i]) *  (gamma0*gamma0+kt*kt) / gamma[i] * igamma0;
        }
    }

    return transmitted;
}


cvector FourierSolver2D::getReflectedCoefficients(Expansion::Component polarization,
                                                  Transfer::IncidentDirection incidence)
{
    size_t idx;

    if (!expansion.initialized && expansion.beta == 0.) expansion.polarization = polarization;
    initCalculation();
    if (!transfer) initTransfer(expansion, true);

    if (!expansion.periodic)
        throw NotImplemented(getId(), "Reflection coefficients can be computed only for periodic geometries");

    return transfer->getReflectionVector(incidentVector(polarization, &idx), incidence);
}

cvector FourierSolver2D::getTransmittedCoefficients(Expansion::Component polarization,
                                                    Transfer::IncidentDirection incidence)
{
    size_t idx;

    if (!expansion.initialized && beta == 0.) expansion.polarization = polarization;
    initCalculation();
    if (!transfer) initTransfer(expansion, true);

    if (!expansion.periodic)
        throw NotImplemented(getId(), "Transmission coefficients can be computed only for periodic geometries");

    return transfer->getTransmissionVector(incidentVector(polarization, &idx), incidence);
}


LazyData<Vec<3,dcomplex>> FourierSolver2D::getE(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method)
{
    if (num >= modes.size()) throw BadInput(this->getId()+".outLightE", "Mode {0} has not been computed", num);
    assert(transfer);
    applyMode(modes[num]);
    return transfer->getFieldE(modes[num].power, dst_mesh, method);
}


LazyData<Vec<3,dcomplex>> FourierSolver2D::getH(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method)
{
    if (num >= modes.size()) throw BadInput(this->getId()+".outLightH", "Mode {0} has not been computed", num);
    assert(transfer);
    applyMode(modes[num]);
    return transfer->getFieldH(modes[num].power, dst_mesh, method);
}


LazyData<double> FourierSolver2D::getMagnitude(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method)
{
    if (num >= modes.size()) throw BadInput(this->getId()+".outLightMagnitude", "Mode {0} has not been computed", num);
    assert(transfer);
    applyMode(modes[num]);
    return transfer->getFieldMagnitude(modes[num].power, dst_mesh, method);
}


}}} // namespace
