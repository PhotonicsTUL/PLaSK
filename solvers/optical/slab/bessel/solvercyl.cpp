#include <memory>

#include "solvercyl.h"

namespace plask { namespace optical { namespace slab {


BesselSolverCyl::BesselSolverCyl(const std::string& name):
    SlabSolver<SolverWithMesh<Geometry2DCylindrical,MeshAxis>>(name),
    domain(DOMAIN_INFINITE),
    m(1),
    size(12),
    rule(RULE_INVERSE),
    kscale(10.),
    kmethod(WAVEVECTORS_UNIFORM),
    integral_error(1e-6),
    max_integration_points(1000),
    outLoss(this, &BesselSolverCyl::getModalLoss,  &BesselSolverCyl::nummodes)
{
    pml.dist = 20.;
    pml.size = 0.;
    this->writelog(LOG_WARNING, "This is an EXPERIMENTAL solver! Calculation results may not be reliable!");
}


void BesselSolverCyl::loadConfiguration(XMLReader& reader, Manager& manager)
{
    while (reader.requireTagOrEnd()) {
        std::string param = reader.getNodeName();
        if (param == "expansion") {
            domain = reader.enumAttribute<BesselDomain>("domain")
                .value("finite", DOMAIN_FINITE)
                .value("infinite", DOMAIN_INFINITE)
                .get(domain);
            size = reader.getAttribute<size_t>("size", size);
            group_layers = reader.getAttribute<bool>("group-layers", group_layers);
            lam0 = reader.getAttribute<double>("lam0", NAN);
            always_recompute_gain = reader.getAttribute<bool>("update-gain", always_recompute_gain);
            max_temp_diff = reader.getAttribute<double>("temp-diff", max_temp_diff);
            temp_dist = reader.getAttribute<double>("temp-dist", temp_dist);
            temp_layer = reader.getAttribute<double>("temp-layer", temp_layer);
            integral_error = reader.getAttribute<double>("integrals-error", integral_error);
            max_integration_points = reader.getAttribute<size_t>("integrals-points", max_integration_points);
            kscale = reader.getAttribute<double>("k-scale", kscale);
            kmethod = reader.enumAttribute<InfiniteWavevectors>("k-method")
                .value("uniform", WAVEVECTORS_UNIFORM)
                // .value("legendre", WAVEVECTORS_LEGENDRE)
                .value("laguerre", WAVEVECTORS_LAGUERRE)
                .value("manual", WAVEVECTORS_MANUAL)
                .get(kmethod);
            if (reader.hasAttribute("k-list")) {
                klist.clear();
                for (auto val: boost::tokenizer<boost::char_separator<char>>(reader.requireAttribute("k-list"),
                                                                             boost::char_separator<char>(" ,;\t\n"))) {
                    try {
                        double val = boost::lexical_cast<double>(val);
                        klist.push_back(val);
                    } catch (boost::bad_lexical_cast&) {
                        throw XMLException(reader, format("Value '{0}' cannot be converted to float", val));
                    }
                }
            }
            rule = reader.enumAttribute<Rule>("rule")
                .value("inverse", RULE_INVERSE)
                .value("direct", RULE_DIRECT)
                .get(rule);
            reader.requireTagEnd();
        } else if (param == "mode") {
            emission = reader.enumAttribute<Emission>("emission")
                                .value("undefined", EMISSION_UNSPECIFIED)
                                .value("top", EMISSION_TOP)
                                .value("bottom", EMISSION_BOTTOM)
                       .get(emission);
            if (reader.hasAttribute("wavelength")) { //TODO Remove in the future
                writelog(LOG_WARNING, "XML line {:d} in <mode>: Attribute 'wavelength' is obsolete, use 'lam' instead", reader.getLineNr());
                if (reader.hasAttribute("lam")) throw XMLConflictingAttributesException(reader, "wavelength", "lam");
                k0 = 2e3*PI / reader.requireAttribute<dcomplex>("wavelength");
            }
            if (reader.hasAttribute("lam")) k0 = 2e3*PI / reader.requireAttribute<dcomplex>("lam");
            reader.requireTagEnd();
        } else if (param == "interface") {
            if (reader.hasAttribute("index")) {
                throw XMLException(reader, "Setting interface by layer index is not supported anymore (set it by object or position)");
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
                if (reader.hasAttribute("shape")) throw XMLConflictingAttributesException(reader, "order", "shape");
                vpml.order = reader.requireAttribute<double>("order");
            }
            vpml.order = reader.getAttribute<double>("shape", vpml.order);
            reader.requireTagEnd();
        } else if (param == "transfer") {
            transfer_method = reader.enumAttribute<Transfer::Method>("method")
                .value("auto", Transfer::METHOD_AUTO)
                .value("reflection", Transfer::METHOD_REFLECTION_ADMITTANCE)
                .value("reflection-admittance", Transfer::METHOD_REFLECTION_ADMITTANCE)
                .value("reflection-impedance", Transfer::METHOD_REFLECTION_IMPEDANCE)
                .value("admittance", Transfer::METHOD_ADMITTANCE)
                .value("impedance", Transfer::METHOD_IMPEDANCE)
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
        } else if (param == "root") {
            readRootDiggerConfig(reader);
        } else
            parseStandardConfiguration(reader, manager);
    }
}


void BesselSolverCyl::onInitialize()
{
    this->setupLayers();
    if (this->interface == -1)
        Solver::writelog(LOG_DETAIL, "Initializing BesselCyl solver ({0} layers in the stack)",
                                     this->stack.size());
    else
        Solver::writelog(LOG_DETAIL, "Initializing BesselCyl solver ({0} layers in the stack, interface after {1} layer{2})",
                                     this->stack.size(), this->interface, (this->interface==1)? "" : "s");
    switch (domain) {
        case DOMAIN_FINITE:
            expansion.reset(new ExpansionBesselFini(this));
            break;
        case DOMAIN_INFINITE:
            expansion.reset(new ExpansionBesselInfini(this));
            break;
        default:
            assert(0);
    }
    setExpansionDefaults();
    expansion->init1();
    this->recompute_integrals = true;
}


void BesselSolverCyl::onInvalidate()
{
    modes.clear();
    expansion->reset();
    transfer.reset();
}


size_t BesselSolverCyl::findMode(dcomplex start, int m)
{
    Solver::initCalculation();
    ensureInterface();
    expansion->setLam0(this->lam0);
    expansion->setM(m);
    initTransfer(*expansion, false);
    std::unique_ptr<RootDigger> root = getRootDigger([this](const dcomplex& x) {
        if (isnan(x)) throw ComputationError(this->getId(), "'lam' converged to NaN");
        expansion->setK0(2e3*PI/x); return transfer->determinant();
    }, "lam");
    root->find(start);
    return insertMode();
}


LazyData<Vec<3,dcomplex>> BesselSolverCyl::getE(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method)
{
    if (num >= modes.size()) throw BadInput(this->getId()+".outLightE", "Mode {0} has not been computed", num);
    assert(transfer);
    applyMode(modes[num]);
    return transfer->getFieldE(modes[num].power, dst_mesh, method);
}


LazyData<Vec<3,dcomplex>> BesselSolverCyl::getH(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method)
{
    if (num >= modes.size()) throw BadInput(this->getId()+".outLightH", "Mode {0} has not been computed", num);
    assert(transfer);
    applyMode(modes[num]);
    return transfer->getFieldH(modes[num].power, dst_mesh, method);
}


LazyData<double> BesselSolverCyl::getMagnitude(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method)
{
    if (num >= modes.size()) throw BadInput(this->getId()+".outLightMagnitude", "Mode {0} has not been computed", num);
    assert(transfer);
    applyMode(modes[num]);
    return transfer->getFieldMagnitude(modes[num].power, dst_mesh, method);
}


double BesselSolverCyl::getWavelength(size_t n) {
    if (n >= modes.size()) throw NoValue(ModeWavelength::NAME);
    return real(2e3*M_PI / modes[n].k0);
}


#ifndef NDEBUG
cmatrix BesselSolverCyl::epsV_k(size_t layer) {
    Solver::initCalculation();
    computeIntegrals();
    return expansion->epsV_k(layer);
}
cmatrix BesselSolverCyl::epsTss(size_t layer) {
    Solver::initCalculation();
    computeIntegrals();
    return expansion->epsTss(layer);
}
cmatrix BesselSolverCyl::epsTpp(size_t layer) {
    Solver::initCalculation();
    computeIntegrals();
    return expansion->epsTpp(layer);
}
cmatrix BesselSolverCyl::epsTsp(size_t layer) {
    Solver::initCalculation();
    computeIntegrals();
    return expansion->epsTsp(layer);
}
cmatrix BesselSolverCyl::epsTps(size_t layer) {
    Solver::initCalculation();
    computeIntegrals();
    return expansion->epsTps(layer);
}

cmatrix BesselSolverCyl::muV_k() {
    Solver::initCalculation();
    if (auto finite_expansion = dynamic_cast<ExpansionBesselFini*>(expansion.get())) {
        computeIntegrals();
        return finite_expansion->muV_k();
    } else {
        return cmatrix();
    }
}
cmatrix BesselSolverCyl::muTss() {
    Solver::initCalculation();
    if (auto finite_expansion = dynamic_cast<ExpansionBesselFini*>(expansion.get())) {
        computeIntegrals();
        return finite_expansion->muTss();
    } else {
        return cmatrix();
    }
}
cmatrix BesselSolverCyl::muTsp() {
    Solver::initCalculation();
    if (auto finite_expansion = dynamic_cast<ExpansionBesselFini*>(expansion.get())) {
        computeIntegrals();
        return finite_expansion->muTsp();
    } else {
        return cmatrix();
    }
}
cmatrix BesselSolverCyl::muTps() {
    Solver::initCalculation();
    if (auto finite_expansion = dynamic_cast<ExpansionBesselFini*>(expansion.get())) {
        computeIntegrals();
        return finite_expansion->muTps();
    } else {
        return cmatrix();
    }
}
cmatrix BesselSolverCyl::muTpp() {
    Solver::initCalculation();
    if (auto finite_expansion = dynamic_cast<ExpansionBesselFini*>(expansion.get())) {
        computeIntegrals();
        return finite_expansion->muTpp();
    } else {
        return cmatrix();
    }
}
#endif


}}} // # namespace plask::optical::slab
