#include <memory>

#include "solvercyl.h"

namespace plask { namespace optical { namespace slab {


BesselSolverCyl::BesselSolverCyl(const std::string& name):
    SlabSolver<SolverWithMesh<Geometry2DCylindrical,OrderedAxis>>(name),
    domain(DOMAIN_INFINITE),
    m(1),
    size(12),
    kscale(10.),
    kmethod(WAVEVECTORS_UNIFORM),
    integral_error(1e-6),
    max_integration_points(1000),
    outWavelength(this, &BesselSolverCyl::getWavelength, &BesselSolverCyl::nummodes),
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
                    } catch (boost::bad_lexical_cast) {
                        throw XMLException(reader, format("Value '{0}' cannot be converted to float", val));
                    }
                }
            }
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
        } else if (param == "root") {
            readRootDiggerConfig(reader);
        } else
            parseStandardConfiguration(reader, manager);
    }
}


void BesselSolverCyl::onInitialize()
{
    this->setupLayers();
    if (this->interface == size_t(-1))
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
    initCalculation();
    ensureInterface();
    expansion->setLam0(this->lam0);
    expansion->setM(m);
    initTransfer(*expansion, false);
    std::unique_ptr<RootDigger> root = getRootDigger([this](const dcomplex& x) { expansion->setK0(2e3*M_PI/x); return transfer->determinant(); }, "lam");
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

#ifndef NDEBUG
cmatrix BesselSolverCyl::epsVmm(size_t layer) {
    initCalculation();
    computeIntegrals();
    return expansion->epsVmm(layer);
}
cmatrix BesselSolverCyl::epsVpp(size_t layer) {
    initCalculation();
    computeIntegrals();
    return expansion->epsVpp(layer);
}
cmatrix BesselSolverCyl::epsTmm(size_t layer) {
    initCalculation();
    computeIntegrals();
    return expansion->epsTmm(layer);
}
cmatrix BesselSolverCyl::epsTpp(size_t layer) {
    initCalculation();
    computeIntegrals();
    return expansion->epsTpp(layer);
}
cmatrix BesselSolverCyl::epsTmp(size_t layer) {
    initCalculation();
    computeIntegrals();
    return expansion->epsTmp(layer);
}
cmatrix BesselSolverCyl::epsTpm(size_t layer) {
    initCalculation();
    computeIntegrals();
    return expansion->epsTpm(layer);
}
cmatrix BesselSolverCyl::epsDm(size_t layer) {
    initCalculation();
    computeIntegrals();
    return expansion->epsDm(layer);
}
cmatrix BesselSolverCyl::epsDp(size_t layer) {
    initCalculation();
    computeIntegrals();
    return expansion->epsDp(layer);
}

// cmatrix BesselSolverCyl::muVmm() {
//     initCalculation();
//     computeIntegrals();
//     return expansion->muVmm();
// }
// cmatrix BesselSolverCyl::muVpp() {
//     initCalculation();
//     computeIntegrals();
//     return expansion->muVpp();
// }
// cmatrix BesselSolverCyl::muTmm() {
//     initCalculation();
//     computeIntegrals();
//     return expansion->muTmm();
// }
// cmatrix BesselSolverCyl::muTpp() {
//     initCalculation();
//     computeIntegrals();
//     return expansion->muTpp();
// }
// cmatrix BesselSolverCyl::muTmp() {
//     initCalculation();
//     computeIntegrals();
//     return expansion->muTmp();
// }
// cmatrix BesselSolverCyl::muTpm() {
//     initCalculation();
//     computeIntegrals();
//     return expansion->muTpm();
// }
// cmatrix BesselSolverCyl::muDm() {
//     initCalculation();
//     computeIntegrals();
//     return expansion->muDm();
// }
// cmatrix BesselSolverCyl::muDp() {
//     initCalculation();
//     computeIntegrals();
//     return expansion->muDp();
// }
#endif


}}} // # namespace plask::optical::slab
