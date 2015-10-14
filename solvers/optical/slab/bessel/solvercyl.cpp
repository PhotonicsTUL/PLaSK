#include "solvercyl.h"

namespace plask { namespace solvers { namespace slab {

    
BesselSolverCyl::BesselSolverCyl(const std::string& name): SlabSolver<SolverWithMesh<Geometry2DCylindrical,OrderedAxis>>(name),
    m(1),
    size(12),
    expansion(this),
    integral_error(1e-6),
    max_itegration_points(1000),
    outWavelength(this, &BesselSolverCyl::getWavelength, &BesselSolverCyl::nummodes),
    outLoss(this, &BesselSolverCyl::getModalLoss,  &BesselSolverCyl::nummodes)
{
    detlog.global_prefix = this->getId();
    detlog.axis_arg_name = "lam";
    pml.dist = 20.;
}


void BesselSolverCyl::loadConfiguration(XMLReader& reader, Manager& manager)
{
    while (reader.requireTagOrEnd()) {
        std::string param = reader.getNodeName();
        if (param == "expansion") {
            size = reader.getAttribute<size_t>("size", size);
            group_layers = reader.getAttribute<bool>("group-layers", group_layers);
            lam0 = reader.getAttribute<double>("lam0");
            always_recompute_gain = reader.getAttribute<bool>("update-gain", always_recompute_gain);
            integral_error = reader.getAttribute<double>("integrals-error", integral_error);
            max_itegration_points = reader.getAttribute<size_t>("integrals-points", max_itegration_points);
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
                writelog(LOG_WARNING, "XML line %d in <vpml>: Attribute 'order' is obsolete, use 'shape' instead", reader.getLineNr());
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
                writelog(LOG_WARNING, "XML line %d in <pml>: Attribute 'order' is obsolete, use 'shape' instead", reader.getLineNr());
                pml.order = reader.requireAttribute<double>("order");
            }
            pml.order = reader.getAttribute<double>("shape", pml.order);
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


void BesselSolverCyl::onInitialize()
{
    this->setupLayers();
    this->ensureInterface();
    Solver::writelog(LOG_DETAIL, "Initializing BesselCyl solver (%1% layers in the stack, interface after %2% layer%3%)",
                               this->stack.size(), this->interface, (this->interface==1)? "" : "s");
    expansion.init();
    this->recompute_integrals = true;
}


void BesselSolverCyl::onInvalidate()
{
    modes.clear();
    expansion.reset();
    transfer.reset();
}


size_t BesselSolverCyl::findMode(dcomplex start, int m)
{
    setM(m);
    initCalculation();
    initTransfer(expansion, false);
    std::unique_ptr<RootDigger> root = getRootDigger([this](const dcomplex& x) { this->setWavelength(x); return transfer->determinant(); });
    root->find(start);
    return insertMode();
}


LazyData<Vec<3,dcomplex>> BesselSolverCyl::getE(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method)
{
    assert(num < modes.size());
    assert(transfer);
    ParamGuard guard(this);
    setLam0(modes[num].lam0);
    setK0(modes[num].k0);
    setM(modes[num].m);
    return transfer->getFieldE(dst_mesh, method);
}


LazyData<Vec<3,dcomplex>> BesselSolverCyl::getH(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method)
{
    assert(num < modes.size());
    assert(transfer);
    ParamGuard guard(this);
    setLam0(modes[num].lam0);
    setK0(modes[num].k0);
    setM(modes[num].m);
    return transfer->getFieldH(dst_mesh, method);
}


LazyData<double> BesselSolverCyl::getMagnitude(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method)
{
    assert(num < modes.size());
    assert(transfer);
    ParamGuard guard(this);
    setLam0(modes[num].lam0);
    setK0(modes[num].k0);
    setM(modes[num].m);
    return transfer->getFieldMagnitude(modes[num].power, dst_mesh, method);
}

#ifndef NDEBUG
cmatrix BesselSolverCyl::epsVmm(size_t layer) {
    initCalculation();
    computeIntegrals();
    return expansion.epsVmm(layer);
}
cmatrix BesselSolverCyl::epsVpp(size_t layer) {
    initCalculation();
    computeIntegrals();
    return expansion.epsVpp(layer);
}
cmatrix BesselSolverCyl::epsTmm(size_t layer) {
    initCalculation();
    computeIntegrals();
    return expansion.epsTmm(layer);
}
cmatrix BesselSolverCyl::epsTpp(size_t layer) {
    initCalculation();
    computeIntegrals();
    return expansion.epsTpp(layer);
}
cmatrix BesselSolverCyl::epsTmp(size_t layer) {
    initCalculation();
    computeIntegrals();
    return expansion.epsTmp(layer);
}
cmatrix BesselSolverCyl::epsTpm(size_t layer) {
    initCalculation();
    computeIntegrals();
    return expansion.epsTpm(layer);
}
cmatrix BesselSolverCyl::epsDm(size_t layer) {
    initCalculation();
    computeIntegrals();
    return expansion.epsDm(layer);
}
cmatrix BesselSolverCyl::epsDp(size_t layer) {
    initCalculation();
    computeIntegrals();
    return expansion.epsDp(layer);
}

cmatrix BesselSolverCyl::muVmm() {
    initCalculation();
    computeIntegrals();
    return expansion.muVmm();
}
cmatrix BesselSolverCyl::muVpp() {
    initCalculation();
    computeIntegrals();
    return expansion.muVpp();
}
cmatrix BesselSolverCyl::muTmm() {
    initCalculation();
    computeIntegrals();
    return expansion.muTmm();
}
cmatrix BesselSolverCyl::muTpp() {
    initCalculation();
    computeIntegrals();
    return expansion.muTpp();
}
cmatrix BesselSolverCyl::muTmp() {
    initCalculation();
    computeIntegrals();
    return expansion.muTmp();
}
cmatrix BesselSolverCyl::muTpm() {
    initCalculation();
    computeIntegrals();
    return expansion.muTpm();
}
cmatrix BesselSolverCyl::muDm() {
    initCalculation();
    computeIntegrals();
    return expansion.muDm();
}
cmatrix BesselSolverCyl::muDp() {
    initCalculation();
    computeIntegrals();
    return expansion.muDp();
}
#endif


}}} // # namespace plask::solvers::slab
