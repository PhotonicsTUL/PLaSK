#include <memory>

#include "solverfdcyl.hpp"

namespace plask { namespace optical { namespace slab {


LinesSolverCyl::LinesSolverCyl(const std::string& name):
    SlabSolver<SolverWithMesh<Geometry2DCylindrical,MeshAxis>>(name),
    m(1), expansion(this),
    outLoss(this, &LinesSolverCyl::getModalLoss,  &LinesSolverCyl::nummodes)
{
    pml.dist = 20.;
    pml.size = 5.;
    pml.factor = dcomplex(1., -2.);
    this->writelog(LOG_WARNING, "This is an EXPERIMENTAL solver! Calculation results may not be reliable!");
}


void LinesSolverCyl::loadConfiguration(XMLReader& reader, Manager& manager)
{
    while (reader.requireTagOrEnd()) {
        std::string param = reader.getNodeName();
        if (param == "expansion") {
            group_layers = reader.getAttribute<bool>("group-layers", group_layers);
            lam0 = reader.getAttribute<double>("lam0", NAN);
            always_recompute_gain = reader.getAttribute<bool>("update-gain", always_recompute_gain);
            max_temp_diff = reader.getAttribute<double>("temp-diff", max_temp_diff);
            temp_dist = reader.getAttribute<double>("temp-dist", temp_dist);
            temp_layer = reader.getAttribute<double>("temp-layer", temp_layer);
            reader.requireTagEnd();
        } else if (param == "mode") {
            emission = reader.enumAttribute<Emission>("emission")
                                .value("undefined", EMISSION_UNSPECIFIED)
                                .value("top", EMISSION_TOP)
                                .value("bottom", EMISSION_BOTTOM)
                       .get(emission);
            k0 = 2e3*PI / reader.getAttribute<dcomplex>("wavelength", 2e3*PI / k0);
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
            parseCommonSlabConfiguration(reader, manager);
    }
}


void LinesSolverCyl::onInitialize()
{
    this->setupLayers();
    if (!this->mesh) throw NoMeshException(this->getId());

    if (this->interface == -1)
        Solver::writelog(LOG_DETAIL, "Initializing LinesCyl solver ({0} layers in the stack)",
                                     this->stack.size());
    else
        Solver::writelog(LOG_DETAIL, "Initializing LinesCyl solver ({0} layers in the stack, interface after {1} layer{2})",
                                     this->stack.size(), this->interface, (this->interface==1)? "" : "s");
    setExpansionDefaults();
    expansion.init();
    this->recompute_integrals = true;
}


void LinesSolverCyl::onInvalidate()
{
    modes.clear();
    expansion.reset();
    transfer.reset();
}


size_t LinesSolverCyl::findMode(dcomplex start, int m)
{
    Solver::initCalculation();
    ensureInterface();
    expansion.setLam0(this->lam0);
    expansion.setM(m);
    initTransfer(expansion, false);
    std::unique_ptr<RootDigger> root = getRootDigger([this](const dcomplex& x) {
        if (isnan(x)) throw ComputationError(this->getId(), "'lam' converged to NaN");
        expansion.setK0(2e3*PI/x); return transfer->determinant();
    }, "lam");
    root->find(start);
    return insertMode();
}


double LinesSolverCyl::getWavelength(size_t n) {
    if (n >= modes.size()) throw NoValue(ModeWavelength::NAME);
    return real(2e3*M_PI / modes[n].k0);
}


}}} // # namespace plask::optical::slab
