#include "solvercyl.h"

namespace plask { namespace solvers { namespace slab {

    
BesselSolverCyl::BesselSolverCyl(const std::string& name): SlabSolver<Geometry2DCylindrical>(name),
    size(12),
    expansion(this),
    outWavelength(this, &BesselSolverCyl::getWavelength, &BesselSolverCyl::nummodes),
    outLoss(this, &BesselSolverCyl::getModalLoss,  &BesselSolverCyl::nummodes)
{
    detlog.global_prefix = this->getId();
    detlog.axis_arg_name = "lam";
}


void BesselSolverCyl::loadConfiguration(XMLReader& reader, Manager& manager)
{
    while (reader.requireTagOrEnd()) {
        std::string param = reader.getNodeName();
        if (param == "expansion") {
            size = reader.getAttribute<size_t>("size", size);
            group_layers = reader.getAttribute<bool>("group-layers", group_layers);
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
            vpml.shift = reader.getAttribute<double>("shift", vpml.shift);
            vpml.order = reader.getAttribute<double>("order", vpml.order);
            reader.requireTagEnd();
        } else if (param == "transfer") {
            transfer_method = reader.enumAttribute<Transfer::Method>("method")
                .value("auto", Transfer::METHOD_AUTO)
                .value("reflection", Transfer::METHOD_REFLECTION)
                .value("admittance", Transfer::METHOD_ADMITTANCE)
                .get(transfer_method);
            reader.requireTagEnd();
//         } else if (param == "pml") {
//             pml.factor = reader.getAttribute<dcomplex>("factor", pml.factor);
//             pml.size = reader.getAttribute<double>("size", pml.size);
//             pml.shift = reader.getAttribute<double>("shift", pml.shift);
//             pml.order = reader.getAttribute<double>("order", pml.order);
//             reader.requireTagEnd();
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
    this->recompute_integrals = true;
    initCalculation();
    initTransfer(expansion, false);
    std::unique_ptr<RootDigger> root = getRootDigger([this](const dcomplex& x) { this->k0 = 2e3*M_PI / x; return transfer->determinant(); });
    root->find(start);
    return insertMode();
}


const DataVector<const Vec<3,dcomplex>> BesselSolverCyl::getE(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method)
{
}


const DataVector<const Vec<3,dcomplex>> BesselSolverCyl::getH(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method)
{
}


const DataVector<const double> BesselSolverCyl::getIntensity(size_t num, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method)
{
}


}}} // # namespace plask::solvers::slab
