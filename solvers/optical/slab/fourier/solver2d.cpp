#include "solver2d.h"
#include "expansion2d.h"
#include "../diagonalizer.h"

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
            max_temp_diff = reader.getAttribute<double>("temp-diff", max_temp_diff);
            temp_dist = reader.getAttribute<double>("temp-dist", temp_dist);
            temp_layer = reader.getAttribute<double>("temp-layer", temp_layer);
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
        } else if (param == "mode") {
            emission = reader.enumAttribute<Emission>("emission")
                                .value("undefined", EMISSION_UNSPECIFIED)
                                .value("top", EMISSION_TOP)
                                .value("bottom", EMISSION_BOTTOM)
                       .get(emission);
            k0 = 2e3*PI / reader.getAttribute<dcomplex>("wavelength", 2e3*PI / k0);
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
    if (this->interface == -1)
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
    Solver::initCalculation();
    ensureInterface();
    if (!transfer) initTransfer(expansion, false);
    std::unique_ptr<RootDigger> root;
    switch (what) {
        case FourierSolver2D::WHAT_WAVELENGTH:
            expansion.setBeta(beta);
            expansion.setKtran(ktran);
            root = getRootDigger([this](const dcomplex& x) {
                if (isnan(x)) throw ComputationError(this->getId(), "'lam' converged to NaN");
                expansion.setK0(2e3*PI/x); return transfer->determinant();
            }, "lam");
            break;
        case FourierSolver2D::WHAT_K0:
            expansion.setBeta(beta);
            expansion.setKtran(ktran);
            root = getRootDigger([this](const dcomplex& x) {
                if (isnan(x)) throw ComputationError(this->getId(), "'k0' converged to NaN");
                expansion.setK0(x); return transfer->determinant();
            }, "k0");
            break;
        case FourierSolver2D::WHAT_NEFF:
            if (expansion.separated())
                throw Exception("{0}: Cannot search for effective index with polarization separation", getId());
            expansion.setK0(k0);
            expansion.setKtran(ktran);
            clearFields();
            root = getRootDigger([this](const dcomplex& x) {
                if (isnan(x)) throw ComputationError(this->getId(), "'neff' converged to NaN");
                expansion.setBeta(x * expansion.k0); return transfer->determinant();
            }, "neff");
            break;
        case FourierSolver2D::WHAT_KTRAN:
            if (expansion.symmetric())
                throw Exception("{0}: Cannot search for transverse wavevector with symmetry", getId());
            expansion.setK0(k0);
            expansion.setBeta(beta);
            root = getRootDigger([this](const dcomplex& x) {
                if (isnan(x)) throw ComputationError(this->getId(), "'ktran' converged to NaN");
                expansion.setKtran(x); return transfer->determinant();
            }, "ktran");
            break;
    }
    root->find(start);
    return insertMode();
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


size_t FourierSolver2D::initIncidence(Transfer::IncidentDirection side, Expansion::Component polarization, dcomplex lam)
{
    bool changed = Solver::initCalculation() || setExpansionDefaults(isnan(lam));
    if (!isnan(lam)) {
        dcomplex k0 = 2e3*M_PI / lam;
        if (!is_zero(k0 - expansion.getK0())) {
            expansion.setK0(k0);
            changed = true;
        }
    }

    if (polarization == Expansion::E_UNSPECIFIED)
        throw BadInput(getId(), "Unspecified incident polarization for reflectivity computation");
    if (expansion.symmetric() && expansion.symmetry != polarization)
        throw BadInput(getId(), "Current symmetry is inconsistent with the specified incident polarization");
    if (expansion.separated())
        expansion.polarization = polarization;

    size_t layer = stack[(side == Transfer::INCIDENCE_BOTTOM)? 0 : stack.size()-1];
    if (!transfer) {
        initTransfer(expansion, true);
        changed = true;
    }
    if (changed) {
        transfer->initDiagonalization();
        transfer->diagonalizer->diagonalizeLayer(layer);
    } else if (!transfer->diagonalizer->isDiagonalized(layer))
        transfer->diagonalizer->diagonalizeLayer(layer);
    return layer;
}

cvector FourierSolver2D::incidentVector(Transfer::IncidentDirection side, Expansion::Component polarization, dcomplex lam)
{
    size_t layer = initIncidence(side, polarization, lam);

    size_t idx;
    if (expansion.separated()) idx = expansion.iE(0);
    else idx = (polarization == Expansion::E_TRAN)? expansion.iEx(0) : expansion.iEz(0);
    cvector incident(expansion.matrixSize(), 0.);
    incident[idx] = (polarization == Expansion::E_TRAN)? 1. : -1.;

    return transfer->diagonalizer->invTE(layer) * incident;
}


cvector FourierSolver2D::incidentGaussian(Transfer::IncidentDirection side, Expansion::Component polarization, double sigma, double center, dcomplex lam)
{
    size_t layer = initIncidence(side, polarization, lam);

    double b = 2.*PI / (expansion.right-expansion.left) * (expansion.symmetric()? 0.5 : 1.0);
    dcomplex d = I * b * (center - expansion.left);
    double c2 = - 0.5 * sigma*sigma * b*b;

    cvector incident(expansion.matrixSize(), 0.);
    for (int i = -int(size); i <= int(size); ++i) {
        size_t idx;
        if (expansion.separated()) idx = expansion.iE(i);
        else idx = (polarization == Expansion::E_TRAN)? expansion.iEx(i) : expansion.iEz(i);
        dcomplex val = exp(c2 * double(i*i) - d*double(i));
        incident[idx] = (polarization == Expansion::E_TRAN)? val : -val;
    }

    return transfer->diagonalizer->invTE(layer) * incident;
}


}}} // namespace
